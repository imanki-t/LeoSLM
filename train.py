"""
train.py  —  LeoSLM Aether  (Kaggle TPU v5e-8)
================================================

ROOT CAUSE OF PREVIOUS CRASH:
  RuntimeError: TPU initialization failed:
  Expected 8 worker addresses, got 1.

  xmp.spawn() uses ProcessPoolExecutor which registers workers one-by-one.
  PJRT on v5e-8 requires ALL 8 workers to register simultaneously.
  First process tried to init with only 1/8 addresses → PJRT killed it.

FIX:
  torch_xla.launch() — the official modern PJRT-native API.
  Handles simultaneous worker registration on all TPU versions including v5e.

  OLD (broken on v5e): xmp.spawn(_mp_fn, args=(flags,))
  NEW (correct):       torch_xla.launch(_mp_fn, args=(flags,))
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path

os.environ.setdefault("PJRT_DEVICE",                  "TPU")
os.environ.setdefault("XLA_TENSOR_ALLOCATOR_MAXSIZE", "1000000000")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",         "3")
os.environ.setdefault("GRPC_VERBOSITY",               "ERROR")
os.environ.pop("XLA_FLAGS",    None)
os.environ.pop("XLA_USE_BF16", None)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

# ── XLA imports ───────────────────────────────────────────────────────────────
try:
    import torch_xla
    import torch_xla.core.xla_model              as xm
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    pl = None

    class _XM:
        def xla_device(self):             return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def optimizer_step(self, o, **k): torch.nn.utils.clip_grad_norm_(list(o.param_groups[0]["params"]), 1.0); o.step()
        def mark_step(self):              pass
        def get_ordinal(self):            return 0
        def xrt_world_size(self):         return 1
        def master_print(self, *a, **k): print(*a, **k)
        def rendezvous(self, t):          pass
        def is_master_ordinal(self):      return True
        def save(self, obj, path, **k):   torch.save(obj, path)

    xm = _XM()
    print("[train] torch_xla not found — CPU/GPU single-device fallback")

try:
    from transformers.optimization import Adafactor
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "-q"], check=False)
    from transformers.optimization import Adafactor

from model    import LeoSLM, LeoConfig, LEO_IDENTITY
from training import LeoLoss, GRPOTrainer, AgenticGRPO
from data     import LeoDataset


# ── Phase configuration ───────────────────────────────────────────────────────
PHASE_CFG = {
    1: dict(epochs=2, ctx= 4096, tau=0.50, lr=1e-3, warmup=500, mdm=0.0, desc="AR warmup (4k ctx)"),
    2: dict(epochs=2, ctx= 8192, tau=0.45, lr=8e-4, warmup=300, mdm=0.5, desc="Diffusion warmup (8k ctx)"),
    3: dict(epochs=2, ctx=16384, tau=0.40, lr=5e-4, warmup=200, mdm=0.5, desc="Full MoE+ECT (16k ctx)"),
    4: dict(epochs=2, ctx=32768, tau=0.35, lr=3e-4, warmup=200, mdm=0.5, desc="SFT alignment (32k ctx)"),
    5: dict(epochs=1, ctx=32768, tau=0.35, lr=1e-4, warmup=100, mdm=0.3, desc="Factuality DPO (32k ctx)"),
    6: dict(epochs=2, ctx=32768, tau=0.30, lr=5e-5, warmup=100, mdm=0.0, desc="GRPO Think RL (32k ctx)"),
    7: dict(epochs=2, ctx=32768, tau=0.30, lr=3e-5, warmup=100, mdm=0.0, desc="Agentic SFT (32k ctx)"),
    8: dict(epochs=2, ctx=32768, tau=0.28, lr=1e-5, warmup= 50, mdm=0.0, desc="Agentic RL (32k ctx)"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_lr(step, warmup, total, hi, lo):
    if total <= 0:
        return hi
    if step < warmup:
        return hi * step / max(1, warmup)
    f = (step - warmup) / max(1, total - warmup)
    return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * f))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


def save_ckpt(model, step, phase, loss, path):
    """Save checkpoint from rank-0 only. xm.save handles XLA→CPU transfer."""
    if not xm.is_master_ordinal():
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw   = model.module if hasattr(model, "module") else model
    state = {k: v.cpu() for k, v in raw.state_dict().items()}
    tmp   = path + ".tmp"
    xm.save({"model": state, "step": step, "phase": phase, "loss": loss}, tmp)
    os.replace(tmp, path)
    xm.master_print(f"   ✓ saved {Path(path).name}  step={step}  loss={loss:.4f}")


def load_ckpt(model, path):
    if not Path(path).exists():
        return 0, 1
    c   = torch.load(path, map_location="cpu")
    raw = model.module if hasattr(model, "module") else model
    st  = {k.replace("_orig_mod.", "").replace("module.", ""): v
           for k, v in c.get("model", c).items()}
    missing, unexpected = raw.load_state_dict(st, strict=False)
    if missing:
        xm.master_print(f"   Warn: {len(missing)} missing keys on resume")
    if unexpected:
        xm.master_print(f"   Warn: {len(unexpected)} unexpected keys (ignored)")
    xm.master_print(
        f"   Resumed {Path(path).name}  "
        f"step={c.get('step', 0)}  phase={c.get('phase', 1)}"
    )
    return c.get("step", 0), c.get("phase", 1)


# ── Per-phase training loop ───────────────────────────────────────────────────

def run_phase(phase, rank, model, optimizer, dataset, loss_fn,
              device, grad_accum, save_every, ckpt_dir,
              start_step=0, smoke=False):

    pc     = PHASE_CFG[phase]
    epochs = 1 if smoke else pc["epochs"]
    lr_hi  = pc["lr"]
    lr_lo  = lr_hi * 0.1

    xm.master_print(f"\n{'='*60}")
    xm.master_print(f"  PHASE {phase} — {pc['desc']}")
    xm.master_print(f"  ctx={pc['ctx']:,} | tau={pc['tau']} | lr={lr_hi} | epochs={epochs}")
    xm.master_print(f"  rank={rank} | world={xm.xrt_world_size()}")
    xm.master_print(f"{'='*60}")

    raw        = model.module if hasattr(model, "module") else model
    world_size = xm.xrt_world_size()

    dataset.set_seq_len(pc["ctx"])
    raw.cfg.uncertainty_thresh = pc["tau"]
    raw.freeze_phase(phase)
    loss_fn.set_lambda_mdm(pc["mdm"])

    batch_size = 2 if phase == 5 else 1

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        shuffle     = (sampler is None),
        num_workers = 0,       # must be 0 on Kaggle TPU
        drop_last   = True,
        pin_memory  = False,
    )
    # MpDeviceLoader: moves data to TPU device, calls mark_step each batch
    if XLA_AVAILABLE and pl is not None:
        loader = pl.MpDeviceLoader(loader, device)

    # Phase-specific trainers
    dpo = grpo = agentic_grpo = None
    if phase == 5:
        from training import FactualityDPO
        dpo = FactualityDPO(raw.cfg, beta=0.1)
    elif phase == 6:
        grpo = GRPOTrainer(
            cfg            = raw.cfg,
            model          = model,
            optimizer      = optimizer,
            think_start_id = raw.cfg.think_start_id,
            think_end_id   = raw.cfg.think_end_id,
            idk_id         = raw.cfg.idk_id,
        )
    elif phase == 8:
        agentic_grpo = AgenticGRPO(cfg=raw.cfg, model=model, optimizer=optimizer)

    total_steps = max(1, len(dataset) // (batch_size * max(world_size, 1))) * epochs
    step  = start_step
    best  = float("inf")
    t0    = time.time()

    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        ep_loss = 0.0
        nb      = 0

        for batch in loader:
            ids = batch["input_ids"].to(device)
            set_lr(optimizer, cosine_lr(step, pc["warmup"], total_steps, lr_hi, lr_lo))

            # ── Forward + loss ────────────────────────────────────────────────
            if phase == 5 and dpo is not None:
                out    = model(ids)
                U_mean = out["uncertainty"].mean(dim=1)
                lo, hi = (0, 1) if U_mean[0] <= U_mean[1] else (1, 0)
                co = {k: v[lo:lo+1] for k, v in out.items()
                      if isinstance(v, torch.Tensor) and v.shape[0] == 2}
                ro = {k: v[hi:hi+1] for k, v in out.items()
                      if isinstance(v, torch.Tensor) and v.shape[0] == 2}
                loss, metrics = dpo(co, ro, ids[lo:lo+1], ids[hi:hi+1])

            elif phase == 6 and grpo is not None:
                metrics = grpo.grpo_step(batch, device)
                loss    = metrics["total"]

            elif phase == 8 and agentic_grpo is not None:
                metrics = agentic_grpo.agentic_grpo_step(batch, device)
                loss    = metrics["total"]

            else:
                out           = model(ids)
                loss, metrics = loss_fn(out, ids, model=raw, phase=phase)

            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                # all_reduce gradients across all 8 chips + optimizer step + mark_step
                xm.optimizer_step(optimizer)
                optimizer.zero_grad()

            nb      += 1
            step    += 1
            lv       = loss.detach().item() if isinstance(loss, torch.Tensor) else float(loss)
            ep_loss += lv

            # ── Logging ───────────────────────────────────────────────────────
            if step % 10 == 0:
                m     = {k: (v.item() if isinstance(v, torch.Tensor) else v)
                         for k, v in metrics.items()}
                extra = "".join(
                    f" {k}={m[k]:.3f}"
                    for k in ("l_mtp", "l_acgi", "l_msra", "reward", "l_idk", "l_ect")
                    if k in m
                )
                xm.master_print(
                    f"  p{phase} e{epoch+1} s{step:6d} | "
                    f"loss={m.get('total', lv):.4f} "
                    f"ar={m.get('l_ar', 0):.4f}"
                    f"{extra} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                    f"{time.time() - t0:.0f}s"
                )

            # ── Checkpoint ────────────────────────────────────────────────────
            if step % save_every == 0:
                save_ckpt(model, step, phase, lv,
                          f"{ckpt_dir}/phase{phase}_step{step}.pt")
                save_ckpt(model, step, phase, lv,
                          f"{ckpt_dir}/latest.pt")

            if smoke and step >= start_step + 50:
                xm.master_print("  Smoke test: 50 steps complete ✓")
                return step

        avg = ep_loss / max(nb, 1)
        xm.master_print(f"  Epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}")
        if avg < best:
            best = avg
            save_ckpt(model, step, phase, avg,
                      f"{ckpt_dir}/best_phase{phase}.pt")

    return step


# ── Worker function — one per TPU chip ───────────────────────────────────────

def _mp_fn(rank: int, flags: dict):
    """
    Runs inside each of the 8 spawned worker processes.
    torch_xla.launch ensures all 8 register with PJRT simultaneously.
    Each worker sees xla_device() as its own chip.
    """
    device = xm.xla_device()

    # Build model
    cfg = LeoConfig()
    cfg.use_gradient_checkpointing = flags["grad_ckpt"]

    model = LeoSLM(cfg)
    if flags["grad_ckpt"]:
        model.enable_gradient_checkpointing()

    # Cast to BF16 before moving to device
    model = model.to(torch.bfloat16).to(device)

    if rank == 0:
        pc = model.count_params_detailed()
        xm.master_print(f"\n  LeoSLM Aether  |  {pc['total']/1e9:.2f}B params")
        xm.master_print(f"  Device: {device}  |  World: {xm.xrt_world_size()} chips")
        xm.master_print(f"  GradCkpt: {'ON' if flags['grad_ckpt'] else 'OFF'}")

    # Adafactor: memory-efficient, no momentum tensors, internal clip_threshold
    optimizer = Adafactor(
        model.parameters(),
        lr              = 1e-3,
        relative_step   = False,
        scale_parameter = False,
        warmup_init     = False,
        weight_decay    = 0.1,
        clip_threshold  = 1.0,
    )

    # Resume from checkpoint if requested
    start_step, start_phase = 0, 1
    if flags["resume"]:
        start_step, start_phase = load_ckpt(model, f"{flags['ckpt_dir']}/latest.pt")

    # Verify training data exists
    if not Path(flags["train_data"]).exists():
        xm.master_print(f"\n  ERROR: {flags['train_data']} not found")
        xm.master_print("  Run Cell 3 to prepare training data first.")
        return

    dataset = LeoDataset(flags["train_data"], max_seq_len=4096, pad_id=cfg.pad_id)
    if rank == 0:
        xm.master_print(
            f"  Dataset: {len(dataset.data):,} tokens  |  "
            f"{len(dataset.data) * 4 / 1e9:.2f} GB  |  "
            f"vocab={cfg.vocab_size:,}"
        )

    loss_fn    = LeoLoss(cfg)
    all_phases = list(range(1, 9))
    phases     = [flags["phase"]] if flags["phase"] in all_phases else all_phases
    step       = start_step

    for phase in phases:
        if flags["resume"] and phase < start_phase:
            xm.master_print(f"  Skip phase {phase} (already complete)")
            continue

        step = run_phase(
            phase      = phase,
            rank       = rank,
            model      = model,
            optimizer  = optimizer,
            dataset    = dataset,
            loss_fn    = loss_fn,
            device     = device,
            grad_accum = flags["grad_accum"],
            save_every = flags["save_every"],
            ckpt_dir   = flags["ckpt_dir"],
            start_step = step,
            smoke      = flags["smoke"],
        )

        if flags["smoke"]:
            break

    xm.master_print("\n  Leo Aether training complete ✓")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeoSLM Aether — Kaggle TPU v5e-8")
    parser.add_argument("--phase",      type=int,  default=0,
                        help="Single phase to run (0 = all phases 1-8)")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from latest.pt checkpoint")
    parser.add_argument("--smoke",      action="store_true",
                        help="Run 50 steps only (sanity check)")
    parser.add_argument("--train_data", type=str,  default="./data/train.npy")
    parser.add_argument("--ckpt_dir",   type=str,  default="./checkpoints")
    parser.add_argument("--grad_accum", type=int,  default=16,
                        help="Gradient accumulation steps (eff batch = 1×N×8 chips)")
    parser.add_argument("--save_every", type=int,  default=200,
                        help="Save checkpoint every N steps")
    parser.add_argument("--grad_ckpt",  action="store_true",
                        help="Per-block activation checkpointing (~70% HBM savings, REQUIRED for 32k ctx)")
    args  = parser.parse_args()
    flags = vars(args)

    print(f"\n{'='*60}")
    print(f"  Leo Aether  |  {LEO_IDENTITY['creator']}")
    print(f"  {LEO_IDENTITY['architecture']}")
    print(f"  {LEO_IDENTITY['parameters']}")
    print(f"  Phase     : {args.phase or 'ALL (1 → 8)'}")
    print(f"  GradCkpt  : {'ON' if args.grad_ckpt else 'OFF (add --grad_ckpt for 32k ctx)'}")
    print(f"  Data      : {args.train_data}")
    print(f"{'='*60}\n")

    if XLA_AVAILABLE:
        # THE FIX: torch_xla.launch() replaces xmp.spawn()
        # Registers all 8 workers simultaneously with PJRT — works on v5e-8
        torch_xla.launch(_mp_fn, args=(flags,))
    else:
        print("[XLA unavailable — CPU/GPU single-device fallback]")
        _mp_fn(rank=0, flags=flags)


if __name__ == "__main__":
    main()
