import os
os.environ["PJRT_DEVICE"]                  = "TPU"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "1000000000"
os.environ["TF_CPP_MIN_LOG_LEVEL"]         = "3"
os.environ["GRPC_VERBOSITY"]               = "ERROR"
os.environ["JAX_PLATFORMS"]               = "tpu"
os.environ.pop("XLA_FLAGS",    None)
os.environ.pop("XLA_USE_BF16", None)

import sys, math, time, argparse, functools
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import torch_xla
    import torch_xla.core.xla_model             as xm
    import torch_xla.distributed.parallel_loader as pl
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
    from torch_xla.distributed.fsdp import checkpoint_module
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    XLA_AVAILABLE = True
    xm.master_print("torch_xla: TPU mode")
except ImportError:
    XLA_AVAILABLE    = False
    FSDP             = None
    checkpoint_module         = lambda m: m
    transformer_auto_wrap_policy = None
    class _XM:
        def xla_device(self):             return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def optimizer_step(self, o, **k): o.step()
        def mark_step(self):              pass
        def get_ordinal(self):            return 0
        def xrt_world_size(self):         return 1
        def master_print(self, *a, **k):  print(*a, **k)
        def rendezvous(self, t):          pass
    xm = _XM()
    print("torch_xla not found — CPU/GPU fallback")

try:
    from transformers.optimization import Adafactor
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--user",
                    "transformers", "-q"], check=False)
    from transformers.optimization import Adafactor

from model    import LeoSLM, LeoConfig, CFG, LEO_IDENTITY
from model    import LeoBlock
from training import LeoLoss, GRPOTrainer, AgenticGRPO
from data     import LeoDataset


def cosine_lr(step, warmup, total, lr_max, lr_min):
    if total <= 0: return lr_max
    if step < warmup: return lr_max * step / max(1, warmup)
    f = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * f))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def save_ckpt(model, optimizer, step, phase, loss, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    tmp = path + ".tmp"
    torch.save({
        "model": raw.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "phase": phase, "loss": loss,
    }, tmp)
    os.replace(tmp, path)
    xm.master_print(f"   saved {path} (step={step} phase={phase} loss={loss:.3f})")


def load_ckpt(model, optimizer, path, device):
    if not os.path.exists(path):
        return 0, 1
    c   = torch.load(path, map_location=device)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(c["model"])
    if optimizer and "optimizer" in c:
        optimizer.load_state_dict(c["optimizer"])
    xm.master_print(f"   Resumed {path} (step={c.get('step',0)} phase={c.get('phase',1)})")
    return c.get("step", 0), c.get("phase", 1)


PHASE_CFG: Dict[int, Dict] = {
    1: {"epochs":2,"ctx": 4096,"tau":0.50,"lr":1e-3,"warmup":500,"mdm":0.0,"desc":"AR warmup (4k ctx)"},
    2: {"epochs":2,"ctx": 8192,"tau":0.45,"lr":8e-4,"warmup":300,"mdm":0.5,"desc":"Diffusion warmup (8k ctx)"},
    3: {"epochs":2,"ctx":16384,"tau":0.40,"lr":5e-4,"warmup":200,"mdm":0.5,"desc":"Full MoE+ECT (16k ctx)"},
    4: {"epochs":2,"ctx":32768,"tau":0.35,"lr":3e-4,"warmup":200,"mdm":0.5,"desc":"SFT alignment (32k ctx)"},
    5: {"epochs":1,"ctx":32768,"tau":0.35,"lr":1e-4,"warmup":100,"mdm":0.3,"desc":"Factuality DPO (32k ctx)"},
    6: {"epochs":2,"ctx":32768,"tau":0.30,"lr":5e-5,"warmup":100,"mdm":0.0,"desc":"GRPO Think RL (32k ctx)"},
    7: {"epochs":2,"ctx":32768,"tau":0.30,"lr":3e-5,"warmup":100,"mdm":0.0,"desc":"Agentic SFT (32k ctx)"},
    8: {"epochs":2,"ctx":32768,"tau":0.28,"lr":1e-5,"warmup": 50,"mdm":0.0,"desc":"Agentic RL (32k ctx)"},
}


def wrap_fsdp(model):
    if not XLA_AVAILABLE or FSDP is None or transformer_auto_wrap_policy is None:
        return model

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LeoBlock},
    )

    auto_wrapper_callable = lambda m, *args, **kwargs: FSDP(
        checkpoint_module(m), *args, **kwargs
    )

    wrapped = FSDP(
        model,
        auto_wrap_policy      = auto_wrap_policy,
        auto_wrapper_callable = auto_wrapper_callable,
        reshard_after_forward = True,
        flatten_parameters    = False,
    )
    xm.rendezvous("fsdp_init")
    xm.master_print("   FSDP: OK (nested per-block, gradient checkpointing, 8-chip ZeRO-3)")
    return wrapped


def run_phase(phase, model, optimizer, dataset, loss_fn,
              device, grad_accum, save_every, ckpt_dir,
              start_step=0, smoke=False):
    pc     = PHASE_CFG[phase]
    epochs = 1 if smoke else pc["epochs"]
    lr_max = pc["lr"]
    lr_min = lr_max * 0.1

    xm.master_print(f"\n{'='*65}")
    xm.master_print(f"  PHASE {phase} — {pc['desc']}")
    xm.master_print(f"  ctx={pc['ctx']:,} | tau={pc['tau']} | lr={lr_max} | epochs={epochs}")
    xm.master_print(f"{'='*65}")

    raw_model = model.module if hasattr(model, "module") else model
    dataset.set_seq_len(pc["ctx"])
    raw_model.cfg.uncertainty_thresh = pc["tau"]
    raw_model.freeze_phase(phase)
    loss_fn.set_lambda_mdm(pc["mdm"])

    batch_size = 2 if phase == 5 else 1
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, prefetch_factor=2,
        drop_last=True, pin_memory=False,
    )
    if XLA_AVAILABLE:
        loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)

    dpo = grpo = agentic_grpo = None
    if phase == 5:
        from training import FactualityDPO
        dpo = FactualityDPO(raw_model.cfg, beta=0.1)
    elif phase == 6:
        grpo = GRPOTrainer(
            cfg=raw_model.cfg, model=model, optimizer=optimizer,
            think_start_id=raw_model.cfg.think_start_id,
            think_end_id=raw_model.cfg.think_end_id,
            idk_id=raw_model.cfg.idk_id,
        )
    elif phase == 8:
        agentic_grpo = AgenticGRPO(cfg=raw_model.cfg, model=model, optimizer=optimizer)

    total_steps = (len(dataset) // batch_size) * epochs
    step = start_step
    best = float("inf")
    t0   = time.time()

    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        ep_loss_sum = torch.zeros(1, device=device)
        nb = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lr = cosine_lr(step, pc["warmup"], total_steps, lr_max, lr_min)
            set_lr(optimizer, lr)

            if phase == 5 and dpo is not None:
                out    = model(input_ids)
                U_mean = out["uncertainty"].mean(dim=1)
                lo, hi = (0, 1) if U_mean[0] <= U_mean[1] else (1, 0)
                chosen_out   = {k: v[lo:lo+1] for k, v in out.items()
                                if isinstance(v, torch.Tensor) and v.shape[0] == 2}
                rejected_out = {k: v[hi:hi+1] for k, v in out.items()
                                if isinstance(v, torch.Tensor) and v.shape[0] == 2}
                loss, metrics = dpo(chosen_out, rejected_out,
                                    input_ids[lo:lo+1], input_ids[hi:hi+1])
            elif phase == 6 and grpo is not None:
                metrics = grpo.grpo_step(batch, device)
                loss    = metrics["total"]
            elif phase == 8 and agentic_grpo is not None:
                metrics = agentic_grpo.agentic_grpo_step(batch, device)
                loss    = metrics["total"]
            else:
                out           = model(input_ids)
                loss, metrics = loss_fn(out, input_ids, model=raw_model, phase=phase)

            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                if XLA_AVAILABLE:
                    optimizer.step()
                    xm.mark_step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            nb   += 1
            step += 1

            if isinstance(loss, torch.Tensor):
                ep_loss_sum = ep_loss_sum + loss.detach()
            else:
                ep_loss_sum = ep_loss_sum + torch.tensor(float(loss), device=device)

            if nb % 10 == 0 and XLA_AVAILABLE:
                xm.mark_step()

            _do_log  = (step % 10        == 0)
            _do_ckpt = (step % save_every == 0)

            if _do_log or _do_ckpt:
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                m_cpu    = {k: (v.item() if isinstance(v, torch.Tensor) else v)
                            for k, v in metrics.items()}

            if _do_log:
                extra = "".join(
                    f" {k}={m_cpu[k]:.3f}"
                    for k in ("l_mtp","l_acgi","l_msra","reward","l_idk","l_ect")
                    if k in m_cpu
                )
                xm.master_print(
                    f"  p{phase} e{epoch+1} s{step:6d} | "
                    f"loss={m_cpu.get('total', loss_val):.3f} "
                    f"ar={m_cpu.get('l_ar', 0):.3f}"
                    f"{extra} lr={lr:.2e} | {time.time()-t0:.0f}s"
                )

            if _do_ckpt:
                lv = m_cpu.get("total", loss_val)
                save_ckpt(model, optimizer, step, phase, lv,
                          f"{ckpt_dir}/phase{phase}_step{step}.pt")
                save_ckpt(model, optimizer, step, phase, lv,
                          f"{ckpt_dir}/latest.pt")

            if smoke and step >= start_step + 50:
                xm.master_print("  Smoke test complete (50 steps)")
                return step

        avg = ep_loss_sum.cpu().item() / max(nb, 1)
        xm.master_print(f"  Epoch {epoch+1}/{epochs} | avg_loss={avg:.3f}")
        if avg < best:
            best = avg
            save_ckpt(model, optimizer, step, phase, avg,
                      f"{ckpt_dir}/best_phase{phase}.pt")

    return step


def main(rank=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",      type=int, default=0)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--smoke",      action="store_true")
    parser.add_argument("--train_data", type=str, default="./data/train.npy")
    parser.add_argument("--ckpt_dir",   type=str, default="./checkpoints")
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=200)
    args = parser.parse_args()

    device = torch_xla.device() if XLA_AVAILABLE else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    xm.master_print(f"\n{'='*65}")
    xm.master_print(f"  {LEO_IDENTITY['full_name']}")
    xm.master_print(f"  Built by  : {LEO_IDENTITY['creator']}")
    xm.master_print(f"  Arch      : {LEO_IDENTITY['architecture']}")
    xm.master_print(f"  Params    : {LEO_IDENTITY['parameters']}")
    xm.master_print(f"  Hardware  : {LEO_IDENTITY['hardware']}")
    xm.master_print(f"  Device    : {device}")
    xm.master_print(f"{'='*65}\n")

    cfg   = LeoConfig()
    model = LeoSLM(cfg)

    pc = model.count_params_detailed()
    xm.master_print(f"   Params   : {pc['total']/1e9:.2f}B total")

    model = wrap_fsdp(model)

    if not (XLA_AVAILABLE and FSDP is not None):
        model = model.to(device)

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3, relative_step=False, scale_parameter=False,
        warmup_init=False, weight_decay=0.1, clip_threshold=1.0,
    )
    xm.master_print("   Optimizer: Adafactor (TPU-native)")

    start_step, start_phase = 0, 1
    if args.resume:
        start_step, start_phase = load_ckpt(
            model, optimizer, f"{args.ckpt_dir}/latest.pt", device
        )

    if not Path(args.train_data).exists():
        xm.master_print(f"\nTraining data not found: {args.train_data}")
        sys.exit(1)

    dataset = LeoDataset(args.train_data, max_seq_len=4096, pad_id=cfg.pad_id)
    xm.master_print(
        f"   Dataset  : {len(dataset.data):,} tokens "
        f"| {len(dataset.data)*4/1e9:.1f} GB | vocab={cfg.vocab_size:,}"
    )

    loss_fn    = LeoLoss(cfg)
    all_phases = list(range(1, 9))
    phases     = [args.phase] if args.phase in all_phases else all_phases
    step       = start_step

    for phase in phases:
        if args.resume and phase < start_phase:
            xm.master_print(f"  Skip phase {phase} (already complete)")
            continue
        step = run_phase(
            phase=phase, model=model, optimizer=optimizer,
            dataset=dataset, loss_fn=loss_fn, device=device,
            grad_accum=args.grad_accum, save_every=args.save_every,
            ckpt_dir=args.ckpt_dir, start_step=step, smoke=args.smoke,
        )
        if args.smoke:
            break

    xm.master_print(f"\n{'='*65}")
    xm.master_print("  Leo Aether training complete!")
    xm.master_print(f"  Next step : python3 generate.py")
    xm.master_print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
