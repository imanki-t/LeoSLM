"""
LeoSLM "Aether" — train.py
============================
Training orchestration ONLY.  No model classes live here.

All architecture is in model/   — import from there.
All loss / RL code is in training/ — import from there.
Dataset is in data/ — import from there.

This file contains only:
    load_leotokenizer()   — tokenizer bootstrap
    cosine_lr()           — learning-rate schedule
    set_lr()              — apply lr to optimizer
    save_ckpt()           — atomic checkpoint save
    load_ckpt()           — checkpoint restore
    PHASE_CFG             — 8-phase training schedule
    run_phase()           — single-phase training loop
    main()                — CLI entry point

Usage:
    python3 train.py                   # all 8 phases
    python3 train.py --phase 1         # AR warmup only
    python3 train.py --resume          # resume from checkpoints/latest.pt
    python3 train.py --smoke           # 50-step smoke test
    python3 train.py --phase 6         # GRPO think RL only
    python3 train.py --phase 8         # Agentic RL only

Training Phases (Progressive Confidence Curriculum — PCC):
    1  AR warmup         4k  ctx  — basic language
    2  Diffusion warmup  8k  ctx  — learn uncertainty
    3  Full MoE + ECT   16k  ctx  — specialization + MTP
    4  SFT alignment    32k  ctx  — instructions + IDK + identity
    5  Factuality DPO   32k  ctx  — FActScore preference training
    6  GRPO Think RL    32k  ctx  — DeepSeek-R1 CoT reinforcement
    7  Agentic SFT      32k  ctx  — tool-use format + MCP + web search
    8  Agentic RL       32k  ctx  — GRPO on full tool-call trajectories
"""

# ── XLA environment — MUST precede all imports ────────────────────────────────
import os
os.environ["XLA_USE_BF16"]                  = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"]  = "1000000000"
os.environ["PJRT_DEVICE"]                   = "TPU"

import sys
import math
import time
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── TPU backend ───────────────────────────────────────────────────────────────
try:
    import torch_xla
    import torch_xla.core.xla_model           as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader    as pl
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
    XLA_AVAILABLE = True
    xm.master_print("✅ torch_xla: TPU mode")
except ImportError:
    XLA_AVAILABLE = False
    FSDP = None

    class _XMFallback:
        def xla_device(self):          return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def optimizer_step(self, o, **k): o.step()
        def mark_step(self):           pass
        def get_ordinal(self):         return 0
        def xrt_world_size(self):      return 1
        def master_print(self, *a, **k): print(*a, **k)

    xm = _XMFallback()
    print("⚠️  torch_xla not found — CPU/GPU fallback")

# ── Adafactor ─────────────────────────────────────────────────────────────────
try:
    from transformers.optimization import Adafactor
except ImportError:
    os.system("pip install transformers -q")
    from transformers.optimization import Adafactor

# ── Project imports (all model/training/data logic lives in modules) ──────────
from model    import LeoSLM, LeoConfig, CFG, LEO_IDENTITY
from training import LeoLoss, GRPOTrainer, AgenticGRPO
from data     import LeoDataset


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

def load_leotokenizer(path: str = "./leo_tokenizer"):
    """Load the custom LeoTokenizer. Falls back to GPT-2 if not yet built."""
    if Path(f"{path}/tokenizer.json").exists():
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(path)
        xm.master_print(f"✅ LeoTokenizer loaded (vocab={tok.vocab_size:,})")
        return tok

    xm.master_print("⚠️  LeoTokenizer not found — run prep_data.py --tok-only first")
    xm.master_print("    Falling back to GPT-2 tokenizer as placeholder...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({
        "pad_token": "[PAD]",
        "additional_special_tokens": [
            "[MASK]", "[IDK]", "<think>", "</think>",
            "[BUDGET]",
            "<|tool_call|>", "<|/tool_call|>",
            "<|tool_result|>", "<|/tool_result|>",
            "<|system|>",
        ],
    })
    return tok


# ══════════════════════════════════════════════════════════════════════════════
# LEARNING-RATE SCHEDULE + OPTIMIZER UTILS
# ══════════════════════════════════════════════════════════════════════════════

def cosine_lr(
    step:    int,
    warmup:  int,
    total:   int,
    lr_max:  float,
    lr_min:  float,
) -> float:
    """Linear warmup → cosine decay schedule."""
    if total <= 0:
        return lr_max
    if step < warmup:
        return lr_max * step / max(1, warmup)
    f = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * f))


def set_lr(optimizer, lr: float):
    """Apply a new learning rate to all optimizer param groups."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILS
# ══════════════════════════════════════════════════════════════════════════════

def save_ckpt(
    model,
    optimizer,
    step:  int,
    phase: int,
    loss:  float,
    path:  str,
):
    """Atomic checkpoint save (write to .tmp then rename)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "phase":     phase,
        "loss":      loss,
    }, tmp)
    os.replace(tmp, path)
    xm.master_print(f"   💾 {path} (step={step}, phase={phase}, loss={loss:.3f})")


def load_ckpt(model, optimizer, path: str, device) -> tuple:
    """
    Restore model + optimizer state from a checkpoint.

    Returns:
        (step, phase) to resume from
    """
    if not os.path.exists(path):
        return 0, 1
    c = torch.load(path, map_location=device)
    model.load_state_dict(c["model"])
    if optimizer and "optimizer" in c:
        optimizer.load_state_dict(c["optimizer"])
    xm.master_print(
        f"   ✅ Resumed {path} (step={c.get('step', 0)}, phase={c.get('phase', 1)})"
    )
    return c.get("step", 0), c.get("phase", 1)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE CONFIGURATION — Progressive Confidence Curriculum
# ══════════════════════════════════════════════════════════════════════════════

PHASE_CFG: Dict[int, Dict] = {
    1: {
        "epochs": 2, "ctx":  4096, "tau": 0.50,
        "lr": 1e-3, "warmup": 500, "mdm": 0.0,
        "desc": "AR warmup (4k ctx) — basic language modelling",
    },
    2: {
        "epochs": 2, "ctx":  8192, "tau": 0.45,
        "lr": 8e-4, "warmup": 300, "mdm": 0.5,
        "desc": "Diffusion warmup (8k ctx) — learn uncertainty",
    },
    3: {
        "epochs": 2, "ctx": 16384, "tau": 0.40,
        "lr": 5e-4, "warmup": 200, "mdm": 0.5,
        "desc": "Full MoE + ECT (16k ctx) — specialization + MTP",
    },
    4: {
        "epochs": 2, "ctx": 32768, "tau": 0.35,
        "lr": 3e-4, "warmup": 200, "mdm": 0.5,
        "desc": "SFT alignment (32k ctx) — instructions + IDK + identity",
    },
    5: {
        "epochs": 1, "ctx": 32768, "tau": 0.35,
        "lr": 1e-4, "warmup": 100, "mdm": 0.3,
        "desc": "Factuality DPO (32k ctx) — FActScore preference",
    },
    6: {
        "epochs": 2, "ctx": 32768, "tau": 0.30,
        "lr": 5e-5, "warmup": 100, "mdm": 0.0,
        "desc": "GRPO Think RL (32k ctx) — DeepSeek-R1 CoT",
    },
    7: {
        "epochs": 2, "ctx": 32768, "tau": 0.30,
        "lr": 3e-5, "warmup": 100, "mdm": 0.0,
        "desc": "Agentic SFT (32k ctx) — tool format + MCP + web search + TTIP",
    },
    8: {
        "epochs": 2, "ctx": 32768, "tau": 0.28,
        "lr": 1e-5, "warmup":  50, "mdm": 0.0,
        "desc": "Agentic RL (32k ctx) — GRPO + MSRA trajectory credit",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_phase(
    phase:      int,
    model:      nn.Module,
    optimizer,
    dataset:    LeoDataset,
    loss_fn:    LeoLoss,
    device,
    grad_accum: int,
    save_every: int,
    ckpt_dir:   str,
    start_step: int  = 0,
    smoke:      bool = False,
) -> int:
    """
    Train for one phase. Returns the final global step count.

    Args:
        phase      : phase number 1-8
        model      : LeoSLM instance
        optimizer  : Adafactor optimizer
        dataset    : LeoDataset (seq_len updated inside this function)
        loss_fn    : LeoLoss instance
        device     : xla / cuda / cpu device
        grad_accum : gradient accumulation steps
        save_every : checkpoint every N steps
        ckpt_dir   : checkpoint directory
        start_step : global step to resume from
        smoke      : if True, stop after 50 steps (smoke test)
    """
    pc      = PHASE_CFG[phase]
    epochs  = 1 if smoke else pc["epochs"]
    ctx     = pc["ctx"]
    tau     = pc["tau"]
    lr_max  = pc["lr"]
    lr_min  = lr_max * 0.1

    xm.master_print(f"\n{'='*65}")
    xm.master_print(f"  PHASE {phase} — {pc['desc']}")
    xm.master_print(f"  ctx={ctx:,} | τ={tau} | lr={lr_max} | epochs={epochs}")
    xm.master_print(f"{'='*65}")

    # Update dataset context length and model uncertainty threshold for this phase
    dataset.set_seq_len(ctx)
    model.cfg.uncertainty_thresh = tau
    model.freeze_phase(phase)
    loss_fn.set_lambda_mdm(pc["mdm"])

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    if XLA_AVAILABLE:
        loader = pl.MpDeviceLoader(loader, device)

    # Instantiate RL trainers for phases 6 and 8 (only created when needed)
    grpo         = None
    agentic_grpo = None
    if phase == 6:
        grpo = GRPOTrainer(
            cfg            = model.cfg,
            model          = model,
            optimizer      = optimizer,
            think_start_id = model.cfg.think_start_id,
            think_end_id   = model.cfg.think_end_id,
            idk_id         = model.cfg.idk_id,
        )
    elif phase == 8:
        agentic_grpo = AgenticGRPO(
            cfg       = model.cfg,
            model     = model,
            optimizer = optimizer,
        )

    total_steps = len(dataset) * epochs
    step        = start_step
    best        = float("inf")
    t0          = time.time()

    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        ep_loss, nb = 0.0, 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)

            # LR schedule
            lr = cosine_lr(step, pc["warmup"], total_steps, lr_max, lr_min)
            set_lr(optimizer, lr)

            # ── Forward + loss ─────────────────────────────────────────────────
            if phase == 6 and grpo is not None:
                metrics = grpo.grpo_step(batch, device)
                loss    = metrics["total"]
                (loss / grad_accum).backward()

            elif phase == 8 and agentic_grpo is not None:
                metrics = agentic_grpo.agentic_grpo_step(batch, device)
                loss    = metrics["total"]
                (loss / grad_accum).backward()

            else:
                # Phases 1-5, 7: standard supervised training
                out           = model(input_ids)
                loss, metrics = loss_fn(out, input_ids, model=model, phase=phase)
                (loss / grad_accum).backward()

            # ── Gradient step ──────────────────────────────────────────────────
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if XLA_AVAILABLE:
                    xm.optimizer_step(optimizer)
                    xm.mark_step()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # ── Logging ────────────────────────────────────────────────────────
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            ep_loss += metrics.get("total", loss_val)
            nb      += 1
            step    += 1

            if step % 10 == 0:
                extra = ""
                for key in ("l_mtp", "l_acgi", "l_msra", "reward", "l_idk", "l_ect"):
                    if key in metrics:
                        extra += f" {key}={metrics[key]:.3f}"
                xm.master_print(
                    f"  p{phase} e{epoch+1} s{step:6d} | "
                    f"loss={metrics.get('total', loss_val):.3f} "
                    f"ar={metrics.get('l_ar', 0):.3f}"
                    f"{extra} lr={lr:.2e} | {time.time()-t0:.0f}s"
                )

            # ── Checkpoint ────────────────────────────────────────────────────
            if step % save_every == 0:
                save_ckpt(model, optimizer, step, phase,
                          metrics.get("total", loss_val),
                          f"{ckpt_dir}/phase{phase}_step{step}.pt")
                save_ckpt(model, optimizer, step, phase,
                          metrics.get("total", loss_val),
                          f"{ckpt_dir}/latest.pt")

            if smoke and step >= start_step + 50:
                xm.master_print("  🔥 Smoke test complete (50 steps)")
                return step

        avg = ep_loss / max(nb, 1)
        xm.master_print(f"  ✅ Epoch {epoch+1}/{epochs} | avg_loss={avg:.3f}")
        if avg < best:
            best = avg
            save_ckpt(model, optimizer, step, phase, avg,
                      f"{ckpt_dir}/best_phase{phase}.pt")

    return step


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(rank=None):
    parser = argparse.ArgumentParser(description="Leo Aether — LeoSLM training")
    parser.add_argument("--phase",      type=int,   default=0,
                        help="Run single phase (1-8); 0 = all phases")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from ./checkpoints/latest.pt")
    parser.add_argument("--smoke",      action="store_true",
                        help="50-step smoke test")
    parser.add_argument("--train_data", type=str,   default="./data/train.npy")
    parser.add_argument("--ckpt_dir",   type=str,   default="./checkpoints")
    parser.add_argument("--grad_accum", type=int,   default=16,
                        help="Gradient accumulation steps (eff. batch = 1 × accum × 8 chips)")
    parser.add_argument("--save_every", type=int,   default=100,
                        help="Save checkpoint every N steps")
    args = parser.parse_args()

    device = xm.xla_device() if XLA_AVAILABLE else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # ── Banner ────────────────────────────────────────────────────────────────
    xm.master_print(f"\n{'='*65}")
    xm.master_print(f"  🦁  {LEO_IDENTITY['full_name']}")
    xm.master_print(f"  Built by  : {LEO_IDENTITY['creator']}")
    xm.master_print(f"  Arch      : {LEO_IDENTITY['architecture']}")
    xm.master_print(f"  Params    : {LEO_IDENTITY['parameters']}")
    xm.master_print(f"  Context   : {LEO_IDENTITY['context']}")
    xm.master_print(f"  Hardware  : {LEO_IDENTITY['hardware']}")
    xm.master_print(f"  Device    : {device}")
    xm.master_print(f"  Phases    : 8 (AR → Diff → MoE → SFT → DPO → GRPO → AgSFT → AgRL)")
    xm.master_print(f"  Novel     : ACGI | SAM | ESTR | TTIP | MSRA | MTP | DSA-lite | EPE")
    xm.master_print(f"{'='*65}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg   = LeoConfig()
    model = LeoSLM(cfg).to(device)

    if XLA_AVAILABLE:
        model = model.to(torch.bfloat16)
    if XLA_AVAILABLE and FSDP is not None:
        model = FSDP(model, compute_dtype=torch.bfloat16, reshard_after_forward=True)
        xm.master_print("   FSDP     : ✅ (8-chip full sharding)")

    pc = model.count_params()
    xm.master_print(f"   Params   : {pc['total']/1e9:.2f}B total "
                    f"| ~{pc['approx_active']/1e9:.2f}B active (MoE)")
    xm.master_print(f"   MTP(N={cfg.mtp_n}) | DSA(>{cfg.dsa_threshold//1024}k) "
                    f"| ACGI(τ={cfg.acgi_threshold}) | SAM(M={cfg.sam_memory_size})")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        weight_decay=0.1,
        clip_threshold=1.0,
    )
    xm.master_print("   Optimizer: Adafactor (TPU-native, ~4× memory-efficient vs Adam)")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step, start_phase = 0, 1
    if args.resume:
        start_step, start_phase = load_ckpt(
            model, optimizer, f"{args.ckpt_dir}/latest.pt", device
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not Path(args.train_data).exists():
        xm.master_print(f"\n❌ Training data not found: {args.train_data}")
        xm.master_print("   Run: python3 prep_data.py")
        sys.exit(1)

    dataset = LeoDataset(args.train_data, max_seq_len=4096, pad_id=cfg.pad_id)
    xm.master_print(
        f"   Dataset  : {len(dataset.data):,} tokens "
        f"| {len(dataset.data) * 4 / 1e9:.1f} GB "
        f"| vocab={cfg.vocab_size:,}"
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = LeoLoss(cfg)

    # ── Phase selection ───────────────────────────────────────────────────────
    all_phases = list(range(1, 9))
    phases     = [args.phase] if args.phase in all_phases else all_phases
    step       = start_step

    for phase in phases:
        if args.resume and phase < start_phase:
            xm.master_print(f"  ⏭  Skip phase {phase} (already complete)")
            continue
        step = run_phase(
            phase      = phase,
            model      = model,
            optimizer  = optimizer,
            dataset    = dataset,
            loss_fn    = loss_fn,
            device     = device,
            grad_accum = args.grad_accum,
            save_every = args.save_every,
            ckpt_dir   = args.ckpt_dir,
            start_step = step,
            smoke      = args.smoke,
        )
        if args.smoke:
            break

    # ── Done ──────────────────────────────────────────────────────────────────
    xm.master_print(f"\n{'='*65}")
    xm.master_print("  🎉 Leo Aether training complete!")
    xm.master_print(f"  Built by      : Unmuted")
    xm.master_print(f"  Architecture  : MLA + MoE + ECT + TDM + ACGI + SAM + MTP + DSA")
    xm.master_print(f"  Anti-halluc.  : 12 layers (ECT+Brier+IDK+Const+DPO+GRPO+PRM+CUR+ACGI)")
    xm.master_print(f"  Agentic       : tool-calling | MCP | web search | TTIP | MSRA")
    xm.master_print(f"  Inference     : think/nothink | 128k YaRN | TDM | speculative via MTP")
    xm.master_print(f"  Next step     : python3 generate.py")
    xm.master_print(f"{'='*65}\n")


if __name__ == "__main__":
    if XLA_AVAILABLE:
        xmp.spawn(main, args=(), nprocs=8, start_method="fork")
    else:
        main()
