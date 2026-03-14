import os
os.environ["XLA_USE_BF16"]                  = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"]  = "1000000000"
os.environ["PJRT_DEVICE"]                   = "TPU"
os.environ.pop("XLA_FLAGS", None)

import sys
import math
import time
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import torch_xla
    import torch_xla.core.xla_model                  as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader     as pl
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
    XLA_AVAILABLE = True
    xm.master_print("torch_xla: TPU mode")
except ImportError:
    XLA_AVAILABLE = False
    FSDP = None

    class _XMFallback:
        def xla_device(self):             return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def optimizer_step(self, o, **k): o.step()
        def mark_step(self):              pass
        def get_ordinal(self):            return 0
        def xrt_world_size(self):         return 1
        def master_print(self, *a, **k):  print(*a, **k)

    xm = _XMFallback()
    print("torch_xla not found — CPU/GPU fallback")

try:
    from transformers.optimization import Adafactor
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "transformers", "-q"], check=True)
    from transformers.optimization import Adafactor

from model    import LeoSLM, LeoConfig, CFG, LEO_IDENTITY
from training import LeoLoss, GRPOTrainer, AgenticGRPO
from data     import LeoDataset


def load_leotokenizer(path: str = "./leo_tokenizer"):
    if Path(f"{path}/tokenizer.json").exists():
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(path)
        xm.master_print(f"LeoTokenizer loaded (vocab={tok.vocab_size:,})")
        return tok

    xm.master_print("LeoTokenizer not found — run prep_data.py --tok-only first")
    xm.master_print("Falling back to GPT-2 tokenizer as placeholder...")
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


def cosine_lr(
    step:    int,
    warmup:  int,
    total:   int,
    lr_max:  float,
    lr_min:  float,
) -> float:
    if total <= 0:
        return lr_max
    if step < warmup:
        return lr_max * step / max(1, warmup)
    f = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * f))


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def save_ckpt(model, optimizer, step: int, phase: int, loss: float, path: str):
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
    xm.master_print(f"   saved {path} (step={step}, phase={phase}, loss={loss:.3f})")


def load_ckpt(model, optimizer, path: str, device) -> tuple:
    if not os.path.exists(path):
        return 0, 1
    c = torch.load(path, map_location=device)
    model.load_state_dict(c["model"])
    if optimizer and "optimizer" in c:
        optimizer.load_state_dict(c["optimizer"])
    xm.master_print(f"   Resumed {path} (step={c.get('step', 0)}, phase={c.get('phase', 1)})")
    return c.get("step", 0), c.get("phase", 1)


PHASE_CFG: Dict[int, Dict] = {
    1: {"epochs": 2, "ctx":  4096, "tau": 0.50, "lr": 1e-3, "warmup": 500, "mdm": 0.0,
        "desc": "AR warmup (4k ctx)"},
    2: {"epochs": 2, "ctx":  8192, "tau": 0.45, "lr": 8e-4, "warmup": 300, "mdm": 0.5,
        "desc": "Diffusion warmup (8k ctx)"},
    3: {"epochs": 2, "ctx": 16384, "tau": 0.40, "lr": 5e-4, "warmup": 200, "mdm": 0.5,
        "desc": "Full MoE + ECT (16k ctx)"},
    4: {"epochs": 2, "ctx": 32768, "tau": 0.35, "lr": 3e-4, "warmup": 200, "mdm": 0.5,
        "desc": "SFT alignment (32k ctx)"},
    5: {"epochs": 1, "ctx": 32768, "tau": 0.35, "lr": 1e-4, "warmup": 100, "mdm": 0.3,
        "desc": "Factuality DPO (32k ctx)"},
    6: {"epochs": 2, "ctx": 32768, "tau": 0.30, "lr": 5e-5, "warmup": 100, "mdm": 0.0,
        "desc": "GRPO Think RL (32k ctx)"},
    7: {"epochs": 2, "ctx": 32768, "tau": 0.30, "lr": 3e-5, "warmup": 100, "mdm": 0.0,
        "desc": "Agentic SFT (32k ctx)"},
    8: {"epochs": 2, "ctx": 32768, "tau": 0.28, "lr": 1e-5, "warmup":  50, "mdm": 0.0,
        "desc": "Agentic RL (32k ctx)"},
}


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
    pc      = PHASE_CFG[phase]
    epochs  = 1 if smoke else pc["epochs"]
    ctx     = pc["ctx"]
    tau     = pc["tau"]
    lr_max  = pc["lr"]
    lr_min  = lr_max * 0.1

    xm.master_print(f"\n{'='*65}")
    xm.master_print(f"  PHASE {phase} — {pc['desc']}")
    xm.master_print(f"  ctx={ctx:,} | tau={tau} | lr={lr_max} | epochs={epochs}")
    xm.master_print(f"{'='*65}")

    raw_model = model.module if hasattr(model, "module") else model

    dataset.set_seq_len(ctx)
    raw_model.cfg.uncertainty_thresh = tau
    raw_model.freeze_phase(phase)
    loss_fn.set_lambda_mdm(pc["mdm"])

    batch_size = 2 if phase == 5 else 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    if XLA_AVAILABLE:
        loader = pl.MpDeviceLoader(loader, device)

    dpo          = None
    grpo         = None
    agentic_grpo = None

    if phase == 5:
        from training import FactualityDPO
        dpo = FactualityDPO(raw_model.cfg, beta=0.1)
    elif phase == 6:
        grpo = GRPOTrainer(
            cfg            = raw_model.cfg,
            model          = model,
            optimizer      = optimizer,
            think_start_id = raw_model.cfg.think_start_id,
            think_end_id   = raw_model.cfg.think_end_id,
            idk_id         = raw_model.cfg.idk_id,
        )
    elif phase == 8:
        agentic_grpo = AgenticGRPO(
            cfg       = raw_model.cfg,
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
        ep_loss_sum = torch.zeros(1, device=device)
        nb          = 0

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
                (loss / grad_accum).backward()

            elif phase == 6 and grpo is not None:
                metrics = grpo.grpo_step(batch, device)
                loss    = metrics["total"]
                (loss / grad_accum).backward()

            elif phase == 8 and agentic_grpo is not None:
                metrics = agentic_grpo.agentic_grpo_step(batch, device)
                loss    = metrics["total"]
                (loss / grad_accum).backward()

            else:
                out           = model(input_ids)
                loss, metrics = loss_fn(out, input_ids, model=raw_model, phase=phase)
                (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if XLA_AVAILABLE:
                    xm.optimizer_step(optimizer)
                    xm.mark_step()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            nb   += 1
            step += 1

            _loss_t = metrics.get("total", loss)
            if isinstance(_loss_t, torch.Tensor):
                ep_loss_sum = ep_loss_sum + _loss_t.detach()
            else:
                ep_loss_sum = ep_loss_sum + torch.tensor(_loss_t, device=device)

            _do_log  = (step % 10        == 0)
            _do_ckpt = (step % save_every == 0)

            if _do_log or _do_ckpt:
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                m_cpu    = {k: (v.item() if isinstance(v, torch.Tensor) else v)
                            for k, v in metrics.items()}

            if _do_log:
                extra = ""
                for key in ("l_mtp", "l_acgi", "l_msra", "reward", "l_idk", "l_ect"):
                    if key in m_cpu:
                        extra += f" {key}={m_cpu[key]:.3f}"
                xm.master_print(
                    f"  p{phase} e{epoch+1} s{step:6d} | "
                    f"loss={m_cpu.get('total', loss_val):.3f} "
                    f"ar={m_cpu.get('l_ar', 0):.3f}"
                    f"{extra} lr={lr:.2e} | {time.time()-t0:.0f}s"
                )

            if _do_ckpt:
                save_ckpt(model, optimizer, step, phase,
                          m_cpu.get("total", loss_val),
                          f"{ckpt_dir}/phase{phase}_step{step}.pt")
                save_ckpt(model, optimizer, step, phase,
                          m_cpu.get("total", loss_val),
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
    parser = argparse.ArgumentParser(description="Leo Aether training")
    parser.add_argument("--phase",      type=int,   default=0)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--smoke",      action="store_true")
    parser.add_argument("--train_data", type=str,   default="./data/train.npy")
    parser.add_argument("--ckpt_dir",   type=str,   default="./checkpoints")
    parser.add_argument("--grad_accum", type=int,   default=16)
    parser.add_argument("--save_every", type=int,   default=100)
    args = parser.parse_args()

    device = xm.xla_device() if XLA_AVAILABLE else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    xm.master_print(f"\n{'='*65}")
    xm.master_print(f"  {LEO_IDENTITY['full_name']}")
    xm.master_print(f"  Built by  : {LEO_IDENTITY['creator']}")
    xm.master_print(f"  Arch      : {LEO_IDENTITY['architecture']}")
    xm.master_print(f"  Params    : {LEO_IDENTITY['parameters']}")
    xm.master_print(f"  Context   : {LEO_IDENTITY['context']}")
    xm.master_print(f"  Hardware  : {LEO_IDENTITY['hardware']}")
    xm.master_print(f"  Device    : {device}")
    xm.master_print(f"{'='*65}\n")

    cfg   = LeoConfig()
    model = LeoSLM(cfg).to(device)

    pc = model.count_params_detailed()
    xm.master_print(
        f"   Params   : {pc['total']/1e9:.2f}B total "
        f"| ~{pc.get('active_approx', pc['total'])/1e9:.2f}B active (MoE)"
    )
    xm.master_print(
        f"   MTP(N={cfg.mtp_n}) | DSA(>{cfg.dsa_threshold//1024}k) "
        f"| ACGI(tau={cfg.acgi_threshold}) | SAM(M={cfg.sam_memory_size})"
    )

    if XLA_AVAILABLE and FSDP is not None:
        model = FSDP(model, reshard_after_forward=True)
        model = model.to(torch.bfloat16)
        xm.master_print("   FSDP     : OK (8-chip full sharding, bf16)")
    elif XLA_AVAILABLE:
        model = model.to(torch.bfloat16)

    optimizer = Adafactor(
        model.parameters(),
        lr              = 1e-3,
        relative_step   = False,
        scale_parameter = False,
        warmup_init     = False,
        weight_decay    = 0.1,
        clip_threshold  = 1.0,
    )
    xm.master_print("   Optimizer: Adafactor (TPU-native, ~4x memory-efficient vs Adam)")

    start_step, start_phase = 0, 1
    if args.resume:
        start_step, start_phase = load_ckpt(
            model, optimizer, f"{args.ckpt_dir}/latest.pt", device
        )

    if not Path(args.train_data).exists():
        xm.master_print(f"\nTraining data not found: {args.train_data}")
        xm.master_print("Run: python3 prep_data.py")
        sys.exit(1)

    dataset = LeoDataset(args.train_data, max_seq_len=4096, pad_id=cfg.pad_id)
    xm.master_print(
        f"   Dataset  : {len(dataset.data):,} tokens "
        f"| {len(dataset.data) * 4 / 1e9:.1f} GB "
        f"| vocab={cfg.vocab_size:,}"
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

    xm.master_print(f"\n{'='*65}")
    xm.master_print("  Leo Aether training complete!")
    xm.master_print(f"  Built by      : Unmuted")
    xm.master_print(f"  Architecture  : MLA + MoE + ECT + TDM + ACGI + SAM + MTP + DSA")
    xm.master_print(f"  Next step     : python3 generate.py")
    xm.master_print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
