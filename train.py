"""
LeoSLM — train.py
==================
Main training loop. Runs all 5 phases sequentially or individually.

Usage:
    python3 train.py                    # runs all phases from scratch
    python3 train.py --phase 1          # AR warmup only
    python3 train.py --phase 2          # diffusion warmup only
    python3 train.py --phase 3          # joint training only
    python3 train.py --resume           # resume from latest checkpoint
"""

import os
import sys
import math
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# ── Local imports ────────────────────────────────────────────────────────────
from model.leoSLM         import LeoSLM, LeoConfig
from diffusion.mdlm_loss  import LeoLoss
from training.calibration_loss import ECTCalibrationLoss, IDKLoss
from training.constitutional   import ConstitutionalConditioner, ConstitutionalLoss

# ── CPU Optimizations (Oracle A1) ─────────────────────────────────────────────
torch.set_num_threads(4)
torch.set_float32_matmul_precision("medium")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config():
    return LeoConfig(
        vocab_size            = 50260,  # GPT-2 + 3 special tokens
        max_seq_len           = 512,
        hidden_dim            = 512,
        num_layers            = 16,
        num_heads             = 8,
        num_kv_heads          = 2,
        num_ect               = 4,
        gate_temperature      = 1.0,
        uncertainty_threshold = 0.35,
        dropout               = 0.0,
        weight_tying          = True,
        pad_token_id          = 50257,
        mask_token_id         = 50258,
        idk_token_id          = 50259,
    )


# ---------------------------------------------------------------------------
# LR Scheduler (cosine with warmup)
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps, total_steps, lr_max, lr_min):
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, phase, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step"   : step,
        "phase"  : phase,
        "metrics": metrics,
        "config" : model.config.__dict__,
    }, path)
    print(f"  💾 Saved checkpoint → {path}")


def load_checkpoint(model, optimizer, path, device):
    if not Path(path).exists():
        print(f"  No checkpoint at {path}, starting fresh.")
        return 0, 1
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step  = ckpt.get("step",  0)
    phase = ckpt.get("phase", 1)
    print(f"  ✅ Resumed from {path} (step={step}, phase={phase})")
    return step, phase


# ---------------------------------------------------------------------------
# Build DataLoader (with fallback for missing data files)
# ---------------------------------------------------------------------------

def build_loader(data_path, tokenizer, batch_size, shuffle=True):
    import numpy as np
    from torch.utils.data import Dataset

    class SimpleDataset(Dataset):
        def __init__(self, path, max_len=512, pad_id=50257):
            self.tokens  = np.load(path, mmap_mode="r")
            self.max_len = max_len
            self.pad_id  = pad_id

        def __len__(self):
            return max(1, len(self.tokens) // self.max_len)

        def __getitem__(self, idx):
            start = idx * self.max_len
            chunk = self.tokens[start : start + self.max_len].astype(np.int64)
            if len(chunk) < self.max_len:
                pad   = [self.pad_id] * (self.max_len - len(chunk))
                chunk = list(chunk) + pad
            ids  = torch.tensor(chunk, dtype=torch.long)
            mask = (ids != self.pad_id).long()
            return {"input_ids": ids, "attention_mask": mask}

    ds = SimpleDataset(data_path, max_len=512, pad_id=50257)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=True)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(
    model, batch, loss_fn, cal_loss_fn, const_loss_fn,
    conditioner, device, phase, step
):
    input_ids = batch["input_ids"].to(device)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def model_forward_fn(noisy_ids, t):
        # For MDLM loss: add constitutional conditioning to noise embedding
        return model(noisy_ids, noise_level=t)

    out = model(input_ids)

    # ── Losses ───────────────────────────────────────────────────────────────
    total_loss, metrics = loss_fn(
        model_out       = out,
        input_ids       = input_ids,
        model_forward_fn= model_forward_fn if phase >= 2 else None,
        device          = device,
    )

    # ECT calibration loss (phase 3+)
    if phase >= 3:
        cal_loss, cal_metrics = cal_loss_fn(
            ar_logits   = out["ar_logits"],
            input_ids   = input_ids,
            uncertainty = out["uncertainty"],
            pad_token_id= model.config.pad_token_id,
        )
        total_loss = total_loss + cal_loss
        metrics.update(cal_metrics)

    # Constitutional loss (phase 2+)
    if phase >= 2 and conditioner is not None:
        p_ids = conditioner.sample_principle(input_ids.shape[0], device)
        const_loss, const_metrics = const_loss_fn(
            diff_logits   = out["diff_logits"],
            uncertainty   = out["uncertainty"],
            principle_ids = p_ids,
            vocab_size    = model.config.vocab_size,
        )
        total_loss = total_loss + const_loss
        metrics.update(const_metrics)

    return total_loss, metrics


# ---------------------------------------------------------------------------
# One full phase training loop
# ---------------------------------------------------------------------------

def run_phase(
    phase          : int,
    model          : LeoSLM,
    optimizer      : torch.optim.Optimizer,
    train_loader   : DataLoader,
    device         : torch.device,
    loss_fn        : LeoLoss,
    cal_loss_fn    : ECTCalibrationLoss,
    const_loss_fn  : ConstitutionalLoss,
    conditioner    : ConstitutionalConditioner,
    epochs         : int,
    lr_max         : float,
    lr_min         : float,
    warmup_steps   : int,
    grad_accum     : int,
    save_every     : int,
    start_step     : int = 0,
    checkpoint_dir : str = "./checkpoints",
):
    total_steps = len(train_loader) * epochs
    step        = start_step
    best_loss   = float("inf")

    print(f"\n{'='*60}")
    print(f"  Phase {phase} — {epochs} epochs, {total_steps} steps total")
    print(f"  Batch size: {train_loader.batch_size} × {grad_accum} accum = {train_loader.batch_size * grad_accum} effective")
    print(f"{'='*60}\n")

    # ── Phase-specific setup ──────────────────────────────────────────────────
    if phase == 1:
        model.freeze_gate_phase1()
        loss_fn.set_lambda_mdm(0.0)
    elif phase == 2:
        model.freeze_backbone_phase2()
        loss_fn.set_lambda_mdm(0.3)
    elif phase == 3:
        model.unfreeze_all()
        model.unfreeze_gate_phase2()
        loss_fn.set_lambda_mdm(0.5)

    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        epoch_loss  = 0.0
        epoch_start = time.time()
        n_batches   = 0

        for batch in train_loader:
            # LR schedule
            lr = get_lr(step, warmup_steps, total_steps, lr_max, lr_min)
            set_lr(optimizer, lr)

            # Forward + loss
            loss, metrics = training_step(
                model, batch, loss_fn, cal_loss_fn, const_loss_fn,
                conditioner, device, phase, step
            )

            # Scale for gradient accumulation
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += metrics.get("total_loss", loss.item() * grad_accum)
            n_batches  += 1
            step       += 1

            # ── Logging ──────────────────────────────────────────────────────
            if step % 50 == 0:
                elapsed = time.time() - epoch_start
                avg_loss = epoch_loss / n_batches
                ar_ppl   = math.exp(min(metrics.get("ar_loss", 10), 20))
                print(
                    f"  [Phase {phase}] Ep {epoch+1}/{epochs} "
                    f"Step {step} | "
                    f"Loss {avg_loss:.4f} | "
                    f"PPL {ar_ppl:.1f} | "
                    f"LR {lr:.2e} | "
                    f"MDM λ={loss_fn.lambda_mdm:.1f} | "
                    f"{elapsed:.0f}s"
                )

            # ── Save checkpoint ───────────────────────────────────────────────
            if step % save_every == 0:
                ckpt_path = f"{checkpoint_dir}/phase{phase}_step{step}.pt"
                save_checkpoint(model, optimizer, step, phase, metrics, ckpt_path)
                # Always keep a "latest" checkpoint for easy resuming
                save_checkpoint(model, optimizer, step, phase, metrics,
                                f"{checkpoint_dir}/latest.pt")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - epoch_start
        print(f"\n  ✅ Phase {phase} Epoch {epoch+1} done | "
              f"Avg loss: {avg_epoch_loss:.4f} | "
              f"Time: {elapsed/60:.1f} min\n")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, step, phase, {"loss": best_loss},
                            f"{checkpoint_dir}/phase{phase}_best.pt")

    # Save final checkpoint for this phase
    save_checkpoint(model, optimizer, step, phase, {"loss": best_loss},
                    f"{checkpoint_dir}/phase{phase}_final.pt")
    save_checkpoint(model, optimizer, step, phase, {"loss": best_loss},
                    f"{checkpoint_dir}/latest.pt")

    print(f"  🏁 Phase {phase} complete. Best loss: {best_loss:.4f}")
    return step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",      type=int, default=0,
                        help="0=all phases, 1-3=specific phase")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--train_data", type=str,
                        default="./data/tinystories_train.npy")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("\n🦁 LeoSLM Training")
    print(f"   Device     : {device}")
    print(f"   Threads    : {torch.get_num_threads()}")
    print(f"   Batch size : {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")

    # ── Model ─────────────────────────────────────────────────────────────────
    config = get_config()
    model  = LeoSLM(config).to(device)

    params = model.count_params()
    print(f"   Parameters : {params['total']:,} ({params['total']/1e6:.1f}M)")

    # ── Compile (PyTorch 2.x speedup on CPU) ──────────────────────────────────
    try:
        model = torch.compile(model)
        print("   torch.compile: ✅")
    except Exception:
        print("   torch.compile: ❌ (skipped, needs PyTorch 2.x)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        betas        = (0.9, 0.95),
        weight_decay = 0.1,
        eps          = 1e-8,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step  = 0
    start_phase = 1
    if args.resume:
        start_step, start_phase = load_checkpoint(
            model, optimizer, f"{args.checkpoint_dir}/latest.pt", device
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    if not Path(args.train_data).exists():
        print(f"\n❌ Training data not found at {args.train_data}")
        print("   Run the data preparation step first:")
        print("   python3 prep_data.py")
        sys.exit(1)

    train_loader = build_loader(
        args.train_data, tokenizer=None,
        batch_size=args.batch_size, shuffle=True
    )
    print(f"   Dataset    : {len(train_loader.dataset):,} chunks")

    # ── Loss functions ─────────────────────────────────────────────────────────
    loss_fn = LeoLoss(
        mask_token_id  = config.mask_token_id,
        pad_token_id   = config.pad_token_id,
        lambda_mdm     = 0.0,
    )
    cal_loss_fn = ECTCalibrationLoss(lambda_cal=0.1)
    conditioner = ConstitutionalConditioner(hidden_dim=config.hidden_dim).to(device)
    const_loss_fn = ConstitutionalLoss(idk_token_id=config.idk_token_id)

    # ── Phase schedule ─────────────────────────────────────────────────────────
    phase_config = {
        1: {"epochs": 3,  "lr_max": 3e-4, "lr_min": 3e-5, "warmup": 200},
        2: {"epochs": 3,  "lr_max": 1e-4, "lr_min": 1e-5, "warmup": 100},
        3: {"epochs": 6,  "lr_max": 1e-4, "lr_min": 1e-5, "warmup": 100},
    }

    phases_to_run = [args.phase] if args.phase in [1, 2, 3] else [1, 2, 3]
    step          = start_step

    for phase in phases_to_run:
        if phase < start_phase and args.resume:
            print(f"  Skipping Phase {phase} (already done)")
            continue

        cfg   = phase_config[phase]
        step  = run_phase(
            phase          = phase,
            model          = model,
            optimizer      = optimizer,
            train_loader   = train_loader,
            device         = device,
            loss_fn        = loss_fn,
            cal_loss_fn    = cal_loss_fn,
            const_loss_fn  = const_loss_fn,
            conditioner    = conditioner,
            epochs         = cfg["epochs"],
            lr_max         = cfg["lr_max"],
            lr_min         = cfg["lr_min"],
            warmup_steps   = cfg["warmup"],
            grad_accum     = args.grad_accum,
            save_every     = args.save_every,
            start_step     = step,
            checkpoint_dir = args.checkpoint_dir,
        )

    print("\n🎉 All phases complete! Model saved to ./checkpoints/")
    print("   Run inference: python3 generate.py --mode hybrid")


if __name__ == "__main__":
    main()
