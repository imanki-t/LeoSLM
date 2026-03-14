"""
LeoSLM "Aether" — eval/evaluate.py
=====================================
Evaluates Leo on 4 key metrics:

    1. Perplexity (PPL)       — AR head on validation set. Lower = better.
    2. Expected Calibration   — ECT scores vs actual errors. ECE=0 is perfect.
    3. Uncertainty Separation — U_wrong >> U_correct. Higher = better.
    4. AUROC                  — ECT as error detector. 1.0 = perfect.

Run:
    python3 eval/evaluate.py --checkpoint ./checkpoints/latest.pt
    python3 eval/evaluate.py --checkpoint ./checkpoints/best_phase6.pt --skip_mauve
"""

import sys
import os
import math
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LeoEvaluator:
    """Evaluation suite for LeoSLM Aether."""

    def __init__(self, model, device: torch.device):
        self.model  = model
        self.device = device
        self.model.eval()
        # Support both old (.config.pad_token_id) and new (.cfg.pad_id) model shapes
        self._pad_id = (getattr(model, "cfg",    None) and getattr(model.cfg, "pad_id", None)
                        or getattr(model, "config", None) and getattr(model.config, "pad_token_id", None)
                        or 0)

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_perplexity(self, dataloader: DataLoader,
                             max_batches: Optional[int] = None) -> Dict[str, float]:
        """Compute validation perplexity on the AR head. Lower = better."""
        total_loss = total_tokens = n_batches = 0

        for batch in dataloader:
            if max_batches and n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            out       = self.model(input_ids)
            logits    = out["ar_logits"]

            logits_s = logits[:, :-1, :].contiguous()
            labels_s = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                logits_s.view(-1, logits_s.shape[-1]),
                labels_s.view(-1),
                ignore_index=self._pad_id,
                reduction="sum",
            )
            n_tokens     = (labels_s != self._pad_id).sum().item()
            total_loss  += loss.item()
            total_tokens += n_tokens
            n_batches   += 1

        avg_loss = total_loss / max(total_tokens, 1)
        ppl      = math.exp(min(avg_loss, 20))
        return {"ppl": ppl, "avg_loss": avg_loss,
                "n_batches": n_batches, "n_tokens": total_tokens}

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_calibration(self, dataloader: DataLoader,
                              max_batches: Optional[int] = None,
                              n_bins: int = 10) -> Dict[str, float]:
        """
        Evaluate ECT calibration.
        Good model: uncertainty score ≈ probability of being wrong.
        """
        all_unc: List[torch.Tensor] = []
        all_err: List[torch.Tensor] = []
        n_batches = 0

        for batch in dataloader:
            if max_batches and n_batches >= max_batches:
                break
            input_ids  = batch["input_ids"].to(self.device)
            out        = self.model(input_ids)
            ar_logits  = out["ar_logits"]
            uncertainty= out["uncertainty"]

            logits_s = ar_logits[:, :-1, :]
            labels_s = input_ids[:, 1:]
            unc_s    = uncertainty[:, :-1]

            preds    = logits_s.argmax(-1)
            is_error = (preds != labels_s)
            not_pad  = (labels_s != self._pad_id)

            all_unc.append(unc_s[not_pad].cpu())
            all_err.append(is_error[not_pad].float().cpu())
            n_batches += 1

        all_unc = torch.cat(all_unc)
        all_err = torch.cat(all_err)

        # ECE (binned)
        ece = 0.0
        N   = len(all_unc)
        for i in range(n_bins):
            lo, hi   = i / n_bins, (i + 1) / n_bins
            in_bin   = (all_unc >= lo) & (all_unc < hi)
            n_in_bin = in_bin.sum().item()
            if n_in_bin == 0:
                continue
            bin_conf = all_unc[in_bin].mean().item()
            bin_err  = all_err[in_bin].mean().item()
            ece     += (n_in_bin / N) * abs(bin_conf - bin_err)

        unc_correct   = all_unc[all_err == 0].mean().item()
        unc_incorrect = all_unc[all_err == 1].mean().item()
        separation    = unc_incorrect - unc_correct
        auroc         = self._compute_auroc(all_unc, all_err)

        return {
            "ece":            ece,
            "unc_correct":    unc_correct,
            "unc_incorrect":  unc_incorrect,
            "unc_separation": separation,
            "auroc":          auroc,
        }

    def _compute_auroc(self, scores: torch.Tensor,
                        labels: torch.Tensor) -> float:
        """AUROC of uncertainty score as error detector."""
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(labels.numpy(), scores.numpy()))
        except ImportError:
            # Fallback: trapezoidal rule
            thresholds = torch.linspace(0, 1, 50)
            tpr_list, fpr_list = [], []
            P = labels.sum().item()
            N = len(labels) - P
            if P == 0 or N == 0:
                return 0.5
            for t in thresholds:
                pred_pos = scores >= t
                tp = ((pred_pos) & (labels == 1)).sum().float().item()
                fp = ((pred_pos) & (labels == 0)).sum().float().item()
                tpr_list.append(tp / P)
                fpr_list.append(fp / N)
            auroc = 0.0
            for i in range(len(fpr_list) - 1):
                auroc += (fpr_list[i] - fpr_list[i+1]) * (tpr_list[i] + tpr_list[i+1]) / 2
            return abs(auroc)

    # -------------------------------------------------------------------------
    def full_eval(self, val_dataloader: DataLoader,
                  max_batches: Optional[int] = None) -> Dict[str, float]:
        print("── Evaluating Perplexity...")
        ppl = self.evaluate_perplexity(val_dataloader, max_batches)
        print(f"   PPL: {ppl['ppl']:.2f}")

        print("── Evaluating Calibration...")
        cal = self.evaluate_calibration(val_dataloader, max_batches)
        print(f"   ECE:              {cal['ece']:.4f}  (0=perfect)")
        print(f"   Unc separation:   {cal['unc_separation']:.4f}  (higher=better)")
        print(f"   AUROC:            {cal['auroc']:.3f}   (0.5=random, 1.0=perfect)")

        return {**ppl, **cal}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LeoSLM Aether — Evaluation")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--data_path",   type=str, default="./data/val.npy")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--output",      type=str, default="./eval_results.json")
    args = parser.parse_args()

    # ── Import model from train.py monolith ──────────────────────────────────
    # train.py is self-contained: LeoSLM and LeoConfig are defined there.
    # Do NOT import from model/leoSLM.py — that is the old v1 architecture.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train import LeoSLM, LeoConfig

    device = torch.device("cpu")

    cfg   = LeoConfig()
    model = LeoSLM(cfg)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys in checkpoint")
    model.eval()
    print(f"Loaded: {args.checkpoint}")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # ── Simple numpy dataset (no dependency on data/dataset.py) ──────────────
    class _ValDataset(torch.utils.data.Dataset):
        def __init__(self, path, max_len=2048, pad_id=0):
            self.data    = np.load(path, mmap_mode="r")
            self.max_len = max_len
            self.pad_id  = pad_id
        def __len__(self):
            return max(1, len(self.data) // self.max_len)
        def __getitem__(self, idx):
            s     = idx * self.max_len
            chunk = self.data[s: s + self.max_len].astype(np.int64)
            if len(chunk) < self.max_len:
                chunk = np.pad(chunk, (0, self.max_len - len(chunk)),
                               constant_values=self.pad_id)
            ids  = torch.tensor(chunk, dtype=torch.long)
            mask = (ids != self.pad_id).long()
            return {"input_ids": ids, "attention_mask": mask}

    if not Path(args.data_path).exists():
        print(f"Validation data not found: {args.data_path}")
        print("Run prep_data.py first to generate data/val.npy")
        return

    val_ds     = _ValDataset(args.data_path, max_len=2048, pad_id=cfg.pad_id)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, drop_last=False)
    print(f"Val set: {len(val_ds):,} sequences")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluator = LeoEvaluator(model, device)
    results   = evaluator.full_eval(val_loader, max_batches=args.max_batches)

    print("\n── Full Results ──────────────────────────────────────")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        else:
            print(f"  {k:<25}: {v}")

    with open(args.output, "w") as f:
        json.dump({k: float(v) if isinstance(v, (int, float)) else v
                   for k, v in results.items()}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
