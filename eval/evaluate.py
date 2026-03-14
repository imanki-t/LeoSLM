"""
LeoSLM "Aether" — eval/evaluate.py
=====================================
Evaluates Leo on 4 core metrics.  All imports come from model/ — nothing
from train.py.

Metrics:
    1. Perplexity (PPL)         — AR head cross-entropy. Lower = better.
    2. Expected Calibration Error (ECE) — ECT scores vs actual errors. 0 = perfect.
    3. Uncertainty Separation   — U_wrong >> U_correct. Higher = better.
    4. AUROC                    — ECT as per-token error detector. 1.0 = perfect.

Usage:
    python3 eval/evaluate.py --checkpoint ./checkpoints/latest.pt
    python3 eval/evaluate.py --checkpoint ./checkpoints/best_phase6.pt \
                             --data_path ./data/val.npy --max_batches 200
"""

import sys
import os
import math
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import DataLoader

# ── All project imports from model/ ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import LeoSLM, LeoConfig       # noqa: E402
from data  import LeoDataset              # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class LeoEvaluator:
    """Evaluation suite for LeoSLM Aether."""

    def __init__(self, model: LeoSLM, device: torch.device):
        self.model  = model
        self.device = device
        self.model.eval()
        # Support both attribute naming conventions for pad token
        self._pad_id = (
            getattr(getattr(model, "cfg",    None), "pad_id",        None) or
            getattr(getattr(model, "config", None), "pad_token_id",  None) or
            0
        )

    # ── Perplexity ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_perplexity(
        self,
        dataloader:  DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute validation perplexity on the AR head. Lower = better."""
        total_loss = total_tokens = n_batches = 0

        for batch in dataloader:
            if max_batches and n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            out       = self.model(input_ids)
            logits    = out["ar_logits"]

            logits_s  = logits[:, :-1, :].contiguous()
            labels_s  = input_ids[:, 1:].contiguous()

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
        ppl      = math.exp(min(avg_loss, 20))   # cap at exp(20) to avoid inf
        return {
            "ppl":       ppl,
            "avg_loss":  avg_loss,
            "n_batches": n_batches,
            "n_tokens":  total_tokens,
        }

    # ── Calibration ────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_calibration(
        self,
        dataloader:  DataLoader,
        max_batches: Optional[int] = None,
        n_bins:      int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate ECT calibration:
            ECE              — Expected Calibration Error (0 = perfect)
            unc_separation   — U_wrong − U_correct (higher = better)
            AUROC            — ECT as error detector (1.0 = perfect)

        A well-calibrated model has ECT scores that match per-token error
        probability: uncertain tokens should actually be wrong more often.
        """
        all_unc: List[torch.Tensor] = []
        all_err: List[torch.Tensor] = []
        n_batches = 0

        for batch in dataloader:
            if max_batches and n_batches >= max_batches:
                break
            input_ids   = batch["input_ids"].to(self.device)
            out         = self.model(input_ids)
            ar_logits   = out["ar_logits"]
            uncertainty = out["uncertainty"]

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

        # ── ECE (binned) ──────────────────────────────────────────────────
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

    def _compute_auroc(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """AUROC of ECT uncertainty as a binary error detector."""
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(labels.numpy(), scores.numpy()))
        except ImportError:
            # Fallback: trapezoidal rule
            thresholds        = torch.linspace(0, 1, 50)
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

    # ── Combined ───────────────────────────────────────────────────────────

    def full_eval(
        self,
        val_dataloader: DataLoader,
        max_batches:    Optional[int] = None,
    ) -> Dict[str, float]:
        print("── Evaluating Perplexity...")
        ppl = self.evaluate_perplexity(val_dataloader, max_batches)
        print(f"   PPL       : {ppl['ppl']:.2f}")
        print(f"   Avg loss  : {ppl['avg_loss']:.4f}")
        print(f"   Batches   : {ppl['n_batches']} | Tokens: {ppl['n_tokens']:,}")

        print("\n── Evaluating ECT Calibration...")
        cal = self.evaluate_calibration(val_dataloader, max_batches)
        print(f"   ECE              : {cal['ece']:.4f}  (0 = perfect)")
        print(f"   Unc separation   : {cal['unc_separation']:.4f}  (higher = better)")
        print(f"   U_correct        : {cal['unc_correct']:.4f}")
        print(f"   U_incorrect      : {cal['unc_incorrect']:.4f}")
        print(f"   AUROC            : {cal['auroc']:.3f}   (0.5 = random, 1.0 = perfect)")

        return {**ppl, **cal}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LeoSLM Aether — Evaluation")
    parser.add_argument("--checkpoint",  type=str,   required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--data_path",   type=str,   default="./data/val.npy",
                        help="Path to uint32 .npy validation token file")
    parser.add_argument("--max_batches", type=int,   default=100,
                        help="Max batches per metric (None = full val set)")
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--seq_len",     type=int,   default=2048)
    parser.add_argument("--output",      type=str,   default="./eval_results.json",
                        help="Save results to this JSON file")
    parser.add_argument("--device",      type=str,   default=None)
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device   : {device}")

    # ── Model ───────────────────────────────────────────────────────────────
    cfg   = LeoConfig()
    model = LeoSLM(cfg)

    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in state.items()
    }
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Warning  : {len(missing)} missing keys")
    model = model.to(device)

    # ── Dataset + DataLoader ────────────────────────────────────────────────
    if not Path(args.data_path).exists():
        print(f"  ERROR: Validation data not found: {args.data_path}")
        print("         Run prep_data.py to generate val.npy")
        sys.exit(1)

    dataset    = LeoDataset(args.data_path, max_seq_len=args.seq_len, pad_id=cfg.pad_id)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"  Dataset  : {len(dataset.data):,} tokens | {len(dataset)} chunks | seq_len={args.seq_len}")
    print(f"  Max eval batches: {args.max_batches}")
    print()

    # ── Evaluate ────────────────────────────────────────────────────────────
    evaluator = LeoEvaluator(model, device)
    results   = evaluator.full_eval(dataloader, max_batches=args.max_batches)

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results["checkpoint"] = args.checkpoint
    results["seq_len"]    = args.seq_len

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {args.output}")


if __name__ == "__main__":
    main()
