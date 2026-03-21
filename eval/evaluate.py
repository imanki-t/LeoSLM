"""
eval/evaluate.py — LeoSLM evaluation: perplexity + ECT calibration

No bugs found. Verified:
  - Correct token shift (logits[:,:-1] vs labels[:,1:])
  - Padding ignored in all metrics
  - AUROC fallback for systems without sklearn
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import LeoSLM, LeoConfig
from data  import LeoDataset


class LeoEvaluator:

    def __init__(self, model: LeoSLM, device: torch.device):
        self.model  = model
        self.device = device
        self.model.eval()
        self._pad_id = (
            getattr(getattr(model, "cfg",    None), "pad_id",       None) or
            getattr(getattr(model, "config", None), "pad_token_id", None) or
            0
        )

    @torch.no_grad()
    def evaluate_perplexity(
        self,
        dataloader:  DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        total_loss = total_tokens = n_batches = 0

        for batch in dataloader:
            if max_batches and n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            out       = self.model(input_ids)
            logits    = out["ar_logits"]

            # Correct shift: predict ids[:,1:] from logits[:,:-1]
            logits_s = logits[:, :-1, :].contiguous()
            labels_s = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                logits_s.view(-1, logits_s.shape[-1]),
                labels_s.view(-1),
                ignore_index=self._pad_id,
                reduction="sum",
            )
            n_tokens      = (labels_s != self._pad_id).sum().item()
            total_loss   += loss.item()
            total_tokens += n_tokens
            n_batches    += 1

        avg_loss = total_loss / max(total_tokens, 1)
        ppl      = math.exp(min(avg_loss, 20))
        return {
            "ppl":       ppl,
            "avg_loss":  avg_loss,
            "n_batches": n_batches,
            "n_tokens":  total_tokens,
        }

    @torch.no_grad()
    def evaluate_calibration(
        self,
        dataloader:  DataLoader,
        max_batches: Optional[int] = None,
        n_bins:      int = 10,
    ) -> Dict[str, float]:
        """
        ECE (Expected Calibration Error) for the ECT uncertainty head.
        A well-calibrated ECT should satisfy:
            P(error) ≈ U   for all U ∈ [0, 1]
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

        # ECE: weighted mean absolute difference between bin confidence and error rate
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

    def _compute_auroc(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(labels.numpy(), scores.numpy()))
        except ImportError:
            # Trapezoidal approximation
            thresholds         = torch.linspace(0, 1, 50)
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

    def full_eval(
        self,
        val_dataloader: DataLoader,
        max_batches:    Optional[int] = None,
    ) -> Dict[str, float]:
        print("Evaluating Perplexity...")
        ppl = self.evaluate_perplexity(val_dataloader, max_batches)
        print(f"  PPL      : {ppl['ppl']:.2f}")
        print(f"  Avg loss : {ppl['avg_loss']:.4f}")
        print(f"  Batches  : {ppl['n_batches']} | Tokens: {ppl['n_tokens']:,}")

        print("\nEvaluating ECT Calibration...")
        cal = self.evaluate_calibration(val_dataloader, max_batches)
        print(f"  ECE            : {cal['ece']:.4f}  (0 = perfect)")
        print(f"  Unc separation : {cal['unc_separation']:.4f}  (higher = better)")
        print(f"  U_correct      : {cal['unc_correct']:.4f}")
        print(f"  U_incorrect    : {cal['unc_incorrect']:.4f}")
        print(f"  AUROC          : {cal['auroc']:.3f}   (0.5 = random, 1.0 = perfect)")

        return {**ppl, **cal}


def main():
    parser = argparse.ArgumentParser(description="LeoSLM Aether Evaluation")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--data_path",   type=str, default="./data/val.npy")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--seq_len",     type=int, default=2048)
    parser.add_argument("--output",      type=str, default="./eval_results.json")
    parser.add_argument("--device",      type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device   : {device}")

    cfg   = LeoConfig()
    model = LeoSLM(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Warning  : {len(missing)} missing keys")
    model = model.to(device)

    if not Path(args.data_path).exists():
        print(f"  ERROR: Validation data not found: {args.data_path}")
        print("         Run prep_data.py to generate val.npy")
        sys.exit(1)

    dataset    = LeoDataset(args.data_path, max_seq_len=args.seq_len, pad_id=cfg.pad_id)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    print(f"  Dataset  : {len(dataset.data):,} tokens | {len(dataset)} chunks "
          f"| seq_len={args.seq_len}")

    evaluator = LeoEvaluator(model, device)
    results   = evaluator.full_eval(dataloader, max_batches=args.max_batches)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results["checkpoint"] = args.checkpoint
    results["seq_len"]    = args.seq_len
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {args.output}")


if __name__ == "__main__":
    main()
