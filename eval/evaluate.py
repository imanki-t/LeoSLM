"""
LeoSLM — Evaluation Suite
===========================
Evaluates LeoSLM on 4 key metrics:

    1. Perplexity (PPL)
       Standard LM quality metric. Lower = better.
       Computed via AR head on validation set.

    2. Expected Calibration Error (ECE)
       How well-calibrated are the ECT uncertainty scores?
       ECE = 0: perfect calibration (model knows exactly when it's wrong)
       ECE = 1: worst case

    3. Uncertainty Separation
       Are ECT scores higher for wrong predictions than correct ones?
       Good model: U_wrong >> U_correct
       Measures anti-hallucination effectiveness directly.

    4. MAUVE Score (optional, requires mauve-text package)
       Distribution-level text quality vs. reference corpus.
       Higher = better.

Run:
    python eval/evaluate.py --checkpoint ./checkpoints/best.pt
    python eval/evaluate.py --checkpoint ./checkpoints/best.pt --skip_mauve
"""

import torch
import torch.nn.functional as F
import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import argparse


class LeoEvaluator:
    """
    Evaluation suite for LeoSLM.

    Args:
        model  : LeoSLM model
        device : torch device
    """

    def __init__(self, model, device: torch.device):
        self.model  = model
        self.device = device
        self.model.eval()

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_perplexity(
        self,
        dataloader  : DataLoader,
        max_batches : Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute validation perplexity on AR head.
        Lower = better language model.
        """
        total_loss   = 0.0
        total_tokens = 0
        n_batches    = 0

        for batch in dataloader:
            if max_batches and n_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)    # (B, T)
            B, T      = input_ids.shape

            out    = self.model(input_ids)
            logits = out["ar_logits"]        # (B, T, V)

            # Shift for next-token prediction
            logits_s = logits[:, :-1, :].contiguous()
            labels_s = input_ids[:, 1:].contiguous()

            # Ignore PAD
            loss = F.cross_entropy(
                logits_s.view(-1, logits_s.shape[-1]),
                labels_s.view(-1),
                ignore_index = self.model.config.pad_token_id,
                reduction    = "sum",
            )

            # Count non-pad tokens
            n_tokens     = (labels_s != self.model.config.pad_token_id).sum().item()
            total_loss  += loss.item()
            total_tokens += n_tokens
            n_batches   += 1

        avg_loss = total_loss / max(total_tokens, 1)
        ppl      = math.exp(min(avg_loss, 20))

        return {
            "ppl"       : ppl,
            "avg_loss"  : avg_loss,
            "n_batches" : n_batches,
            "n_tokens"  : total_tokens,
        }

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_calibration(
        self,
        dataloader  : DataLoader,
        max_batches : Optional[int] = None,
        n_bins      : int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate ECT calibration quality.
        Measures how well uncertainty scores predict actual errors.

        Returns:
            ece              : Expected Calibration Error (lower = better)
            unc_separation   : mean(U_wrong) - mean(U_correct) (higher = better)
            auroc_uncertainty : AUROC of uncertainty score for error detection
        """
        all_uncertainty = []
        all_is_error    = []

        n_batches = 0
        for batch in dataloader:
            if max_batches and n_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)

            out        = self.model(input_ids)
            ar_logits  = out["ar_logits"]       # (B, T, V)
            uncertainty= out["uncertainty"]     # (B, T)

            # Shift
            logits_s = ar_logits[:, :-1, :]
            labels_s = input_ids[:, 1:]
            unc_s    = uncertainty[:, :-1]

            # Is model wrong?
            preds    = logits_s.argmax(dim=-1)   # (B, T-1)
            is_error = (preds != labels_s)       # (B, T-1) bool

            # Mask padding
            not_pad  = (labels_s != self.model.config.pad_token_id)
            unc_s    = unc_s[not_pad].cpu()
            is_error = is_error[not_pad].cpu()

            all_uncertainty.append(unc_s)
            all_is_error.append(is_error)
            n_batches += 1

        all_uncertainty = torch.cat(all_uncertainty)
        all_is_error    = torch.cat(all_is_error).float()

        # ── ECE (binned) ────────────────────────────────────────────────────
        ece    = 0.0
        N      = len(all_uncertainty)
        edges  = torch.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            lo, hi   = edges[i].item(), edges[i + 1].item()
            in_bin   = (all_uncertainty >= lo) & (all_uncertainty < hi)
            n_in_bin = in_bin.sum().item()
            if n_in_bin == 0:
                continue
            bin_conf = all_uncertainty[in_bin].mean().item()
            bin_err  = all_is_error[in_bin].mean().item()
            ece     += (n_in_bin / N) * abs(bin_conf - bin_err)

        # ── Uncertainty separation ───────────────────────────────────────────
        unc_correct   = all_uncertainty[all_is_error == 0].mean().item()
        unc_incorrect = all_uncertainty[all_is_error == 1].mean().item()
        separation    = unc_incorrect - unc_correct

        # ── AUROC (uncertainty as error predictor) ───────────────────────────
        auroc = self._compute_auroc(all_uncertainty, all_is_error)

        return {
            "ece"           : ece,
            "unc_correct"   : unc_correct,
            "unc_incorrect" : unc_incorrect,
            "unc_separation": separation,
            "auroc"         : auroc,
        }

    # -----------------------------------------------------------------------
    def _compute_auroc(
        self,
        scores : torch.Tensor,   # (N,) predicted uncertainty
        labels : torch.Tensor,   # (N,) float 0/1 (1=error)
    ) -> float:
        """Compute AUROC for uncertainty as error detector. Higher = better."""
        # Sort by score descending
        sorted_idx = scores.argsort(descending=True)
        sorted_lab = labels[sorted_idx]

        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tp = 0.0; fp = 0.0
        auc = 0.0
        prev_tp = 0.0; prev_fp = 0.0

        for lab in sorted_lab:
            if lab == 1:
                tp += 1
            else:
                fp += 1
                # Trapezoidal rule
                auc += (tp + prev_tp) / 2.0 * (1.0 / n_neg)
                prev_tp = tp
                prev_fp = fp

        return auc / n_pos if n_pos > 0 else 0.5

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_generation_quality(
        self,
        prompts        : List[torch.Tensor],
        max_new_tokens : int = 64,
        mode           : str = "hybrid",
    ) -> Dict[str, float]:
        """
        Evaluate generation quality:
            - Mean uncertainty on generated tokens (lower = more confident)
            - Fraction of tokens flagged for diffusion refinement
            - Mean number of diffusion steps needed

        Full MAUVE score requires mauve-text package (optional).
        """
        from diffusion import SelectiveDiffusionSampler

        sampler = SelectiveDiffusionSampler(
            model         = self.model,
            mask_token_id = self.model.config.mask_token_id,
        )

        mean_uncs  = []
        flagged_fracs = []

        for prompt in prompts:
            prompt = prompt.to(self.device)
            if mode == "hybrid":
                out_ids, info = sampler.hybrid_generate(prompt, max_new_tokens)
                unc = info["uncertainty"]
                if unc is not None:
                    prompt_len = prompt.shape[1]
                    gen_unc    = unc[:, prompt_len:]
                    mean_uncs.append(gen_unc.mean().item())
                    flagged    = (gen_unc > self.model.config.uncertainty_threshold)
                    flagged_fracs.append(flagged.float().mean().item())

        return {
            "mean_uncertainty"   : sum(mean_uncs) / max(len(mean_uncs), 1),
            "flagged_fraction"   : sum(flagged_fracs) / max(len(flagged_fracs), 1),
            "n_samples_evaluated": len(prompts),
        }

    # -----------------------------------------------------------------------
    def full_eval(
        self,
        val_dataloader : DataLoader,
        max_batches    : int = 50,
    ) -> Dict[str, float]:
        """
        Run all evaluations and return combined metrics.
        """
        print("── Evaluating Perplexity...")
        ppl_metrics = self.evaluate_perplexity(val_dataloader, max_batches)
        print(f"   PPL: {ppl_metrics['ppl']:.2f}")

        print("── Evaluating Calibration...")
        cal_metrics = self.evaluate_calibration(val_dataloader, max_batches)
        print(f"   ECE: {cal_metrics['ece']:.4f} (0=perfect)")
        print(f"   Uncertainty separation: {cal_metrics['unc_separation']:.4f} (higher=better)")
        print(f"   AUROC: {cal_metrics['auroc']:.3f} (0.5=random, 1.0=perfect)")

        return {**ppl_metrics, **cal_metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LeoSLM Evaluation")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--data_path",   type=str, default="./data/tinystories_val.npy")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--skip_mauve", action="store_true")
    parser.add_argument("--output",      type=str, default="./eval_results.json")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, ".")
    from model.leoSLM import LeoSLM, LeoConfig

    config = LeoConfig()
    model  = LeoSLM(config)
    ckpt   = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    evaluator = LeoEvaluator(model, torch.device("cpu"))

    # Load val data
    from data.dataset import LeoDataset, create_dataloader
    from data.dataset import LeoTokenizer

    print("Loading validation data...")
    tokenizer  = LeoTokenizer()
    val_ds     = LeoDataset(args.data_path, tokenizer)
    val_loader = create_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    results = evaluator.full_eval(val_loader, max_batches=args.max_batches)

    print("\n── Full Results ──────────────────")
    for k, v in results.items():
        print(f"  {k:<25}: {v:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
