"""
LeoSLM — Inference / Generation Script
=========================================
Three generation modes:
    1. --mode ar        : pure autoregressive (fast, standard)
    2. --mode diffusion : pure masked diffusion (iterative, parallel)
    3. --mode hybrid    : AR draft → ECT scan → selective diffusion (default, best quality)

Usage:
    python generate.py --prompt "Once upon a time" --mode hybrid
    python generate.py --prompt "What is 2+2?" --mode ar --max_tokens 64
    python generate.py --checkpoint ./checkpoints/leoslm_v1.pt --mode hybrid
"""

import torch
import torch.nn.functional as F
import argparse
import json
from pathlib import Path
from typing import Optional

from model.leoSLM     import LeoSLM, LeoConfig
from diffusion        import SelectiveDiffusionSampler


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class LeoGenerator:
    """
    High-level generation interface for LeoSLM.

    Args:
        model_path  : path to .pt checkpoint
        config      : LeoConfig (if None, loaded from checkpoint)
        device      : torch device
    """

    def __init__(
        self,
        model_path : Optional[str] = None,
        config     : Optional[LeoConfig] = None,
        device     : str = "cpu",
    ):
        self.device = torch.device(device)

        if config is None:
            config = LeoConfig()

        self.model = LeoSLM(config)

        if model_path and Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"✅ Loaded checkpoint from {model_path}")
        else:
            print("⚠️  No checkpoint loaded — using random weights (for testing)")

        self.model.to(self.device)
        self.model.eval()

        self.config = config

        self.sampler = SelectiveDiffusionSampler(
            model                  = self.model,
            mask_token_id          = config.mask_token_id,
            uncertainty_threshold  = config.uncertainty_threshold,
            num_diffusion_steps    = 10,
            temperature            = 0.8,
            top_k                  = 50,
            top_p                  = 0.9,
            consistency_votes      = 3,
            consistency_threshold  = 0.70,
        )

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids      : torch.Tensor,    # (1, T_prompt)
        mode           : str   = "hybrid",
        max_new_tokens : int   = 128,
        temperature    : float = 0.8,
        top_k          : int   = 50,
        verbose        : bool  = True,
    ) -> dict:
        """
        Generate tokens using specified mode.

        Returns dict with:
            output_ids   : (1, T) generated token ids
            uncertainty  : (1, T) per-token uncertainty (hybrid/diffusion only)
            flagged      : (1, T) positions refined by diffusion
            info         : mode-specific metadata
        """
        input_ids = input_ids.to(self.device)
        self.sampler.temperature = temperature
        self.sampler.top_k       = top_k

        if mode == "ar":
            output_ids = self._ar_generate(input_ids, max_new_tokens, temperature, top_k)
            return {
                "output_ids" : output_ids,
                "uncertainty": None,
                "flagged"    : None,
                "info"       : {"mode": "ar"},
            }

        elif mode == "diffusion":
            total_len = input_ids.shape[1] + max_new_tokens
            output_ids = self.sampler.pure_diffusion_generate(input_ids, total_len)
            # Get uncertainty on final output
            out = self.model(output_ids)
            return {
                "output_ids" : output_ids,
                "uncertainty": out["uncertainty"],
                "flagged"    : None,
                "info"       : {"mode": "diffusion"},
            }

        elif mode == "hybrid":
            output_ids, info = self.sampler.hybrid_generate(input_ids, max_new_tokens)
            if verbose:
                print(f"  Initial flagged tokens  : {info['initial_flagged']}")
                print(f"  Final flagged tokens    : {info['final_flagged']}")
                if info["step_stats"]:
                    final_unc = info["step_stats"][-1].get("mean_unc", 0)
                    print(f"  Final mean uncertainty  : {final_unc:.3f}")
            return {
                "output_ids" : output_ids,
                "uncertainty": info["uncertainty"],
                "flagged"    : None,
                "info"       : info,
            }

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from: ar, diffusion, hybrid")

    # -----------------------------------------------------------------------
    def _ar_generate(
        self,
        input_ids     : torch.Tensor,
        max_new_tokens: int,
        temperature   : float,
        top_k         : int,
    ) -> torch.Tensor:
        """Pure AR greedy/sampling generation."""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            out    = self.model(generated)
            logits = out["ar_logits"][:, -1, :]   # (1, V)

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    top_vals, _ = logits.topk(top_k, dim=-1)
                    threshold   = top_vals[:, -1].unsqueeze(-1)
                    logits      = logits.masked_fill(logits < threshold, float("-inf"))
                probs    = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tok], dim=1)

        return generated

    # -----------------------------------------------------------------------
    def print_uncertainty_map(
        self,
        tokens      : torch.Tensor,   # (1, T) token ids
        uncertainty : torch.Tensor,   # (1, T) uncertainty scores
        tokenizer   = None,
    ):
        """Print tokens color-coded by uncertainty (terminal visualization)."""
        t   = tokens[0].tolist()
        u   = uncertainty[0].tolist()
        tau = self.config.uncertainty_threshold

        print("\n── Uncertainty Map (" + "█" * 10 + " = high uncertainty) ──")
        for tid, unc in zip(t, u):
            if tokenizer:
                tok_str = tokenizer.decode(torch.tensor([tid]))
            else:
                tok_str = str(tid)

            bar_len = int(unc * 8)
            bar     = "█" * bar_len + "░" * (8 - bar_len)
            flag    = "⚠" if unc > tau else " "
            print(f"  {flag} [{bar}] {unc:.2f}  {tok_str!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LeoSLM Generation")
    parser.add_argument("--prompt",     type=str,   default="Once upon a time")
    parser.add_argument("--mode",       type=str,   default="hybrid",
                        choices=["ar", "diffusion", "hybrid"])
    parser.add_argument("--max_tokens", type=int,   default=64)
    parser.add_argument("--temperature",type=float, default=0.8)
    parser.add_argument("--top_k",     type=int,   default=50)
    parser.add_argument("--checkpoint",type=str,   default=None)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--show_uncertainty", action="store_true")
    args = parser.parse_args()

    print(f"\n🦁 LeoSLM Generator")
    print(f"   Mode       : {args.mode}")
    print(f"   Max tokens : {args.max_tokens}")
    print(f"   Prompt     : {args.prompt!r}")
    print()

    # Initialize (small config for testing)
    config = LeoConfig(
        vocab_size=50260,   # GPT-2 vocab + 3 special tokens
        hidden_dim=512,
        num_layers=16,
    )

    generator = LeoGenerator(
        model_path = args.checkpoint,
        config     = config,
        device     = args.device,
    )

    # Dummy tokenize (replace with real tokenizer when available)
    prompt_ids = torch.randint(3, 1000, (1, 8))
    print(f"Input ids: {prompt_ids.tolist()}")

    # Generate
    result = generator.generate(
        input_ids      = prompt_ids,
        mode           = args.mode,
        max_new_tokens = args.max_tokens,
        temperature    = args.temperature,
        top_k          = args.top_k,
        verbose        = True,
    )

    print(f"\nOutput shape: {result['output_ids'].shape}")
    print(f"Output ids (first 20): {result['output_ids'][0, :20].tolist()}")

    if result["uncertainty"] is not None and args.show_uncertainty:
        unc = result["uncertainty"]
        print(f"\nUncertainty stats:")
        print(f"  mean  : {unc.mean().item():.3f}")
        print(f"  max   : {unc.max().item():.3f}")
        print(f"  min   : {unc.min().item():.3f}")
        flagged = (unc > config.uncertainty_threshold).sum().item()
        print(f"  flagged ({config.uncertainty_threshold} threshold): {flagged} / {unc.numel()} tokens")

    print("\n✅ Generation complete")


if __name__ == "__main__":
    main()
