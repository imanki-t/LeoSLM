"""
model/leo_slm.py — LeoSLM "Aether" — Full Model Assembly
==========================================================
~3.1B total parameters | ~1.9B active (MoE) | 32k trained | 128k via YaRN

Architecture (Aether):
    Input tokens
        → Embedding (65543 × 2560)
        → Initial ECT pass   — seeds U before transformer blocks
        → SAM prefix         — structured agentic memory slots
        → 32 × LeoBlock
            ├─ MLA dual-path attention (causal + bidir, gated by α)
            ├─ UWMR MoE FFN (blocks 4-31) / Dense SwiGLU (blocks 0-3)
            └─ TDM rolling memory compression at each chunk boundary
        → Final ECT pass     — produces U_final across full sequence
        → ACGI               — per-token tool-call gate
        → RMSNorm
        → AR head  (lm_head, weight-tied to embedding)
        → Diff head (diff_head, separate weights)
        → MTP heads (N=4 future positions, optional)

V1 features merged into Aether:
    _noise_embedding()        — sinusoidal diffusion timestep embedding (v1)
    get_inference_output()    — hybrid AR/diffusion inference helper (v1)
    count_params_detailed()   — per-component parameter breakdown (v1)
    alpha_list in output dict — per-layer gate values for analysis (v1)
    denoise_head alias        — diff_head aliased as denoise_head for compat (v1)
    HardThresholdGate         — thresholded uncertain-token mask (v1)

Imports:
    All components imported from their own module files.
    No class definitions live here except LeoSLM itself.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .config    import LeoConfig, CFG
from .identity  import LEO_IDENTITY
from .norm      import RMSNorm
from .ect       import ECTv3Module
from .memory    import TemporalDiffusionMemory, StructuredAgenticMemory
from .attention import DSALite
from .moe       import ExpertFFN
from .mtp       import MultiTokenPredictionHead
from .agentic   import AgenticConfidenceGatedInvocation
from .leo_block import LeoBlock


# ── V1 compatibility helper ────────────────────────────────────────────────────

class HardThresholdGate:
    """
    V1 inference helper: binary mask for uncertain token positions.
    Tokens with U > threshold are flagged for diffusion refinement.
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def get_uncertain_positions(
        self,
        uncertainty: torch.Tensor,
        input_ids:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            flagged   : (B, T) bool — True = uncertain → use diffusion logits
            threshold : float
        """
        flagged = uncertainty > self.threshold
        return flagged, self.threshold


# ══════════════════════════════════════════════════════════════════════════════
# LeoSLM — Full Model
# ══════════════════════════════════════════════════════════════════════════════

class LeoSLM(nn.Module):
    """
    LeoSLM "Aether" — built by Unmuted.

    Forward pass returns a dict with:
        ar_logits     : (B, T, V)         — AR head logits
        diff_logits   : (B, T, V)         — diffusion / denoise head logits
        uncertainty   : (B, T)            — ECT uncertainty ∈ [0,1]
        hidden        : (B, T, D)         — final hidden states (pre-norm)
        aux_loss      : scalar            — MoE + TDM + SAM auxiliary losses
        prm_scores    : (B, T) | None     — CoT step quality scores
        think_mask    : (B, T) bool       — positions inside <think>…</think>
        tool_mask     : (B, T) bool       — positions inside tool-call spans
        acgi_gate     : (B, T)            — ACGI gate scores
        acgi_gate_mask: (B, T) bool       — ACGI hard gate (True = invoke tool)
        tool_logits   : (B, T, n_tools)   — tool routing logits
        mtp_logits    : List[(B, T, V)]   — N future-token predictions (phase 3+)
        alpha_list    : List[(B, T)]      — per-layer gate values (v1 compat)

    Args:
        cfg : LeoConfig (default: module-level CFG singleton)
    """

    def __init__(self, cfg: LeoConfig = CFG):
        super().__init__()
        self.cfg      = cfg
        self.identity = LEO_IDENTITY
        D, V          = cfg.hidden_dim, cfg.vocab_size

        # ── Embedding ─────────────────────────────────────────────────────────
        self.tok_embed  = nn.Embedding(V, D)

        # ── Epistemic Confidence Tokens ────────────────────────────────────────
        self.ect        = ECTv3Module(cfg)

        # ── Temporal Diffusion Memory ──────────────────────────────────────────
        self.tdm        = TemporalDiffusionMemory(cfg)

        # ── Decoder blocks ─────────────────────────────────────────────────────
        self.blocks     = nn.ModuleList([LeoBlock(cfg, i) for i in range(cfg.num_layers)])

        # ── Final norm + output heads ─────────────────────────────────────────
        self.final_norm = RMSNorm(D)
        self.lm_head    = nn.Linear(D, V, bias=False)
        self.diff_head  = nn.Linear(D, V, bias=False)

        # V1 alias: denoise_head → diff_head (keeps old checkpoints loadable)
        self.denoise_head = self.diff_head

        # ── V1: HardThresholdGate for hybrid inference ─────────────────────────
        self.hard_gate  = HardThresholdGate(threshold=cfg.uncertainty_thresh)

        # ── Multi-Token Prediction ─────────────────────────────────────────────
        if cfg.use_mtp:
            self.mtp    = MultiTokenPredictionHead(cfg)

        # ── DSA-lite (sparse attention for >32k contexts) ──────────────────────
        self.dsa        = DSALite(cfg)

        # ── ACGI: Agentic Confidence-Gated Invocation ──────────────────────────
        self.acgi       = AgenticConfidenceGatedInvocation(cfg)

        # ── SAM: Structured Agentic Memory ─────────────────────────────────────
        if cfg.use_sam:
            self.sam    = StructuredAgenticMemory(cfg)

        # ── Weight tying ─────────────────────────────────────────────────────
        if cfg.weight_tying:
            self.lm_head.weight = self.tok_embed.weight
            if cfg.use_mtp:
                for proj in self.mtp.projs:
                    proj.weight = self.tok_embed.weight

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        """Scaled normal init (LLaMA-style: std ∝ 1/√(2×num_layers))."""
        std = 0.02 / math.sqrt(2 * self.cfg.num_layers)
        for n, p in self.named_parameters():
            if p.dim() >= 2 and "weight" in n:
                nn.init.normal_(p, mean=0.0, std=std)
            elif "bias" in n:
                nn.init.zeros_(p)

    # ── V1: sinusoidal noise-level embedding ──────────────────────────────────

    def _noise_embedding(
        self,
        noise_level: torch.Tensor,
        device:      torch.device,
    ) -> torch.Tensor:
        """
        Sinusoidal embedding for diffusion noise level t ∈ [0,1].
        Matches v1 API: noise_level (B,) → (B, hidden_dim).
        """
        D    = self.cfg.hidden_dim
        half = D // 2
        freqs = torch.exp(
            -torch.arange(half, device=device).float()
            * (math.log(10000.0) / (half - 1))
        )
        args = noise_level.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, D)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:   torch.Tensor,
        noise_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids   : (B, T) — token ids; may contain MASK tokens for diffusion
            noise_level : (B,)   — diffusion timestep t ∈ [0,1] (v1 compat, optional)
        """
        B, T = input_ids.shape
        C    = self.cfg.chunk_size

        # ── Region masks ──────────────────────────────────────────────────────
        # Scan once for <think> and tool-call spans (XLA: no dynamic shapes)
        think_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        tool_mask  = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(B):
            in_think = in_tool = False
            for t in range(T):
                tok = input_ids[b, t].item()
                if tok == self.cfg.think_start_id:  in_think = True
                if in_think:                         think_mask[b, t] = True
                if tok == self.cfg.think_end_id:    in_think = False
                if tok == self.cfg.tool_call_start: in_tool = True
                if in_tool:                          tool_mask[b, t] = True
                if tok == self.cfg.tool_result_end: in_tool = False

        # ── Embed ─────────────────────────────────────────────────────────────
        x = self.tok_embed(input_ids)                                  # (B, T, D)

        # V1 noise conditioning (diffusion timestep injection)
        if noise_level is not None:
            x = x + 0.05 * noise_level.float().view(B, 1, 1) * torch.ones_like(x)

        # ── Initial ECT pass — seed U before transformer blocks ───────────────
        _ect_h, U, prm_init = self.ect(x, is_think=think_mask)

        # ── SAM prefix ────────────────────────────────────────────────────────
        sam_prefix = None
        sam_aux    = x.new_zeros(1)
        if self.cfg.use_sam:
            sam_prefix, sam_aux = self.sam(x)

        # ── Chunked transformer blocks ─────────────────────────────────────────
        all_hidden  = torch.zeros_like(x)
        mem_tokens  = sam_prefix
        total_aux   = sam_aux
        alpha_list  = []                          # V1 compat: collect per-block alphas
        n_chunks    = math.ceil(T / C)

        for ci in range(n_chunks):
            s     = ci * C
            e     = min(s + C, T)
            chunk = x[:, s:e]
            Uc    = U[:, s:e]

            chunk_alphas = []
            for bi, block in enumerate(self.blocks):
                mt          = mem_tokens if bi == 0 else None
                chunk, aux, alpha = block(chunk, U=Uc, mem=mt)
                total_aux   = total_aux + aux
                chunk_alphas.append(alpha)

            all_hidden[:, s:e] = chunk
            # Aggregate alpha per chunk as mean across blocks (for logging)
            if chunk_alphas:
                alpha_list.append(
                    torch.stack(chunk_alphas, dim=0).mean(0)          # (B, chunk_T)
                )

            # TDM: compress confident chunk states → rolling memory
            if ci < n_chunks - 1:
                _, Uc_ref, _ = self.ect(chunk)
                mem_tokens, tdm_loss = self.tdm(chunk, Uc_ref)
                total_aux    = total_aux + tdm_loss

        # ── Final ECT pass (full sequence) ────────────────────────────────────
        ect_h_final, U_final, prm_final = self.ect(all_hidden, is_think=think_mask)

        # ── ACGI: per-token tool-call gate ────────────────────────────────────
        acgi_out = self.acgi(all_hidden, U_final, ect_h_final)

        # ── Output heads ──────────────────────────────────────────────────────
        h_norm      = self.final_norm(all_hidden)
        ar_logits   = self.lm_head(h_norm)
        diff_logits = self.diff_head(h_norm)

        # ── MTP heads (active from phase 3 onwards) ───────────────────────────
        mtp_logits = None
        if self.cfg.use_mtp:
            mtp_logits = self.mtp(h_norm)

        # V1 compat: flatten alpha_list to cover full sequence
        full_alpha = torch.cat(alpha_list, dim=1) if alpha_list else None

        return {
            "ar_logits":      ar_logits,
            "diff_logits":    diff_logits,
            "uncertainty":    U_final,
            "hidden":         all_hidden,
            "aux_loss":       total_aux,
            "prm_scores":     prm_final,
            "think_mask":     think_mask,
            "tool_mask":      tool_mask,
            "acgi_gate":      acgi_out["gate_score"],
            "acgi_gate_mask": acgi_out["gate_mask"],
            "tool_logits":    acgi_out["tool_logits"],
            "mtp_logits":     mtp_logits,
            "alpha_list":     full_alpha,    # (B, T) mean gate — v1 compat
            "ect_final":      ect_h_final,   # (B, E, D)        — v1 compat
        }

    # ── V1: hybrid inference helper ───────────────────────────────────────────

    def get_inference_output(
        self,
        input_ids: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference helper (v1 API preserved).
        Merges AR and diffusion logits per-token based on ECT uncertainty:
            U ≤ threshold → AR logits (confident)
            U >  threshold → diffusion logits (uncertain, gets refinement)

        Args:
            input_ids : (B, T)
            threshold : override uncertainty_thresh from config (optional)
        Returns:
            dict with "logits" key (merged) plus all standard forward keys
        """
        if threshold is not None:
            self.hard_gate.threshold = threshold

        out         = self.forward(input_ids)
        ar_logits   = out["ar_logits"]
        diff_logits = out["diff_logits"]
        uncertainty = out["uncertainty"]

        flagged, _  = self.hard_gate.get_uncertain_positions(uncertainty, input_ids)
        flagged_3d  = flagged.unsqueeze(-1).expand_as(ar_logits)

        merged_logits = torch.where(flagged_3d, diff_logits, ar_logits)

        return {
            "logits":      merged_logits,
            "uncertainty": uncertainty,
            "flagged":     flagged,
            **out,
        }

    # ── Think-budget gating ───────────────────────────────────────────────────

    def get_think_budget(self, U: torch.Tensor) -> int:
        """
        Decide how many think tokens to allocate based on mean ECT uncertainty.
        High U → full budget (hard problem); low U → zero (no-think mode).
        """
        mean_U = U.mean().item()
        if mean_U > self.cfg.ect_spawn_thresh:
            return self.cfg.think_budget_max
        elif mean_U > 0.4:
            return self.cfg.think_budget_max // 2
        elif mean_U > 0.2:
            return self.cfg.think_budget_max // 4
        return 0

    # ── Phase-wise parameter freezing ────────────────────────────────────────

    def freeze_phase(self, phase: int):
        """
        Freeze / unfreeze parameter groups for each training phase.
        Called at the start of each phase in run_phase().
        """
        if phase == 1:
            # AR warmup: freeze ECT, diffusion, agentic — backbone + lm_head only
            for p in self.ect.parameters():       p.requires_grad_(False)
            for p in self.diff_head.parameters(): p.requires_grad_(False)
            for p in self.tdm.parameters():       p.requires_grad_(False)
            for p in self.acgi.parameters():      p.requires_grad_(False)
            if self.cfg.use_sam:
                for p in self.sam.parameters():   p.requires_grad_(False)
            if self.cfg.use_mtp:
                for p in self.mtp.parameters():   p.requires_grad_(False)

        elif phase == 2:
            # Diffusion warmup: add diff_head; everything else still from phase 1
            for p in self.diff_head.parameters(): p.requires_grad_(True)

        elif phase in (3, 4, 5, 6):
            # Joint / SFT / DPO / GRPO: everything except agentic modules
            for p in self.parameters():            p.requires_grad_(True)
            for p in self.acgi.parameters():       p.requires_grad_(False)
            if self.cfg.use_sam:
                for p in self.sam.parameters():    p.requires_grad_(False)

        elif phase == 7:
            # Agentic SFT: freeze backbone; unfreeze ACGI + SAM + lm_head + MTP
            for p in self.parameters():            p.requires_grad_(False)
            for p in self.acgi.parameters():       p.requires_grad_(True)
            if self.cfg.use_sam:
                for p in self.sam.parameters():    p.requires_grad_(True)
            for p in self.lm_head.parameters():    p.requires_grad_(True)
            if self.cfg.use_mtp:
                for p in self.mtp.parameters():    p.requires_grad_(True)

        elif phase == 8:
            # Agentic RL: full unfreeze for end-to-end trajectory credit
            for p in self.parameters():            p.requires_grad_(True)

    # ── Parameter counts ─────────────────────────────────────────────────────

    def count_params(self) -> Dict[str, int]:
        """V1 API: per-component parameter counts."""
        def n(m): return sum(p.numel() for p in m.parameters())

        embedding_count = n(self.tok_embed)
        blocks_count    = n(self.blocks)
        ect_count       = n(self.ect)
        tdm_count       = n(self.tdm)
        acgi_count      = n(self.acgi)
        sam_count       = n(self.sam) if self.cfg.use_sam else 0
        mtp_count       = n(self.mtp) if self.cfg.use_mtp else 0
        # lm_head is tied → 0 extra; diff_head is separate
        lm_head_count   = 0 if self.cfg.weight_tying else n(self.lm_head)
        diff_head_count = n(self.diff_head)
        total           = n(self)

        # Approximate active params: subtract inactive MoE expert params
        try:
            inactive_expert_params = sum(
                sum(p.numel() for p in b.ffn.experts[b.ffn.top_k:].parameters())
                for b in self.blocks if b.is_moe
            )
        except AttributeError:
            inactive_expert_params = 0
        approx_active = total - inactive_expert_params

        return {
            "embedding":     embedding_count,
            "blocks":        blocks_count,
            "ect":           ect_count,
            "tdm":           tdm_count,
            "acgi":          acgi_count,
            "sam":           sam_count,
            "mtp":           mtp_count,
            "lm_head":       lm_head_count,
            "diff_head":     diff_head_count,
            "total":         total,
            "approx_active": approx_active,
        }
