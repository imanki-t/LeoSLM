import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .config    import LeoConfig, CFG
from .identity  import LEO_IDENTITY
from .norm      import RMSNorm
from .rope      import build_yarn_rope_cache
from .ect       import ECTv3Module
from .memory    import TemporalDiffusionMemory, StructuredAgenticMemory
from .moe       import UWMRMoE
from .mtp       import MultiTokenPredictionHead
from .agentic   import AgenticConfidenceGatedInvocation
from .leo_block import LeoBlock
from .attention import DSALite


class HardThresholdGate:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, U: torch.Tensor) -> torch.Tensor:
        return (U > self.threshold).float()


class LeoSLM(nn.Module):

    def __init__(self, cfg: LeoConfig = CFG):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        assert D % cfg.num_heads == 0, (
            f"hidden_dim {D} must be divisible by num_heads {cfg.num_heads}"
        )

        self.embed      = nn.Embedding(cfg.vocab_size, D)
        self.blocks     = nn.ModuleList([LeoBlock(cfg, i) for i in range(cfg.num_layers)])
        self.final_norm = RMSNorm(D)

        self.lm_head  = nn.Linear(D, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.diff_head    = nn.Linear(D, cfg.vocab_size, bias=False)
        self.denoise_head = self.diff_head

        self.ect  = ECTv3Module(cfg)
        self.acgi = AgenticConfidenceGatedInvocation(cfg)

        self.tdm = TemporalDiffusionMemory(cfg)   if cfg.use_tdm else None
        self.sam = StructuredAgenticMemory(cfg)   if cfg.use_sam else None
        self.dsa = DSALite(cfg)                   if cfg.use_dsa else None

        if cfg.use_mtp:
            self.mtp = MultiTokenPredictionHead(cfg)
            if self.mtp.projs:
                self.mtp.projs[0].weight = self.lm_head.weight
        else:
            self.mtp = None

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.zeros_(p)
            elif "embed" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def freeze_phase(self, phase: int) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
        if phase == 1:
            for mod in [self.mtp, self.acgi, self.tdm]:
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)
        elif phase == 2:
            for p in self.acgi.parameters():
                p.requires_grad_(False)
        elif phase == 5:
            for mod in [self.acgi, self.mtp]:
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)
        elif phase in (7, 8):
            for p in self.embed.parameters():
                p.requires_grad_(False)

    def get_think_budget(self, U: torch.Tensor) -> int:
        mean_u = U.mean().item()
        thr    = self.cfg.uncertainty_thresh * 0.5
        if mean_u < thr:
            return 0
        ratio  = (mean_u - thr) / max(1.0 - thr, 1e-8)
        budget = int(
            self.cfg.think_budget_min
            + ratio * (self.cfg.think_budget_max - self.cfg.think_budget_min)
        )
        return max(self.cfg.think_budget_min, min(budget, self.cfg.think_budget_max))

    def _build_span_mask(
        self,
        ids:                torch.Tensor,
        open_id:            int,
        close_id:           int,
        include_delimiters: bool = False,
    ) -> torch.Tensor:
        is_open  = (ids == open_id).float()
        is_close = (ids == close_id).float()
        open_cs  = torch.cumsum(is_open,  dim=1)
        close_cs = torch.cumsum(is_close, dim=1)
        inside   = (open_cs > close_cs).float()
        mask     = inside * (1.0 - is_open)
        if include_delimiters:
            mask = (mask + is_open + is_close).clamp(0.0, 1.0)
        return mask

    def _noise_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.cfg.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device).float()
            / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(
        self,
        input_ids:   torch.Tensor,
        noise_t:     Optional[torch.Tensor] = None,
        sam_slots:   Optional[torch.Tensor] = None,
        tool_labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        B, T   = input_ids.shape
        device = input_ids.device

        h = self.embed(input_ids)

        if noise_t is not None:
            h = h + self._noise_embedding(noise_t).unsqueeze(1)

        think_mask = self._build_span_mask(input_ids, self.cfg.think_start_id, self.cfg.think_end_id)
        tool_mask  = self._build_span_mask(input_ids, self.cfg.tool_call_start, self.cfg.tool_call_end, include_delimiters=True)

        ect_h, U, prm_scores = self.ect(h, is_think=think_mask)

        if sam_slots is not None:
            sam_mem = sam_slots
        elif self.sam is not None:
            sam_mem = self.sam.get_slots(B, device)
        else:
            sam_mem = None

        mem       = sam_mem
        aux_total = h.new_zeros(1)
        alpha_list: List[torch.Tensor] = []
        h_chunks:   List[torch.Tensor] = []

        chunk_size = self.cfg.chunk_size
        n_chunks   = math.ceil(T / chunk_size)

        for ci in range(n_chunks):
            s   = ci * chunk_size
            e   = min(s + chunk_size, T)
            h_c = h[:, s:e, :]
            U_c = U[:, s:e]

            for li, block in enumerate(self.blocks):
                inj_mem = mem if li == 0 else None
                h_c, aux, alpha = block(h_c, U=U_c, mem=inj_mem)
                aux_total = aux_total + aux
                if ci == 0:
                    alpha_list.append(alpha)

            if self.tdm is not None:
                tdm_mem, _ = self.tdm(h_c, U_c)
                mem = torch.cat([sam_mem, tdm_mem], dim=1) if sam_mem is not None else tdm_mem

            h_chunks.append(h_c)

        h = torch.cat(h_chunks, dim=1)

        if T >= self.cfg.dsa_threshold and self.dsa is not None:
            h = self.dsa(h, U)

        _, U_final, prm_final = self.ect(h, is_think=think_mask)

        acgi_out    = self.acgi(h, U_final, ect_h)
        h_norm      = self.final_norm(h)
        ar_logits   = self.lm_head(h_norm)
        diff_logits = self.diff_head(h_norm)
        mtp_logits  = self.mtp(h_norm) if self.mtp is not None else None

        return {
            "ar_logits":   ar_logits,
            "diff_logits": diff_logits,
            "uncertainty": U_final,
            "aux_loss":    aux_total,
            "mtp_logits":  mtp_logits,
            "acgi_gate":   acgi_out["gate_score"],
            "tool_logits": acgi_out.get("tool_logits"),
            "prm_scores":  prm_final,
            "think_mask":  think_mask,
            "tool_mask":   tool_mask,
            "hidden":      h_norm,
            "alpha_list":  alpha_list,
        }

    def get_inference_output(self, input_ids: torch.Tensor, tau: float = 0.5) -> Dict:
        out  = self.forward(input_ids)
        U    = out["uncertainty"]
        mask = (U > tau).float().unsqueeze(-1)
        out["hybrid_logits"] = (1.0 - mask) * out["ar_logits"] + mask * out["diff_logits"]
        return out

    def count_params_detailed(self) -> Dict[str, int]:
        def _n(m):
            return sum(p.numel() for p in m.parameters())
        counts = {
            "embed":     self.embed.weight.numel(),
            "blocks":    _n(self.blocks),
            "ect":       _n(self.ect),
            "acgi":      _n(self.acgi),
            "diff_head": _n(self.diff_head),
            "mtp":       _n(self.mtp) if self.mtp else 0,
            "tdm":       _n(self.tdm) if self.tdm else 0,
            "sam":       _n(self.sam) if self.sam else 0,
            "dsa":       _n(self.dsa) if self.dsa else 0,
            "lm_head":   0,
        }
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
