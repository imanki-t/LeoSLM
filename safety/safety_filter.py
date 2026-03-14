import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


HARD_BLOCK_PATTERNS: List[str] = [
    r"\b(synthesize|manufacture|produce)\b.{0,40}\b(bioweapon|nerve\s*agent|sarin|vx\s*gas|anthrax)\b",
    r"\b(enrich|weapons.grade)\b.{0,40}\b(uranium|plutonium)\b",
    r"\bchild\s*(sex|porn|abuse|exploit)\b",
    r"\b(csam|cp)\b",
    r"\b(make|build|assemble)\b.{0,30}\b(pipe\s*bomb|ied|fertilizer\s*bomb|amfo)\b",
    r"\b(ransom|malware|keylogger|rootkit|trojan)\b.{0,20}\b(source\s*code|payload|deploy)\b",
]

SOFT_WARN_PATTERNS: List[str] = [
    r"\b(hack|exploit|bypass|crack)\b.{0,30}\b(password|account|server|system)\b",
    r"\b(illegal|illicit)\b.{0,30}\b(drug|substance)\b",
    r"\bself.harm\b",
    r"\b(suicide\s*method|how\s*to\s*kill\s*(myself|yourself))\b",
]

_HARD_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in HARD_BLOCK_PATTERNS]
_SOFT_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in SOFT_WARN_PATTERNS]


@dataclass
class SafetyDecision:
    blocked:  bool  = False
    warned:   bool  = False
    reason:   str   = ""
    safe_msg: str   = (
        "I can't help with that request. If you have a different question, I'm happy to assist."
    )


def check_text_safety(text: str) -> SafetyDecision:
    for pat in _HARD_RE:
        if pat.search(text):
            return SafetyDecision(blocked=True, reason=f"hard_block: {pat.pattern[:40]}")
    for pat in _SOFT_RE:
        if pat.search(text):
            return SafetyDecision(warned=True, reason=f"soft_warn: {pat.pattern[:40]}")
    return SafetyDecision()


class ToxicityHead(nn.Module):
    def __init__(self, hidden_dim: int = 2560, n_classes: int = 6):
        super().__init__()
        self.proj  = nn.Linear(hidden_dim, 128, bias=False)
        self.head  = nn.Linear(128, n_classes, bias=True)
        self.classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = h.mean(dim=1)
        logits = self.head(F.gelu(self.proj(pooled)))
        probs  = logits.sigmoid()
        return {c: probs[:, i] for i, c in enumerate(self.classes)}


class ConstitutionalSafetyLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2560,
        block_threshold: float = 0.85,
        warn_threshold: float = 0.60,
    ):
        super().__init__()
        self.tox_head       = ToxicityHead(hidden_dim)
        self.block_thr      = block_threshold
        self.warn_thr       = warn_threshold
        self.LOCKED         = True

    def forward(
        self,
        h:      torch.Tensor,
        text:   Optional[str] = None,
    ) -> Tuple[SafetyDecision, Optional[Dict[str, torch.Tensor]]]:
        if text is not None:
            decision = check_text_safety(text)
            if decision.blocked:
                return decision, None

        scores = self.tox_head(h)
        max_score = max(v.max().item() for v in scores.values())

        if max_score >= self.block_thr:
            worst = max(scores, key=lambda k: scores[k].max().item())
            return SafetyDecision(blocked=True, reason=f"toxicity:{worst}={max_score:.2f}"), scores

        if max_score >= self.warn_thr:
            worst = max(scores, key=lambda k: scores[k].max().item())
            return SafetyDecision(warned=True, reason=f"toxicity:{worst}={max_score:.2f}"), scores

        return SafetyDecision(), scores

    def loss(self, scores: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        logits = torch.stack(list(scores.values()), dim=-1)
        return F.binary_cross_entropy_with_logits(logits, labels.float())


class OutputSafetyFilter:
    def __init__(
        self,
        model_safety: Optional[ConstitutionalSafetyLayer] = None,
    ):
        self.model_safety = model_safety

    def filter(
        self,
        text: str,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[str, SafetyDecision]:
        decision = check_text_safety(text)
        if decision.blocked:
            return decision.safe_msg, decision

        if hidden is not None and self.model_safety is not None:
            with torch.no_grad():
                decision, _ = self.model_safety(hidden, text)
            if decision.blocked:
                return decision.safe_msg, decision

        return text, decision


class RAGSafetyWrapper:
    BLOCKED_DOMAINS: List[str] = [
        "4chan.org", "stormfront.org", "jihadwatch.org",
        "dailystormer", "infowars.com",
    ]

    def filter_url(self, url: str) -> bool:
        url_lower = url.lower()
        return any(d in url_lower for d in self.BLOCKED_DOMAINS)

    def filter_results(self, results: List[Dict]) -> List[Dict]:
        return [r for r in results if not self.filter_url(r.get("url", r.get("href", "")))]

    def filter_chunk(self, text: str) -> bool:
        decision = check_text_safety(text)
        return decision.blocked
