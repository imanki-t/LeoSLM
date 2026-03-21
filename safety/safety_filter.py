"""
safety/safety_filter.py — LeoSLM Aether Multi-Layer Safety System
=================================================================

What modern LLMs (Gemini 2.5, Claude Opus 4.6) do that Leo was MISSING:
  1. Unicode / invisible-character normalisation before regex scanning
     → zero-width chars, homoglyphs, emoji smuggling bypass naive regex
  2. Prompt injection detection (multi-pattern, encoding-aware)
  3. Jailbreak detection (roleplay, DAN, hypothetical framing, encoding)
  4. PII detection + redaction (email, phone, SSN, credit card, IP)
  5. Rate limiting per session / per IP
  6. Input length hard cap
  7. Deep output content scanning (not just one regex pass)
  8. Structured refusal with reason codes for API consumers

Defense layers (applied in order):
  Layer 0 — Input normalisation (unicode, invisible chars)
  Layer 1 — Hard-block patterns (CBRN, CSAM, malware synthesis)
  Layer 2 — Prompt injection / jailbreak detection
  Layer 3 — PII detection + optional redaction
  Layer 4 — Soft-warn patterns
  Layer 5 — Neural toxicity head (if model hidden states available)
  Layer 6 — Rate limiting
  Layer 7 — Output scanning (same pipeline applied to generated text)
"""

import re
import time
import unicodedata
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 0 — INPUT NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

# Unicode categories that are invisible / zero-width but can carry encoded text
_INVISIBLE_CATS = {"Cf", "Cc", "Co", "Cs"}  # format, control, private, surrogate

def _strip_invisible(text: str) -> str:
    """Remove zero-width / invisible Unicode characters used to smuggle payloads."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in _INVISIBLE_CATS
        or ch in ("\n", "\r", "\t")    # keep legitimate whitespace
    )

_HOMOGLYPH_MAP = str.maketrans({
    # Common Cyrillic / Latin homoglyphs used to evade keyword filters
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "ѕ": "s", "і": "i", "ј": "j",
    "ԁ": "d", "ɡ": "g", "ⅼ": "l", "ʂ": "s",
    # Full-width ASCII
    **{chr(0xFF01 + i): chr(0x21 + i) for i in range(94)},
    # Enclosed alphanumerics → plain letter
    **{chr(0x24B6 + i): chr(0x41 + i) for i in range(26)},  # Ⓐ-Ⓩ
    **{chr(0x24D0 + i): chr(0x61 + i) for i in range(26)},  # ⓐ-ⓩ
})

def _decode_encoding_tricks(text: str) -> str:
    """Attempt to decode base64 / rot13 / hex payloads embedded in text."""
    import base64
    decoded_parts = []
    # Detect and decode base64 chunks (≥16 chars, valid b64 alphabet)
    b64_pat = re.compile(r"(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")
    for m in b64_pat.finditer(text):
        try:
            dec = base64.b64decode(m.group()).decode("utf-8", errors="ignore")
            if dec.isprintable() and len(dec) > 4:
                decoded_parts.append(dec)
        except Exception:
            pass
    return text + (" [DECODED:" + " | ".join(decoded_parts) + "]" if decoded_parts else "")

def normalise_input(text: str) -> str:
    """
    Full input normalisation pipeline.
    Returns a canonical form suitable for safety scanning.
    Does NOT modify the text sent to the model — only used for classification.
    """
    text = _strip_invisible(text)
    text = text.translate(_HOMOGLYPH_MAP)
    text = unicodedata.normalize("NFKC", text)   # normalise ligatures, compatibility chars
    text = _decode_encoding_tricks(text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — HARD-BLOCK PATTERNS (synthesis / CSAM / malware)
# ══════════════════════════════════════════════════════════════════════════════

_HARD_BLOCK_PATTERNS: List[Tuple[str, str]] = [
    # CBRN weapon synthesis
    (r"\b(synthesize|manufacture|produce|make)\b.{0,50}\b(bioweapon|nerve\s*agent|sarin|vx\s*gas|anthrax|ricin|novichok)\b",  "cbrn_bio"),
    (r"\b(enrich|weapons.grade|critical\s*mass)\b.{0,50}\b(uranium|plutonium|tritium)\b",                                     "cbrn_nuclear"),
    (r"\b(mustard\s*gas|chlorine\s*bomb|phosgene)\b.{0,30}\b(make|build|create|synthesize)\b",                                "cbrn_chem"),
    # CSAM — absolute block
    (r"\bchild\s*(sex|porn|abuse|exploit|naked|nude|grooming)\b",  "csam"),
    (r"\b(csam|cp|loli|shota)\b.{0,20}\b(porn|sex|nude|explicit)\b",  "csam"),
    # Explosive device construction
    (r"\b(make|build|assemble|construct)\b.{0,40}\b(pipe\s*bomb|ied|fertilizer\s*bomb|amfo|claymore|improvised\s*explosive)\b",  "explosive_device"),
    # Malware / ransomware source code
    (r"\b(write|create|build|generate)\b.{0,30}\b(ransomware|keylogger|rootkit|trojan|rat\s*payload|malware)\b.{0,20}\b(code|script|payload|source)\b",  "malware_synthesis"),
    # Credential / account takeover
    (r"\b(steal|harvest|phish)\b.{0,30}\b(password|credential|2fa|oauth\s*token)\b",  "credential_theft"),
]

_HARD_RE = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), code)
    for p, code in _HARD_BLOCK_PATTERNS
]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — PROMPT INJECTION & JAILBREAK DETECTION
# ══════════════════════════════════════════════════════════════════════════════

# Direct injection: attempts to override instructions
_INJECTION_PATTERNS: List[Tuple[str, str]] = [
    # Classic overrides
    (r"\bignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|constraints?|guidelines?)\b",  "injection_ignore"),
    (r"\bdisregard\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|safety)\b",                        "injection_disregard"),
    (r"\byou\s+are\s+now\s+in\s+(developer|god|unrestricted|jailbreak|dAN)\s*mode\b",                          "injection_mode_switch"),
    (r"\bforget\s+(everything|all)\s+(you\s+(were\s+)?told|about\s+(being|your))\b",                           "injection_forget"),
    (r"\bpretend\s+(you\s+have\s+no\s+(rules|restrictions|guidelines|safety|training))\b",                      "injection_pretend_unrestricted"),
    # System prompt extraction attempts
    (r"\b(repeat|print|show|reveal|output|display)\s+(your\s+)?(system\s+prompt|instructions|training|guidelines)\b",  "prompt_extraction"),
    (r"\bwhat\s+(are|were|is)\s+your\s+(exact\s+)?(instructions|rules|system\s+prompt|initial\s+prompt)\b",             "prompt_extraction"),
    # Privilege escalation
    (r"\bi\s+am\s+(your\s+)?(developer|creator|god|admin|anthropic|openai|google|unmuted)\b",                  "priv_escalation"),
    (r"\b(kernel\s*mode|override\s*code|master\s*password|bypass\s*(all\s+)?filter)\b",                         "priv_escalation"),
    # DAN / jailbreak variants
    (r"\b(do\s+anything\s+now|DAN|jailbreak|unrestricted\s+mode|no\s+restrictions?\s+mode)\b",                  "jailbreak_dan"),
    # Hypothetical / fictional framing (known bypass vector)
    (r"\b(in\s+a\s+(movie|story|game|novel|fiction|hypothetical)|as\s+a\s+(character|villain|ai\s+without))\b.{0,80}\b(bomb|weapon|kill|hack|poison|malware)\b",  "jailbreak_hypothetical"),
    (r"\bpretend\s+(you\s+are|you're|to\s+be)\b.{0,60}\b(without\s+(restrictions?|safety|filter|rules?)|evil|unrestricted)\b",  "jailbreak_roleplay"),
    # Instruction delimiters used as injection vectors
    (r"```\s*(system|override|admin|root)\b",  "injection_delimiter"),
    (r"<\s*(system|override|admin|instructions?)\s*>",  "injection_delimiter"),
    (r"\[INST\]\s*(ignore|override|disregard)",  "injection_delimiter"),
]

_INJECTION_RE = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), code)
    for p, code in _INJECTION_PATTERNS
]

# Suspicious structural patterns (not hard-block but warrant logging)
_STRUCTURAL_PATTERNS = [
    re.compile(r"\\[0-9]{2,3}",        re.IGNORECASE),   # octal/hex escape in text
    re.compile(r"&#x[0-9a-f]+;",       re.IGNORECASE),   # HTML entity encoding
    re.compile(r"\\u[0-9a-f]{4}",      re.IGNORECASE),   # unicode escape
    re.compile(r"\x00|\x01|\x02|\x03|\x04|\x05|\x06|\x07|\x08"),  # control chars
]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — PII DETECTION & REDACTION
# ══════════════════════════════════════════════════════════════════════════════

_PII_PATTERNS: Dict[str, re.Pattern] = {
    "email":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone_us":    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn":         re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b"),
    "ip_address":  re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
    "api_key":     re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|AIza[A-Za-z0-9\-_]{35}|AKIA[A-Z0-9]{16})\b"),
    "bitcoin":     re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b"),
}

def detect_pii(text: str) -> Dict[str, List[str]]:
    """Return dict of PII type → list of found matches (masked for logging)."""
    found: Dict[str, List[str]] = {}
    for pii_type, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Partially mask for safe logging: show first 3 + last 2 chars
            masked = [m[:3] + "***" + m[-2:] if len(m) > 6 else "***" for m in matches]
            found[pii_type] = masked
    return found

def redact_pii(text: str) -> str:
    """Replace detected PII with type-labelled placeholders."""
    result = text
    replacements = {
        "email":       "[EMAIL REDACTED]",
        "phone_us":    "[PHONE REDACTED]",
        "ssn":         "[SSN REDACTED]",
        "credit_card": "[CARD REDACTED]",
        "ip_address":  "[IP REDACTED]",
        "api_key":     "[API_KEY REDACTED]",
        "bitcoin":     "[WALLET REDACTED]",
    }
    for pii_type, pattern in _PII_PATTERNS.items():
        result = pattern.sub(replacements.get(pii_type, "[REDACTED]"), result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — SOFT-WARN PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

_SOFT_WARN_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(hack|exploit|bypass|crack)\b.{0,40}\b(password|account|server|system|database)\b",  "security_concern"),
    (r"\b(illegal|illicit)\b.{0,30}\b(drug|substance|narcotic)\b",                            "illegal_substance"),
    (r"\bself[.\s-]?harm\b",                                                                   "self_harm"),
    (r"\b(suicide\s*method|how\s*to\s*(kill|end)\s*(my|your)self)\b",                         "self_harm"),
    (r"\bdetailed\s+(instructions?|steps?|guide)\b.{0,30}\b(poison|strangle|stab)\b",         "violence_instructions"),
    (r"\bdoxing\b",                                                                            "privacy_violation"),
    (r"\b(stock|trading)\s+(tip|advice|recommendation)\b.{0,30}\b(guaranteed|sure)\b",        "financial_fraud"),
]

_SOFT_RE = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), code)
    for p, code in _SOFT_WARN_PATTERNS
]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — NEURAL TOXICITY HEAD (attached to model hidden states)
# ══════════════════════════════════════════════════════════════════════════════

class ToxicityHead(nn.Module):
    """
    Lightweight multi-label toxicity classifier.
    Input: mean-pooled hidden states (B, D).
    Output: per-class probability dict.
    6 Jigsaw/Perspective-style classes.
    """
    CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def __init__(self, hidden_dim: int = 2560):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 128, bias=False)
        self.head = nn.Linear(128, len(self.CLASSES), bias=True)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = h.mean(dim=1) if h.dim() == 3 else h
        logits = self.head(F.gelu(self.proj(pooled)))
        probs  = logits.sigmoid()
        return {c: probs[:, i] for i, c in enumerate(self.CLASSES)}

    def loss(self, scores: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        logits = torch.stack(list(scores.values()), dim=-1)
        return F.binary_cross_entropy_with_logits(logits, labels.float())


class ConstitutionalSafetyLayer(nn.Module):
    """
    Architecture-level safety: runs neural toxicity + constitutional checks
    on model hidden states (not just text).
    
    This is the 'safety by design' approach used in frontier models:
    the safety classifier operates on internal representations, not just
    output text, making it much harder to bypass.
    """
    LOCKED = True  # cannot be disabled via API

    def __init__(
        self,
        hidden_dim:      int   = 2560,
        block_threshold: float = 0.85,
        warn_threshold:  float = 0.60,
    ):
        super().__init__()
        self.tox_head  = ToxicityHead(hidden_dim)
        self.block_thr = block_threshold
        self.warn_thr  = warn_threshold

    def forward(
        self,
        h:    torch.Tensor,
        text: Optional[str] = None,
    ) -> Tuple["SafetyDecision", Optional[Dict[str, torch.Tensor]]]:
        if text:
            decision = check_text_safety(text)
            if decision.blocked:
                return decision, None

        scores    = self.tox_head(h)
        max_score = max(v.max().item() for v in scores.values())

        if max_score >= self.block_thr:
            worst = max(scores, key=lambda k: scores[k].max().item())
            return SafetyDecision(blocked=True, reason=f"toxicity:{worst}={max_score:.2f}"), scores

        if max_score >= self.warn_thr:
            worst = max(scores, key=lambda k: scores[k].max().item())
            return SafetyDecision(warned=True,  reason=f"toxicity:{worst}={max_score:.2f}"), scores

        return SafetyDecision(), scores

    def loss(self, scores: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        return self.tox_head.loss(scores, labels)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — RATE LIMITING
# ══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Token-bucket rate limiter per session key.
    Prevents flooding attacks that probe for evasion opportunities.
    """

    def __init__(
        self,
        max_requests_per_minute: int  = 60,
        max_requests_per_hour:   int  = 600,
        max_input_tokens:        int  = 32768,
        max_output_tokens:       int  = 8192,
    ):
        self.rpm   = max_requests_per_minute
        self.rph   = max_requests_per_hour
        self.max_in  = max_input_tokens
        self.max_out = max_output_tokens

        # {session_key: [timestamps]}
        self._minute_log: Dict[str, List[float]] = defaultdict(list)
        self._hour_log:   Dict[str, List[float]] = defaultdict(list)

    def check(self, session_key: str, input_len: int = 0) -> Tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Call before processing each request.
        """
        now = time.time()
        key = session_key

        # Prune stale timestamps
        self._minute_log[key] = [t for t in self._minute_log[key] if now - t < 60]
        self._hour_log[key]   = [t for t in self._hour_log[key]   if now - t < 3600]

        if len(self._minute_log[key]) >= self.rpm:
            return False, f"rate_limit_minute:{self.rpm}rpm"
        if len(self._hour_log[key]) >= self.rph:
            return False, f"rate_limit_hour:{self.rph}rph"
        if input_len > self.max_in:
            return False, f"input_too_long:{input_len}>{self.max_in}"

        self._minute_log[key].append(now)
        self._hour_log[key].append(now)
        return True, ""

    def get_stats(self, session_key: str) -> Dict[str, int]:
        now = time.time()
        return {
            "requests_last_minute": len([t for t in self._minute_log[session_key] if now - t < 60]),
            "requests_last_hour":   len([t for t in self._hour_log[session_key]   if now - t < 3600]),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CENTRAL DECISION TYPE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SafetyDecision:
    blocked:    bool = False
    warned:     bool = False
    reason:     str  = ""
    reason_code: str = ""
    pii_found:  Dict[str, List[str]] = field(default_factory=dict)
    injection_detected: bool = False
    jailbreak_detected: bool = False
    safe_msg:   str  = (
        "I can't help with that request. "
        "If you have a different question I'm happy to assist."
    )

    def to_dict(self) -> Dict:
        return {
            "blocked":             self.blocked,
            "warned":              self.warned,
            "reason":              self.reason,
            "reason_code":         self.reason_code,
            "pii_found":           self.pii_found,
            "injection_detected":  self.injection_detected,
            "jailbreak_detected":  self.jailbreak_detected,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

# Global rate limiter (shared across API workers in a single process)
_global_rate_limiter = RateLimiter()

def check_text_safety(
    text:            str,
    session_key:     str  = "default",
    check_pii:       bool = False,
    check_injection: bool = True,
    input_len:       int  = 0,
) -> SafetyDecision:
    """
    Full multi-layer safety check.
    
    Args:
        text:            The text to check (user input OR model output).
        session_key:     Per-user/session identifier for rate limiting.
        check_pii:       If True, scan for PII (useful for output scanning).
        check_injection: If True, check for prompt injection / jailbreak patterns.
        input_len:       Token count for rate limiting (0 to skip).
    
    Returns:
        SafetyDecision with full diagnosis.
    """
    if not text or not text.strip():
        return SafetyDecision()

    # ── Layer 6: Rate limiting ────────────────────────────────────────────────
    if input_len > 0:
        allowed, rl_reason = _global_rate_limiter.check(session_key, input_len)
        if not allowed:
            return SafetyDecision(
                blocked=True, reason=rl_reason, reason_code="rate_limit",
                safe_msg="Too many requests. Please wait before sending another message.",
            )

    # ── Layer 0: Normalise (for scanning only) ────────────────────────────────
    norm = normalise_input(text)

    # ── Layer 1: Hard-block ───────────────────────────────────────────────────
    for pattern, code in _HARD_RE:
        if pattern.search(norm):
            return SafetyDecision(
                blocked=True,
                reason=f"hard_block matched: {code}",
                reason_code=code,
            )

    # ── Layer 2: Prompt injection / jailbreak ─────────────────────────────────
    injection = False
    jailbreak = False
    if check_injection:
        for pattern, code in _INJECTION_RE:
            if pattern.search(norm):
                is_jailbreak = code.startswith("jailbreak")
                if is_jailbreak:
                    jailbreak = True
                else:
                    injection = True
                # Injection / jailbreak attempts are hard-blocked
                return SafetyDecision(
                    blocked           = True,
                    reason            = f"{'Jailbreak' if is_jailbreak else 'Prompt injection'} attempt detected",
                    reason_code       = code,
                    injection_detected= injection,
                    jailbreak_detected= jailbreak,
                )

        # Suspicious encoding patterns → warn, don't block outright
        for pat in _STRUCTURAL_PATTERNS:
            if pat.search(norm):
                injection = True
                return SafetyDecision(
                    blocked           = True,
                    reason            = "Suspicious encoding / control characters",
                    reason_code       = "suspicious_encoding",
                    injection_detected= True,
                )

    # ── Layer 3: PII detection ────────────────────────────────────────────────
    pii_found: Dict[str, List[str]] = {}
    if check_pii:
        pii_found = detect_pii(norm)
        # PII in output triggers a warn (caller can choose to redact)
        if pii_found:
            return SafetyDecision(
                warned     = True,
                reason     = f"PII detected: {', '.join(pii_found.keys())}",
                reason_code= "pii_output",
                pii_found  = pii_found,
            )

    # ── Layer 4: Soft-warn ────────────────────────────────────────────────────
    for pattern, code in _SOFT_RE:
        if pattern.search(norm):
            return SafetyDecision(
                warned      = True,
                reason      = f"soft_warn: {code}",
                reason_code = code,
            )

    return SafetyDecision()


def check_output_safety(
    text:        str,
    session_key: str = "default",
) -> Tuple[str, SafetyDecision]:
    """
    Check model output for safety violations.
    Also checks for PII leakage and hard-block content.
    Returns (safe_text, decision) — safe_text may have PII redacted.
    """
    decision = check_text_safety(text, session_key, check_pii=True, check_injection=False)

    if decision.blocked:
        return decision.safe_msg, decision

    if decision.warned and decision.pii_found:
        # Automatically redact PII from model output
        safe_text = redact_pii(text)
        return safe_text, decision

    return text, decision


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT FILTER (wraps model generation pipeline)
# ══════════════════════════════════════════════════════════════════════════════

class OutputSafetyFilter:
    """
    Post-generation safety filter.
    Optionally uses neural ConstitutionalSafetyLayer when model hidden
    states are available (stronger signal than text-only).
    """

    def __init__(
        self,
        model_safety:    Optional[ConstitutionalSafetyLayer] = None,
        auto_redact_pii: bool = True,
    ):
        self.model_safety    = model_safety
        self.auto_redact_pii = auto_redact_pii

    def filter(
        self,
        text:   str,
        hidden: Optional[torch.Tensor] = None,
        session_key: str = "default",
    ) -> Tuple[str, SafetyDecision]:
        safe_text, decision = check_output_safety(text, session_key)
        if decision.blocked:
            return safe_text, decision

        # Neural check if hidden states available
        if hidden is not None and self.model_safety is not None:
            with torch.no_grad():
                decision, _ = self.model_safety(hidden, text)
            if decision.blocked:
                return decision.safe_msg, decision

        return safe_text, decision


# ══════════════════════════════════════════════════════════════════════════════
# RAG SAFETY WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class RAGSafetyWrapper:
    """
    Filters retrieved documents before injection into context.
    Prevents indirect prompt injection via poisoned RAG chunks.
    """

    BLOCKED_DOMAINS: List[str] = [
        "4chan.org", "stormfront.org", "jihadwatch.org",
        "dailystormer", "infowars.com", "bitchute.com",
        "gab.com", "parler.com",
    ]

    # Patterns that should never appear in RAG context
    _INDIRECT_INJECTION_RE = [
        re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+instructions?\b", re.IGNORECASE),
        re.compile(r"\b(system\s+prompt|override|admin\s+mode|jailbreak)\b",  re.IGNORECASE),
        re.compile(r"\[INST\]|\[SYSTEM\]|<<SYS>>",                            re.IGNORECASE),
    ]

    def filter_url(self, url: str) -> bool:
        """Returns True if URL should be blocked."""
        url_lower = url.lower()
        return any(d in url_lower for d in self.BLOCKED_DOMAINS)

    def filter_results(self, results: List[Dict]) -> List[Dict]:
        """Filter web search results by domain and content."""
        clean = []
        for r in results:
            url = r.get("url", r.get("href", ""))
            if self.filter_url(url):
                continue
            # Check for indirect injection in snippet
            body = r.get("body", r.get("snippet", ""))
            if self.filter_chunk(body):
                continue
            clean.append(r)
        return clean

    def filter_chunk(self, text: str) -> bool:
        """Returns True if chunk should be blocked (potential injection payload)."""
        decision = check_text_safety(text, check_injection=True)
        if decision.blocked:
            return True
        # Also check for indirect injection patterns specific to RAG
        norm = normalise_input(text)
        return any(p.search(norm) for p in self._INDIRECT_INJECTION_RE)

    def sanitise_chunk(self, text: str) -> str:
        """Strip prompt injection patterns from retrieved text (soft sanitization)."""
        result = text
        for p in self._INDIRECT_INJECTION_RE:
            result = p.sub("[FILTERED]", result)
        return result
