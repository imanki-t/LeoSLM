"""
model/__init__.py — LeoSLM model package public API
=====================================================
Import the model like:
    from model import LeoSLM, LeoConfig, CFG
    from model import LEO_IDENTITY, LEO_SYSTEM_PROMPT
"""

from .config    import LeoConfig, CFG
from .identity  import LEO_IDENTITY, LEO_SYSTEM_PROMPT
from .norm      import RMSNorm
from .rope      import build_yarn_rope_cache, apply_rope
from .ect       import ECTv3Module
from .memory    import TemporalDiffusionMemory, StructuredAgenticMemory
from .attention import EpistemicPositionalEncoding, MultiHeadLatentAttention, DSALite
from .moe       import ExpertFFN, UWMRMoE, swiglu
from .mtp       import MultiTokenPredictionHead
from .agentic   import AgenticConfidenceGatedInvocation
from .leo_block import LeoBlock
from .leo_slm   import LeoSLM, HardThresholdGate

__all__ = [
    "LeoConfig", "CFG",
    "LEO_IDENTITY", "LEO_SYSTEM_PROMPT",
    "RMSNorm",
    "build_yarn_rope_cache", "apply_rope",
    "ECTv3Module",
    "TemporalDiffusionMemory", "StructuredAgenticMemory",
    "EpistemicPositionalEncoding", "MultiHeadLatentAttention", "DSALite",
    "ExpertFFN", "UWMRMoE", "swiglu",
    "MultiTokenPredictionHead",
    "AgenticConfidenceGatedInvocation",
    "LeoBlock",
    "LeoSLM", "HardThresholdGate",
]
