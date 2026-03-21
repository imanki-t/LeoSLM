"""
api.py — LeoSLM Aether Flask inference API

BUG FIXES vs original:
  1. Top-level `from leo_rag import LeoKnowledgeLayer` and
     `from safety import OutputSafetyFilter` crashed on import if those
     optional modules were missing. Wrapped in try/except with stub fallbacks.
  2. `_get_engine` used `torch_xla.device()` which requires explicit
     `import torch_xla` (not just `import torch_xla.core.xla_model as xm`).
     Fixed to use `xm.xla_device()`.
  3. Added `check_text_safety` stub so the generate route works even without
     the safety module.
"""

import json
import os
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# ── Optional Flask ─────────────────────────────────────────────────────────────
try:
    from flask import Flask, Response, jsonify, request, stream_with_context
    _FLASK = True
except ImportError:
    _FLASK = False

# ── XLA ────────────────────────────────────────────────────────────────────────
try:
    import torch_xla                           # explicit import!
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    class _XM:
        def xla_device(self): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xm = _XM()

from model import LeoSLM, LeoConfig, CFG

# BUG FIX: wrap optional module imports in try/except so the API server
# can start even if leo_rag / safety are not installed.
try:
    from leo_rag import LeoKnowledgeLayer
    RAG_AVAILABLE = True
except ImportError:
    LeoKnowledgeLayer = None
    RAG_AVAILABLE     = False

try:
    from safety import OutputSafetyFilter, check_text_safety

    class _SafetyDecisionStub:
        blocked = False
        warned  = False
        reason  = ""
        safe_msg = "I can't help with that."

    _HAS_SAFETY = True
except ImportError:
    _HAS_SAFETY = False
    OutputSafetyFilter = None

    class _SafetyDecisionStub:
        blocked  = False
        warned   = False
        reason   = ""
        safe_msg = "I can't help with that."

    def check_text_safety(text: str) -> "_SafetyDecisionStub":
        return _SafetyDecisionStub()


PERSONALITY_PATH = "./personality.json"

_LOCKED_KEYS = frozenset({
    "num_layers", "hidden_dim", "num_heads", "moe_experts", "moe_top_k",
    "vocab_size", "mla_c_kv", "mla_c_q", "num_ect", "ect_heads",
    "acgi_threshold", "tdm_memory_size", "cmg_threshold",
    "use_mla", "use_mtp", "use_tdm", "use_sam", "use_ect_spawn",
    "chunk_size", "max_seq_len",
})


@dataclass
class PersonalityConfig:
    name:             str   = "Leo"
    full_name:        str   = "Leo Aether"
    creator:          str   = "Unmuted"
    tone:             str   = "Direct, honest, and curious."
    response_style:   str   = "thoughtful"
    system_prefix:    str   = ""
    max_think_budget: int   = 8192
    default_mode:     str   = "auto"
    temperature:      float = 0.7
    top_p:            float = 0.9

    def save(self, path: str = PERSONALITY_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = PERSONALITY_PATH) -> "PersonalityConfig":
        if not Path(path).exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_system_prompt(self) -> str:
        lines = [
            f"You are {self.full_name}, built by {self.creator}.",
            f"Tone: {self.tone}",
        ]
        if self.system_prefix:
            lines.append(self.system_prefix)
        return "<|system|>\n" + "\n".join(lines) + "\n<|/system|>\n"

    def update(self, updates: Dict[str, Any]) -> List[str]:
        rejected = []
        for k, v in updates.items():
            if k in _LOCKED_KEYS:
                rejected.append(k)
                continue
            if k in self.__dataclass_fields__:
                setattr(self, k, v)
        return rejected


def _load_tokenizer(path: str = "./leo_tokenizer"):
    try:
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast.from_pretrained(path)
    except Exception:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.add_special_tokens({
            "pad_token": "[PAD]",
            "additional_special_tokens": [
                "[MASK]", "[IDK]", "<think>", "</think>",
                "<|tool_call|>", "<|/tool_call|>",
                "<|tool_result|>", "<|/tool_result|>",
                "<|system|>", "<|/system|>",
            ],
        })
        return tok


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    probs  = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum  = sorted_p.cumsum(-1)
        mask = (cum - sorted_p) > top_p
        sorted_p[mask] = 0.0
        sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True).clamp(1e-10)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_i, sorted_p)
    return torch.multinomial(probs, 1)


class _NoOpSafetyFilter:
    """Stub when safety module is not available."""
    def filter(self, text, hidden=None):
        return text, _SafetyDecisionStub()


class LeoGenerationEngine:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        personality: PersonalityConfig,
        knowledge=None,
        safety=None,
    ):
        self.model       = model.eval()
        self.tok         = tokenizer
        self.device      = device
        self.personality = personality
        self.knowledge   = knowledge
        self.safety      = safety if safety is not None else _NoOpSafetyFilter()
        self.cfg         = model.cfg

    @torch.no_grad()
    def stream_tokens(
        self,
        prompt:      str,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        top_p:       float = 0.9,
        mode:        str   = "auto",
        rag_query:   Optional[str] = None,
    ) -> Iterator[Dict]:
        safety_in = check_text_safety(prompt)
        if safety_in.blocked:
            yield {"token": safety_in.safe_msg, "done": True, "blocked": True}
            return

        sys_prompt = self.personality.to_system_prompt()

        if self.knowledge:
            rag_ctx = self.knowledge.augment(rag_query or prompt)
        else:
            rag_ctx = ""

        if mode == "think":
            sys_prompt += "<think>\n"

        full_input = sys_prompt + rag_ctx + prompt
        ids = self.tok.encode(full_input, return_tensors="pt").to(self.device)

        stop_ids         = {self.cfg.eos_id, self.cfg.pad_id}
        generated_tokens: List[int] = []
        tool_buffer      = ""
        in_tool_call     = False

        for step in range(max_tokens):
            out    = self.model(ids)
            logits = out["ar_logits"][:, -1, :]
            U      = out["uncertainty"][:, -1].item()

            if mode == "auto":
                use_diff = U > 0.5
                if use_diff:
                    logits = (1 - U) * logits + U * out["diff_logits"][:, -1, :]

            tok_id  = _sample(logits, temperature, top_p)
            tok_int = tok_id.item()

            if tok_int in stop_ids:
                yield {"token": "", "done": True, "uncertainty": U}
                break

            generated_tokens.append(tok_int)
            ids = torch.cat([ids, tok_id.unsqueeze(0)], dim=-1)

            decoded = self.tok.decode([tok_int], skip_special_tokens=False)

            if self.cfg.tool_call_start == tok_int:
                in_tool_call = True
                tool_buffer  = ""
                continue

            if in_tool_call:
                if self.cfg.tool_call_end == tok_int:
                    in_tool_call = False
                    if self.knowledge:
                        result_text = self.knowledge.tools.parse_and_call(tool_buffer)
                    else:
                        result_text = f"[Tool not available: {tool_buffer}]"
                    result_ids = self.tok.encode(result_text, return_tensors="pt").to(self.device)
                    ids        = torch.cat([ids, result_ids], dim=-1)
                    yield {"tool_call": tool_buffer, "tool_result": result_text, "done": False}
                    tool_buffer = ""
                else:
                    tool_buffer += decoded
                continue

            yield {"token": decoded, "done": False, "uncertainty": U, "step": step}

        text = self.tok.decode(generated_tokens, skip_special_tokens=True)
        safe_text, safety_out = self.safety.filter(text)
        if safety_out.blocked:
            yield {"token": "", "safe_text": safe_text, "done": True, "blocked": True}

    def generate(
        self,
        prompt:      str,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        top_p:       float = 0.9,
        mode:        str   = "auto",
    ) -> str:
        tokens: List[str] = []
        for chunk in self.stream_tokens(prompt, max_tokens, temperature, top_p, mode):
            if "token" in chunk and chunk["token"]:
                tokens.append(chunk["token"])
        return "".join(tokens)


_STATE: Dict[str, Any] = {"engine": None}


def _get_engine(checkpoint: Optional[str] = None) -> LeoGenerationEngine:
    if _STATE["engine"] is not None:
        return _STATE["engine"]

    # BUG FIX: use xm.xla_device() — safe in all torch_xla versions.
    if XLA_AVAILABLE:
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = LeoSLM(CFG)
    if XLA_AVAILABLE:
        model = model.to(torch.bfloat16)
    model = model.to(device)

    if checkpoint and Path(checkpoint).exists():
        ckpt  = torch.load(checkpoint, map_location="cpu")
        state = {k.replace("_orig_mod.", "").replace("module.", ""): v
                 for k, v in ckpt.get("model", ckpt).items()}
        model.load_state_dict(state, strict=False)
        print(f"[API] Loaded checkpoint: {checkpoint}")

    tok         = _load_tokenizer()
    personality = PersonalityConfig.load()

    # BUG FIX: graceful fallback when optional modules are absent.
    knowledge = None
    if RAG_AVAILABLE and LeoKnowledgeLayer is not None:
        try:
            knowledge = LeoKnowledgeLayer.from_config("./config/leo_config.yaml")
        except Exception as e:
            print(f"[API] Knowledge layer unavailable: {e}")

    safety = None
    if _HAS_SAFETY and OutputSafetyFilter is not None:
        try:
            safety = OutputSafetyFilter()
        except Exception as e:
            print(f"[API] Safety filter unavailable: {e}")

    _STATE["engine"] = LeoGenerationEngine(
        model, tok, device, personality, knowledge, safety
    )
    return _STATE["engine"]


def create_app(checkpoint: Optional[str] = None) -> "Flask":
    if not _FLASK:
        raise ImportError("pip install flask flask-cors")

    app   = Flask("LeoAPI")
    _ckpt = [checkpoint]

    @app.route("/health")
    def health():
        total = sum(p.numel() for p in _get_engine(_ckpt[0]).model.parameters())
        return jsonify({"status": "ok", "model": "LeoSLM-Aether",
                        "params": f"{total/1e9:.2f}B"})

    @app.route("/personality", methods=["GET"])
    def get_personality():
        return jsonify(asdict(_get_engine(_ckpt[0]).personality))

    @app.route("/personality", methods=["POST"])
    def set_personality():
        eng      = _get_engine(_ckpt[0])
        updates  = request.get_json(force=True)
        rejected = eng.personality.update(updates)
        eng.personality.save()
        resp = {"status": "ok", "personality": asdict(eng.personality)}
        if rejected:
            resp["locked_keys_ignored"] = rejected
        return jsonify(resp)

    @app.route("/generate", methods=["POST"])
    def generate():
        eng  = _get_engine(_ckpt[0])
        body = request.get_json(force=True)
        prompt      = body.get("prompt", "")
        max_tokens  = int(body.get("max_tokens", 512))
        temperature = float(body.get("temperature", eng.personality.temperature))
        top_p       = float(body.get("top_p", eng.personality.top_p))
        mode        = body.get("mode", eng.personality.default_mode)
        do_stream   = bool(body.get("stream", False))

        if do_stream:
            def event_stream():
                for chunk in eng.stream_tokens(prompt, max_tokens, temperature, top_p, mode):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

        text = eng.generate(prompt, max_tokens, temperature, top_p, mode)
        return jsonify({"text": text, "model": "LeoSLM-Aether"})

    @app.route("/tools", methods=["GET"])
    def list_tools():
        eng = _get_engine(_ckpt[0])
        if eng.knowledge:
            return jsonify(eng.knowledge.tools.list_tools())
        return jsonify([])

    @app.route("/safety/check", methods=["POST"])
    def safety_check():
        body     = request.get_json(force=True)
        text     = body.get("text", "")
        decision = check_text_safety(text)
        return jsonify({"blocked": decision.blocked, "warned": decision.warned,
                        "reason": decision.reason})

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LeoSLM API Server")
    parser.add_argument("--checkpoint", default="./checkpoints/latest.pt")
    parser.add_argument("--port",       type=int, default=8080)
    parser.add_argument("--host",       default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(args.checkpoint)
    print(f"[API] Leo Aether API starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
