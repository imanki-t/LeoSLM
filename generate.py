"""
LeoSLM "Aether" — generate.py
================================
Inference script for Leo Aether. Built by Unmuted.

Generation modes:
    ar        — Pure autoregressive (fast)
    diffusion — Pure masked diffusion (parallel, iterative)
    hybrid    — AR + ECT-gated selective diffusion (best quality, default)
    think     — Forces <think>...</think> reasoning before answer
    nothink   — Forces fast mode, skips think block
    agentic   — Full tool-calling loop via LeoKnowledgeLayer (ACGI + TTIP)
    auto      — ECT decides mode automatically (think if uncertain, ar if confident)

Aether-specific features:
    MTP speculative decoding  — N draft tokens per step, no separate draft model
    YaRN context extension    — 32k trained → 128k at inference
    ACGI tool gating          — architecture-level tool invocation
    TTIP                      — tool calls allowed mid-think
    ECT uncertainty display   — per-token confidence visualisation
    Leo identity              — Leo knows who he is

Usage:
    python3 generate.py --prompt "What is 2+2?"
    python3 generate.py --prompt "Explain transformers" --mode think
    python3 generate.py --prompt "Search for today's news" --mode agentic
    python3 generate.py --chat
    python3 generate.py --prompt "Hi" --mode ar --max_tokens 200 --checkpoint ./checkpoints/latest.pt
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# ── XLA environment — must precede torch_xla import ──────────────────────────
os.environ.setdefault("PJRT_DEVICE", "TPU")

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    class _XM:
        def xla_device(self):       return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def mark_step(self):        pass
        def master_print(self, *a): print(*a)
    xm = _XM()
    XLA_AVAILABLE = False

# ── Import model from the train.py monolith ───────────────────────────────────
# All of LeoSLM, LeoConfig, LEO_IDENTITY, LEO_SYSTEM_PROMPT live in train.py.
# generate.py does NOT import from model/, diffusion/, or training/ folders.
sys.path.insert(0, str(Path(__file__).parent))
from train import LeoSLM, LeoConfig, LEO_IDENTITY, LEO_SYSTEM_PROMPT, CFG  # noqa: E402

# ── Optional: RAG + knowledge layer ──────────────────────────────────────────
try:
    from leo_rag import LeoKnowledgeLayer
    RAG_AVAILABLE = True
except ImportError:
    LeoKnowledgeLayer = None
    RAG_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

def load_tokenizer(path: str = "./leo_tokenizer"):
    """Load LeoTokenizer. Falls back to GPT-2 + Aether special tokens."""
    from transformers import PreTrainedTokenizerFast, AutoTokenizer

    if Path(f"{path}/tokenizer.json").exists():
        tok = PreTrainedTokenizerFast.from_pretrained(path)
        print(f"  Tokenizer : LeoTokenizer (vocab={tok.vocab_size:,})")
        return tok

    print("  Tokenizer : LeoTokenizer not found — using GPT-2 + Aether tokens")
    print("              Run prep_data.py first to train the custom tokenizer.")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({
        "pad_token": "[PAD]",
        "additional_special_tokens": [
            "[MASK]", "[IDK]", "<think>", "</think>",
            "<|tool_call|>", "<|/tool_call|>",
            "<|tool_result|>", "<|/tool_result|>",
            "<|system|>",
        ],
    })
    return tok


# ══════════════════════════════════════════════════════════════════════════════
# YARN SCALING
# ══════════════════════════════════════════════════════════════════════════════

def set_yarn_scale(model: LeoSLM, scale: float):
    """Set YaRN scale on all attention blocks for inference context extension."""
    for block in model.blocks:
        if hasattr(block, "attn") and hasattr(block.attn, "cfg"):
            block.attn.cfg.yarn_scale = scale
    model.cfg.yarn_scale = scale


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_token(logits: torch.Tensor, temperature: float,
                 top_k: int, top_p: float) -> torch.Tensor:
    """Sample one token: temperature + top-k + nucleus. Returns (1,) tensor."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        kth = logits.topk(min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_l, sorted_idx = logits.sort(descending=True)
        cum = sorted_l.softmax(-1).cumsum(-1)
        remove = cum - sorted_l.softmax(-1) > top_p
        sorted_l[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_l)

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


# ══════════════════════════════════════════════════════════════════════════════
# LEO GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class LeoGenerator:
    """
    Full inference engine for LeoSLM Aether.
    Supports all generation modes, MTP speculative decoding,
    YaRN context extension, ACGI tool gating, and identity-aware chat.
    """

    def __init__(
        self,
        checkpoint      : Optional[str]  = None,
        config          : Optional[LeoConfig] = None,
        device          : Optional[str]  = None,
        tok_path        : str  = "./leo_tokenizer",
        yarn_scale      : float = 1.0,
        knowledge_layer = None,
    ):
        # Device
        if device:
            self.device = torch.device(device)
        elif XLA_AVAILABLE:
            self.device = xm.xla_device()
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.cfg   = config or LeoConfig()
        self.model = LeoSLM(self.cfg)

        if checkpoint and Path(checkpoint).exists():
            ckpt  = torch.load(checkpoint, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            # Strip FSDP / torch.compile prefixes
            state = {k.replace("_orig_mod.", "").replace("module.", ""): v
                     for k, v in state.items()}
            missing, _ = self.model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Warning: {len(missing)} missing keys")
            print(f"  Checkpoint: {checkpoint}")
        else:
            print(f"  Checkpoint: {'not found' if checkpoint else 'none'} — random weights")

        if self.device.type in ("xla", "cuda"):
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device)
        self.model.eval()

        if yarn_scale > 1.0:
            set_yarn_scale(self.model, yarn_scale)
            ctx_k = int(self.cfg.max_seq_len * yarn_scale / 1024)
            print(f"  YaRN      : {yarn_scale}× → {ctx_k}k context")

        self.tok       = load_tokenizer(tok_path)
        self.knowledge = knowledge_layer

        # Cache special token IDs
        self._eos         = self.cfg.eos_id
        self._think_start = self.cfg.think_start_id
        self._think_end   = self.cfg.think_end_id
        self._idk         = self.cfg.idk_id

        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Model     : LeoSLM Aether | {total/1e9:.2f}B params | {self.device}")

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _ar_generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                     temperature: float, top_k: int, top_p: float,
                     stop_at: Optional[List[int]] = None,
                     use_mtp: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        AR generation with MTP speculative decoding.

        MTP drafts N future tokens in one forward pass then verifies them.
        On full acceptance this gives ~N× throughput with no quality loss.
        On partial/no acceptance it falls back to the standard single step.
        """
        generated = input_ids.clone()
        stops     = set(stop_at or [self._eos])
        all_u: List[torch.Tensor] = []
        steps = 0

        while steps < max_new_tokens:
            out    = self.model(generated)
            logits = out["ar_logits"]
            U      = out["uncertainty"]

            # ── MTP speculative decoding ──────────────────────────────────────
            if (use_mtp and self.cfg.use_mtp
                    and steps < max_new_tokens - self.cfg.mtp_n
                    and hasattr(self.model, "mtp")):
                h_norm  = self.model.final_norm(out["hidden"])
                drafts  = self.model.mtp.speculative_draft(h_norm, temperature)  # (1, N)
                d_seq   = torch.cat([generated, drafts], dim=1)
                v_out   = self.model(d_seq)
                v_logits = v_out["ar_logits"][:, generated.shape[1]-1:-1, :]

                n_accept = 0
                for i in range(drafts.shape[1]):
                    greedy = v_logits[:, i, :].argmax(-1)
                    if greedy.item() == drafts[0, i].item():
                        n_accept += 1
                        all_u.append(v_out["uncertainty"][:, generated.shape[1]+i-1])
                        if greedy.item() in stops:
                            generated = torch.cat([generated, drafts[:, :i+1]], dim=1)
                            goto_done = True
                            break
                    else:
                        break
                else:
                    goto_done = False

                if n_accept > 0:
                    if goto_done:
                        break
                    generated = torch.cat([generated, drafts[:, :n_accept]], dim=1)
                    steps    += n_accept
                    if XLA_AVAILABLE: xm.mark_step()
                    continue

            # ── Standard single-token step ────────────────────────────────────
            next_id = sample_token(logits[:, -1, :], temperature, top_k, top_p)
            all_u.append(U[:, -1])
            generated = torch.cat([generated, next_id], dim=1)
            steps    += 1
            if next_id.item() in stops:
                break
            if XLA_AVAILABLE:
                xm.mark_step()

        U_seq = torch.cat(all_u, dim=0) if all_u else torch.zeros(1, device=generated.device)
        return generated, U_seq

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _diffusion_refine(self, ids: torch.Tensor, n_steps: int = 8) -> torch.Tensor:
        """
        Selective diffusion refinement: only uncertain tokens get re-masked.
        Confident tokens are never touched — preserves good AR output.
        """
        tau     = self.cfg.uncertainty_thresh
        refined = ids.clone()
        for _ in range(n_steps):
            out  = self.model(refined)
            mask = out["uncertainty"] > tau
            if not mask.any():
                break
            masked = refined.clone()
            masked[mask] = self.cfg.mask_id
            d_out   = self.model(masked)
            new_ids = d_out["diff_logits"].argmax(-1)
            refined[mask] = new_ids[mask]
        return refined

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, prompt: str, mode: str = "hybrid",
                 max_new_tokens: int = 512, temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.9,
                 think_budget: Optional[int] = None,
                 use_mtp: bool = True, verbose: bool = True) -> Dict:
        """
        Main generate entry point. Handles all modes.
        Returns: text, output_ids, uncertainty, think_text, answer_text, mode.
        """
        # Augment with knowledge layer in agentic mode
        aug_prompt = prompt
        if self.knowledge and mode == "agentic":
            aug_prompt = self.knowledge.augment(prompt)

        # Prepend Leo's identity system prompt
        full = LEO_SYSTEM_PROMPT + "\n" + aug_prompt

        enc = self.tok(full, return_tensors="pt",
                       truncation=True, max_length=self.cfg.max_seq_len // 2)
        ids = enc["input_ids"].to(self.device)

        if mode == "ar" or mode == "nothink":
            return self._run_ar(ids, max_new_tokens, temperature, top_k, top_p,
                                use_mtp, verbose)

        elif mode in ("hybrid", "diffusion"):
            return self._run_hybrid(ids, max_new_tokens, temperature, top_k, top_p,
                                    use_mtp, mode, verbose)

        elif mode == "think":
            return self._run_think(ids, max_new_tokens, temperature, top_k, top_p,
                                   think_budget, use_mtp, verbose)

        elif mode == "agentic":
            return self._run_agentic(ids, aug_prompt, max_new_tokens, temperature,
                                     top_k, top_p, use_mtp, verbose)

        elif mode == "auto":
            with torch.no_grad():
                out = self.model(ids)
            budget = self.model.get_think_budget(out["uncertainty"])
            if budget > 0:
                return self._run_think(ids, max_new_tokens, temperature, top_k,
                                       top_p, budget, use_mtp, verbose)
            return self._run_ar(ids, max_new_tokens, temperature, top_k, top_p,
                                use_mtp, verbose)

        else:
            raise ValueError(f"Unknown mode {mode!r}. "
                             "Options: ar, hybrid, diffusion, think, nothink, agentic, auto")

    # ─────────────────────────────────────────────────────────────────────────
    def _run_ar(self, ids, max_new, temp, top_k, top_p, use_mtp, verbose):
        mtp = f" + MTP(N={self.cfg.mtp_n})" if use_mtp and self.cfg.use_mtp else ""
        if verbose: print(f"  [AR{mtp}] max_tokens={max_new}")
        out_ids, U = self._ar_generate(ids, max_new, temp, top_k, top_p, use_mtp=use_mtp)
        text = self.tok.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True)
        return dict(text=text, output_ids=out_ids, uncertainty=U,
                    think_text=None, answer_text=text, tool_calls=[], mode="ar")

    def _run_hybrid(self, ids, max_new, temp, top_k, top_p, use_mtp, mode, verbose):
        if verbose: print("  [Hybrid] AR → ECT scan → selective diffusion refinement")
        out_ids, _ = self._ar_generate(ids, max_new, temp, top_k, top_p, use_mtp=use_mtp)
        refined    = self._diffusion_refine(out_ids) if mode == "hybrid" else out_ids
        with torch.no_grad():
            U_final = self.model(refined)["uncertainty"][0, ids.shape[1]:]
        new_ids = refined[:, ids.shape[1]:]
        if verbose:
            n = (U_final > self.cfg.uncertainty_thresh).sum().item()
            print(f"  Flagged: {n}/{new_ids.shape[1]} | mean U={U_final.mean().item():.3f}")
        text = self.tok.decode(new_ids[0], skip_special_tokens=True)
        return dict(text=text, output_ids=refined, uncertainty=U_final,
                    think_text=None, answer_text=text, tool_calls=[], mode=mode)

    def _run_think(self, ids, max_new, temp, top_k, top_p,
                   think_budget, use_mtp, verbose):
        """
        Think mode: <think>...</think> block then answer.
        ECT auto-determines budget depth if not specified.
        Budget forcing keeps generating inside <think> until budget exhausted.
        """
        if think_budget is None:
            with torch.no_grad():
                out = self.model(ids)
            think_budget = max(self.model.get_think_budget(out["uncertainty"]), 256)

        if verbose: print(f"  [Think] budget={think_budget} tokens")

        # Force open <think>
        think_tok = torch.tensor([[self._think_start]], dtype=torch.long, device=self.device)
        generated = torch.cat([ids, think_tok], dim=1)
        all_u     = []
        steps     = 0

        # Generate inside <think>
        while steps < think_budget:
            with torch.no_grad():
                out = self.model(generated)
            next_id = sample_token(out["ar_logits"][:, -1, :], temp, top_k, top_p)
            all_u.append(out["uncertainty"][:, -1])
            generated = torch.cat([generated, next_id], dim=1)
            steps    += 1
            if next_id.item() in (self._think_end, self._eos):
                break
            if XLA_AVAILABLE: xm.mark_step()

        # Force close </think> if model didn't
        if generated[0, -1].item() != self._think_end:
            end = torch.tensor([[self._think_end]], dtype=torch.long, device=self.device)
            generated = torch.cat([generated, end], dim=1)

        answer_start = generated.shape[1]

        # Generate answer after </think>
        out_ids, U_ans = self._ar_generate(generated, max_new, temp, top_k, top_p,
                                            use_mtp=use_mtp)

        think_text  = self.tok.decode(out_ids[0, ids.shape[1]+1:answer_start],
                                       skip_special_tokens=True)
        answer_text = self.tok.decode(out_ids[0, answer_start:],
                                       skip_special_tokens=True)

        if verbose:
            print(f"  Think: {answer_start - ids.shape[1]} tokens | "
                  f"Answer: {out_ids.shape[1] - answer_start} tokens")

        U_all = torch.stack(all_u, dim=0).squeeze() if all_u else U_ans
        return dict(text=answer_text, output_ids=out_ids, uncertainty=U_all,
                    think_text=think_text, answer_text=answer_text,
                    tool_calls=[], mode="think")

    def _run_agentic(self, ids, raw_prompt, max_new, temp,
                     top_k, top_p, use_mtp, verbose):
        """
        Agentic mode (TTIP): tool calls allowed mid-think.
        LeoKnowledgeLayer handles execution and re-injection of tool results.
        """
        if verbose: print("  [Agentic] TTIP mode — tools allowed mid-think")

        if not RAG_AVAILABLE or self.knowledge is None:
            if verbose: print("  LeoKnowledgeLayer not connected — falling back to think mode")
            return self._run_think(ids, max_new, temp, top_k, top_p,
                                   None, use_mtp, verbose)

        def _gen_fn(prompt_str: str) -> str:
            enc = self.tok(prompt_str, return_tensors="pt",
                          truncation=True, max_length=self.cfg.max_seq_len)
            g_ids = enc["input_ids"].to(self.device)
            out_ids, _ = self._ar_generate(g_ids, max_new, temp, top_k, top_p,
                                           use_mtp=use_mtp)
            return self.tok.decode(out_ids[0, g_ids.shape[1]:], skip_special_tokens=False)

        final = self.knowledge.agentic_loop(raw_prompt, _gen_fn, max_turns=5)
        clean = self.tok.decode(self.tok.encode(final), skip_special_tokens=True)
        U     = torch.zeros(max(len(clean.split()), 1), device=self.device)
        return dict(text=clean, output_ids=None, uncertainty=U,
                    think_text=None, answer_text=clean, tool_calls=[], mode="agentic")

    # ─────────────────────────────────────────────────────────────────────────
    def print_uncertainty_map(self, output_ids: torch.Tensor,
                               uncertainty: torch.Tensor, prompt_len: int = 0):
        """Print per-token uncertainty bars. █ = uncertain, ░ = confident."""
        tau   = self.cfg.uncertainty_thresh
        ids   = output_ids[0, prompt_len:].tolist()
        U_arr = uncertainty.flatten().tolist()
        print("\n── Uncertainty Map ──────────────────────────────────")
        print(f"   τ={tau} | █ uncertain | ░ confident | ⚠ flagged")
        print("─────────────────────────────────────────────────────")
        for tid, u in zip(ids[:len(U_arr)], U_arr):
            try:    tok_str = self.tok.decode([tid])
            except: tok_str = f"[{tid}]"
            bar  = "█" * int(u * 8) + "░" * (8 - int(u * 8))
            flag = "⚠ " if u > tau else "  "
            print(f"  {flag}[{bar}] {u:.3f}  {tok_str!r}")
        flagged = sum(1 for u in U_arr if u > tau)
        print(f"─────────────────────────────────────────────────────")
        print(f"  Flagged: {flagged}/{len(U_arr)} | Mean U: {sum(U_arr)/max(len(U_arr),1):.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CHAT
# ══════════════════════════════════════════════════════════════════════════════

def chat_loop(generator: LeoGenerator, args):
    print(f"\n{'='*60}")
    print(f"  🦁 {LEO_IDENTITY['full_name']}  |  Built by {LEO_IDENTITY['creator']}")
    print(f"  Commands: /think  /nothink  /agentic  /auto  /clear  /status  /exit")
    print(f"{'='*60}\n")

    mode    = args.mode
    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit"):
            print("  Goodbye!"); break
        if cmd == "/clear":
            history.clear(); print("  [Cleared]\n"); continue
        if cmd == "/think":
            mode = "think";   print(f"  [→ think]\n"); continue
        if cmd == "/nothink":
            mode = "ar";      print(f"  [→ fast]\n"); continue
        if cmd == "/agentic":
            mode = "agentic"; print(f"  [→ agentic]\n"); continue
        if cmd == "/auto":
            mode = "auto";    print(f"  [→ auto]\n"); continue
        if cmd == "/status":
            s = generator.knowledge.status() if generator.knowledge else "not connected"
            print(f"  Knowledge: {s}\n"); continue

        history.append(f"User: {user_input}")
        context = "\n".join(history[-6:])

        try:
            result = generator.generate(
                prompt=context, mode=mode,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                verbose=False,
            )
        except Exception as e:
            print(f"  [Error: {e}]\n"); continue

        if result.get("think_text") and args.show_thinking:
            print(f"\n<think>\n{result['think_text']}\n</think>")

        print(f"\nLeo: {result['answer_text']}\n")
        history.append(f"Leo: {result['answer_text']}")

        if args.show_uncertainty:
            U = result["uncertainty"]
            if isinstance(U, torch.Tensor) and U.numel() > 1:
                tau = generator.cfg.uncertainty_thresh
                print(f"  [U mean={U.mean().item():.3f} | "
                      f"max={U.max().item():.3f} | "
                      f"flagged={(U > tau).sum().item()}]\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=f"🦁 {LEO_IDENTITY['full_name']} — built by {LEO_IDENTITY['creator']}"
    )
    parser.add_argument("--prompt",          type=str,   default=None)
    parser.add_argument("--mode",            type=str,   default="hybrid",
                        choices=["ar","hybrid","diffusion","think","nothink","agentic","auto"])
    parser.add_argument("--max_tokens",      type=int,   default=512)
    parser.add_argument("--temperature",     type=float, default=0.8)
    parser.add_argument("--top_k",           type=int,   default=50)
    parser.add_argument("--top_p",           type=float, default=0.9)
    parser.add_argument("--checkpoint",      type=str,   default="./checkpoints/latest.pt")
    parser.add_argument("--tok_path",        type=str,   default="./leo_tokenizer")
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--yarn_scale",      type=float, default=1.0,
                        help="1.0=32k trained ctx, 4.0=128k at inference")
    parser.add_argument("--config",          type=str,   default="./leo_config.yaml")
    parser.add_argument("--chat",            action="store_true")
    parser.add_argument("--show_uncertainty",action="store_true")
    parser.add_argument("--show_thinking",   action="store_true")
    parser.add_argument("--no_mtp",          action="store_true")
    parser.add_argument("--think_budget",    type=int,   default=None)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  🦁 {LEO_IDENTITY['full_name']}")
    print(f"  Built by : {LEO_IDENTITY['creator']}")
    print(f"  Params   : {LEO_IDENTITY['parameters']}")
    print(f"  Context  : {LEO_IDENTITY['context']}")
    print(f"{'='*60}")

    # Load knowledge layer if enabled in config
    knowledge = None
    if RAG_AVAILABLE and Path(args.config).exists():
        try:
            import yaml
            with open(args.config) as f:
                cfg_yaml = yaml.safe_load(f)
            if cfg_yaml.get("external_knowledge", {}).get("enabled", False):
                print("  Loading knowledge layer...")
                knowledge = LeoKnowledgeLayer.from_config(args.config)
                print(f"  Knowledge : {knowledge.status()}")
        except Exception as e:
            print(f"  Knowledge layer skipped: {e}")

    print()
    generator = LeoGenerator(
        checkpoint      = args.checkpoint,
        device          = args.device,
        tok_path        = args.tok_path,
        yarn_scale      = args.yarn_scale,
        knowledge_layer = knowledge,
    )
    print()

    if args.chat:
        chat_loop(generator, args)
        return

    prompt = args.prompt or "Who are you? Tell me about yourself."

    print(f"  Prompt : {prompt!r}")
    print(f"  Mode   : {args.mode}")
    print()

    result = generator.generate(
        prompt         = prompt,
        mode           = args.mode,
        max_new_tokens = args.max_tokens,
        temperature    = args.temperature,
        top_k          = args.top_k,
        top_p          = args.top_p,
        think_budget   = args.think_budget,
        use_mtp        = not args.no_mtp,
        verbose        = True,
    )

    if result.get("think_text"):
        print(f"\n── Think ──────────────────────────────────────────────")
        print(result["think_text"])
        print(f"── Answer ─────────────────────────────────────────────")
    else:
        print(f"\n── Output ─────────────────────────────────────────────")

    print(result["answer_text"])

    if args.show_uncertainty and result["output_ids"] is not None:
        U = result["uncertainty"]
        if isinstance(U, torch.Tensor) and U.numel() > 0:
            generator.print_uncertainty_map(result["output_ids"], U)

    print()


if __name__ == "__main__":
    main()
