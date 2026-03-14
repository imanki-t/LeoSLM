import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PJRT_DEVICE", "TPU")

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    class _XMFallback:
        def xla_device(self):       return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def mark_step(self):        pass
        def master_print(self, *a): print(*a)
    xm = _XMFallback()
    XLA_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from model import LeoSLM, LeoConfig, CFG, LEO_IDENTITY, LEO_SYSTEM_PROMPT

try:
    from leo_rag import LeoKnowledgeLayer
    RAG_AVAILABLE = True
except ImportError:
    LeoKnowledgeLayer = None
    RAG_AVAILABLE     = False


def load_tokenizer(path: str = "./leo_tokenizer"):
    from transformers import PreTrainedTokenizerFast, AutoTokenizer
    if Path(f"{path}/tokenizer.json").exists():
        tok = PreTrainedTokenizerFast.from_pretrained(path)
        print(f"  Tokenizer : LeoTokenizer (vocab={tok.vocab_size:,})")
        return tok
    print("  Tokenizer : LeoTokenizer not found — using GPT-2 + Aether tokens")
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


def set_yarn_scale(model: LeoSLM, scale: float):
    for block in model.blocks:
        if hasattr(block, "attn") and hasattr(block.attn, "cfg"):
            block.attn.cfg.yarn_scale = scale
    model.cfg.yarn_scale = scale


def sample_token(
    logits:      torch.Tensor,
    temperature: float,
    top_k:       int,
    top_p:       float,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        kth    = logits.topk(min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_l, sorted_idx = logits.sort(descending=True)
        cum    = sorted_l.softmax(-1).cumsum(-1)
        remove = cum - sorted_l.softmax(-1) > top_p
        sorted_l[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_l)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


class LeoGenerator:

    def __init__(
        self,
        checkpoint:  Optional[str]       = None,
        config:      Optional[LeoConfig] = None,
        device:      Optional[str]       = None,
        tok_path:    str                 = "./leo_tokenizer",
        yarn_scale:  float               = 1.0,
        knowledge_layer                  = None,
    ):
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
            state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
            state = {
                k.replace("_orig_mod.", "").replace("module.", ""): v
                for k, v in state.items()
            }
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Warning  : {len(missing)} missing keys")
            if unexpected:
                print(f"  Warning  : {len(unexpected)} unexpected keys (ignored)")
            print(f"  Checkpoint: {checkpoint}")
        else:
            if checkpoint:
                print(f"  Checkpoint: NOT FOUND — {checkpoint}")
            print("  Weights   : random (no checkpoint loaded)")

        if self.device.type in ("xla", "cuda"):
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device)
        self.model.eval()

        if yarn_scale > 1.0:
            set_yarn_scale(self.model, yarn_scale)
            ctx_k = int(self.cfg.max_seq_len * yarn_scale / 1024)
            print(f"  YaRN      : {yarn_scale}x -> {ctx_k}k context")

        self.tok       = load_tokenizer(tok_path)
        self.knowledge = knowledge_layer

        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Model     : LeoSLM Aether | {total/1e9:.2f}B params | {self.device}")

    @torch.no_grad()
    def _ar_generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int,
        temperature:    float,
        top_k:          int,
        top_p:          float,
        stop_at:        Optional[List[int]] = None,
        use_mtp:        bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids      = input_ids.clone()
        all_unc  = []
        stop_ids = set(stop_at or [self.cfg.eos_id])

        while ids.shape[1] - input_ids.shape[1] < max_new_tokens:
            out    = self.model(ids)
            logits = out["ar_logits"][:, -1, :]
            U      = out["uncertainty"][:, -1]

            if use_mtp and self.cfg.use_mtp and self.model.mtp is not None:
                h_norm = self.model.final_norm(out["hidden"])
                drafts = self.model.mtp.speculative_draft(h_norm, temperature=temperature)

                tok_id = sample_token(logits, temperature, top_k, top_p)
                ids    = torch.cat([ids, tok_id], dim=1)
                all_unc.append(U)
                if tok_id.cpu().item() in stop_ids:
                    break

                hit_stop = False
                for di in range(drafts.shape[1]):
                    if ids.shape[1] - input_ids.shape[1] >= max_new_tokens:
                        break
                    draft_tok = drafts[:, di:di+1]
                    verify    = self.model(ids)
                    v_logits  = verify["ar_logits"][:, -1, :]
                    v_U       = verify["uncertainty"][:, -1]
                    expected  = v_logits.argmax(-1, keepdim=True)

                    if expected.cpu().item() == draft_tok.cpu().item():
                        ids = torch.cat([ids, draft_tok], dim=1)
                        all_unc.append(v_U)
                        if draft_tok.cpu().item() in stop_ids:
                            hit_stop = True
                            break
                    else:
                        corrected = sample_token(v_logits, temperature, top_k, top_p)
                        ids = torch.cat([ids, corrected], dim=1)
                        all_unc.append(v_U)
                        if corrected.cpu().item() in stop_ids:
                            hit_stop = True
                        break

                if hit_stop:
                    break

            else:
                tok_id = sample_token(logits, temperature, top_k, top_p)
                ids    = torch.cat([ids, tok_id], dim=1)
                all_unc.append(U)
                if tok_id.cpu().item() in stop_ids:
                    break

            if XLA_AVAILABLE:
                xm.mark_step()

        uncertainty = torch.stack(all_unc).squeeze(-1) if all_unc else torch.zeros(1)
        return ids, uncertainty

    @torch.no_grad()
    def _diffusion_refine(self, ids: torch.Tensor, n_steps: int = 8) -> torch.Tensor:
        tau     = self.cfg.uncertainty_thresh
        refined = ids.clone()
        for _ in range(n_steps):
            out         = self.model(refined)
            mask_float  = (out["uncertainty"] > tau).float()
            masked      = (refined * (1 - mask_float.long())
                           + self.cfg.mask_id * mask_float.long())
            d_out       = self.model(masked)
            new_ids     = d_out["diff_logits"].argmax(-1)
            refined     = (refined * (1 - mask_float.long())
                           + new_ids * mask_float.long())
        return refined

    @torch.no_grad()
    def _run_ar(self, ids, max_new, temp, top_k, top_p, use_mtp, verbose) -> Dict:
        if verbose:
            print("  [AR mode]")
        out_ids, unc = self._ar_generate(ids, max_new, temp, top_k, top_p, use_mtp=use_mtp)
        text         = self.tok.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True)
        return dict(text=text, output_ids=out_ids, uncertainty=unc,
                    think_text=None, answer_text=text, mode="ar")

    @torch.no_grad()
    def _run_hybrid(self, ids, max_new, temp, top_k, top_p, use_mtp, mode, verbose) -> Dict:
        if verbose:
            print(f"  [{mode} mode]")
        out_ids, unc = self._ar_generate(ids, max_new, temp, top_k, top_p, use_mtp=use_mtp)
        if mode in ("diffusion", "hybrid"):
            out_ids = self._diffusion_refine(out_ids)
        text = self.tok.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True)
        return dict(text=text, output_ids=out_ids, uncertainty=unc,
                    think_text=None, answer_text=text, mode=mode)

    @torch.no_grad()
    def _run_think(self, ids, max_new, temp, top_k, top_p, budget, use_mtp, verbose) -> Dict:
        think_tok = torch.tensor([[self.cfg.think_start_id]], device=self.device)
        ids       = torch.cat([ids, think_tok], dim=1)

        if budget is None:
            init_out = self.model(ids)
            budget   = self.model.get_think_budget(init_out["uncertainty"])
        budget = max(budget, self.cfg.think_budget_min)
        if verbose:
            print(f"  [Think mode | budget={budget} tokens]")

        stop = [self.cfg.think_end_id, self.cfg.eos_id]
        think_ids, think_unc = self._ar_generate(
            ids, budget, temp, top_k, top_p, stop_at=stop, use_mtp=use_mtp
        )

        if think_ids[0, -1].cpu().item() != self.cfg.think_end_id:
            end_tok   = torch.tensor([[self.cfg.think_end_id]], device=self.device)
            think_ids = torch.cat([think_ids, end_tok], dim=1)

        ans_ids, ans_unc = self._ar_generate(
            think_ids, max_new, temp, top_k, top_p, use_mtp=use_mtp
        )

        think_start = ids.shape[1]
        think_end   = think_ids.shape[1]

        think_text  = self.tok.decode(think_ids[0, think_start:think_end], skip_special_tokens=True)
        answer_text = self.tok.decode(ans_ids[0, think_end:],              skip_special_tokens=True)
        full_text   = think_text + "\n\n" + answer_text
        full_unc    = torch.cat([think_unc, ans_unc])

        return dict(text=full_text, output_ids=ans_ids, uncertainty=full_unc,
                    think_text=think_text, answer_text=answer_text, mode="think")

    @torch.no_grad()
    def _run_agentic(self, ids, raw_prompt, max_new, temp, top_k, top_p, use_mtp, verbose) -> Dict:
        if verbose:
            print("  [Agentic mode | TTIP + ACGI]")

        if not RAG_AVAILABLE or self.knowledge is None:
            if verbose:
                print("  LeoKnowledgeLayer not connected — falling back to think mode")
            return self._run_think(ids, max_new, temp, top_k, top_p, None, use_mtp, verbose)

        def _gen_fn(prompt_str: str) -> str:
            enc   = self.tok(prompt_str, return_tensors="pt",
                             truncation=True, max_length=self.cfg.max_seq_len)
            g_ids = enc["input_ids"].to(self.device)
            o_ids, _ = self._ar_generate(g_ids, max_new, temp, top_k, top_p, use_mtp=use_mtp)
            return self.tok.decode(o_ids[0, g_ids.shape[1]:], skip_special_tokens=False)

        final  = self.knowledge.agentic_loop(raw_prompt, _gen_fn, max_turns=5)
        clean  = self.tok.decode(self.tok.encode(final), skip_special_tokens=True)
        U      = torch.zeros(max(len(clean.split()), 1), device=self.device)
        return dict(text=clean, output_ids=None, uncertainty=U,
                    think_text=None, answer_text=clean, tool_calls=[], mode="agentic")

    @torch.no_grad()
    def generate(
        self,
        prompt:         str,
        mode:           str   = "hybrid",
        max_new_tokens: int   = 512,
        temperature:    float = 0.8,
        top_k:          int   = 50,
        top_p:          float = 0.9,
        think_budget:   Optional[int] = None,
        use_mtp:        bool  = True,
        verbose:        bool  = True,
    ) -> Dict:
        raw_prompt = prompt
        if self.knowledge and mode == "agentic":
            prompt = self.knowledge.augment(prompt)

        full = LEO_SYSTEM_PROMPT + "\n" + prompt
        enc  = self.tok(full, return_tensors="pt",
                        truncation=True, max_length=self.cfg.max_seq_len // 2)
        ids  = enc["input_ids"].to(self.device)

        if mode in ("ar", "nothink"):
            return self._run_ar(ids, max_new_tokens, temperature, top_k, top_p, use_mtp, verbose)

        if mode in ("hybrid", "diffusion"):
            return self._run_hybrid(ids, max_new_tokens, temperature, top_k, top_p,
                                    use_mtp, mode, verbose)

        if mode == "think":
            return self._run_think(ids, max_new_tokens, temperature, top_k, top_p,
                                   think_budget, use_mtp, verbose)

        if mode == "agentic":
            return self._run_agentic(ids, raw_prompt, max_new_tokens, temperature,
                                     top_k, top_p, use_mtp, verbose)

        if mode == "auto":
            with torch.no_grad():
                init = self.model(ids)
            budget = self.model.get_think_budget(init["uncertainty"])
            if budget > 0:
                return self._run_think(ids, max_new_tokens, temperature, top_k, top_p,
                                       budget, use_mtp, verbose)
            return self._run_ar(ids, max_new_tokens, temperature, top_k, top_p, use_mtp, verbose)

        raise ValueError(
            f"Unknown mode {mode!r}. "
            "Choose: ar | diffusion | hybrid | think | nothink | agentic | auto"
        )

    def print_uncertainty_map(
        self,
        output_ids:  torch.Tensor,
        uncertainty: torch.Tensor,
        prompt_len:  int = 0,
    ):
        toks = output_ids[0, prompt_len:].tolist()
        U    = uncertainty.tolist()
        tau  = self.cfg.uncertainty_thresh
        print("\n-- Uncertainty map (# uncertain | . confident) --")
        for tok_id, u in zip(toks[:len(U)], U):
            word = self.tok.decode([tok_id])
            bar  = "#" if u > tau else "."
            print(f"  {bar} {u:.2f}  {word!r}")
        print()


def chat_loop(generator: LeoGenerator, args):
    print(f"\n{'='*60}")
    print(f"  Leo Aether -- Interactive Chat")
    print(f"  Mode: {args.mode} | /think /nothink /agentic /auto /clear /exit")
    print(f"{'='*60}\n")

    history: List[str] = []
    mode = args.mode

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
            print("  Goodbye!")
            break
        if cmd == "/clear":
            history.clear()
            print("  [Conversation cleared]\n")
            continue
        if cmd == "/think":
            mode = "think"
            print("  [-> think mode]\n")
            continue
        if cmd == "/nothink":
            mode = "ar"
            print("  [-> fast mode]\n")
            continue
        if cmd == "/agentic":
            mode = "agentic"
            print("  [-> agentic mode]\n")
            continue
        if cmd == "/auto":
            mode = "auto"
            print("  [-> auto mode]\n")
            continue
        if cmd == "/status":
            s = generator.knowledge.status() if generator.knowledge else "not connected"
            print(f"  Knowledge layer: {s}\n")
            continue

        history.append(f"User: {user_input}")
        context = "\n".join(history[-6:])

        try:
            result = generator.generate(
                prompt         = context,
                mode           = mode,
                max_new_tokens = args.max_tokens,
                temperature    = args.temperature,
                top_k          = args.top_k,
                top_p          = args.top_p,
                verbose        = False,
            )
        except Exception as e:
            print(f"  [Error: {e}]\n")
            continue

        if result.get("think_text") and args.show_thinking:
            print(f"\n<think>\n{result['think_text']}\n</think>")

        print(f"\nLeo: {result['answer_text']}\n")
        history.append(f"Leo: {result['answer_text']}")

        if args.show_uncertainty and result.get("output_ids") is not None:
            generator.print_uncertainty_map(
                result["output_ids"], result["uncertainty"], prompt_len=0,
            )


def main():
    parser = argparse.ArgumentParser(description="Leo Aether -- Generate")
    parser.add_argument("--prompt",            type=str,   default=None)
    parser.add_argument("--mode",              type=str,   default="hybrid",
                        choices=["ar","diffusion","hybrid","think","nothink","agentic","auto"])
    parser.add_argument("--checkpoint",        type=str,   default=None)
    parser.add_argument("--tok_path",          type=str,   default="./leo_tokenizer")
    parser.add_argument("--max_tokens",        type=int,   default=512)
    parser.add_argument("--temperature",       type=float, default=0.8)
    parser.add_argument("--top_k",             type=int,   default=50)
    parser.add_argument("--top_p",             type=float, default=0.9)
    parser.add_argument("--yarn",              type=float, default=1.0)
    parser.add_argument("--no_mtp",            action="store_true")
    parser.add_argument("--chat",              action="store_true")
    parser.add_argument("--show_thinking",     action="store_true")
    parser.add_argument("--show_uncertainty",  action="store_true")
    parser.add_argument("--device",            type=str,   default=None)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Leo Aether -- built by {LEO_IDENTITY['creator']}")
    print(f"  Arch: {LEO_IDENTITY['architecture']}")
    print(f"{'='*60}")

    knowledge = None
    if RAG_AVAILABLE and LeoKnowledgeLayer is not None:
        try:
            knowledge = LeoKnowledgeLayer()
            print("  Knowledge : LeoKnowledgeLayer connected")
        except Exception as e:
            print(f"  Knowledge : not available ({e})")

    generator = LeoGenerator(
        checkpoint      = args.checkpoint,
        device          = args.device,
        tok_path        = args.tok_path,
        yarn_scale      = args.yarn,
        knowledge_layer = knowledge,
    )

    if args.chat:
        chat_loop(generator, args)
        return

    if not args.prompt:
        parser.error("Provide --prompt or use --chat for interactive mode")

    result = generator.generate(
        prompt         = args.prompt,
        mode           = args.mode,
        max_new_tokens = args.max_tokens,
        temperature    = args.temperature,
        top_k          = args.top_k,
        top_p          = args.top_p,
        use_mtp        = not args.no_mtp,
        verbose        = True,
    )

    print(f"\n{'='*60}")
    if result.get("think_text"):
        if args.show_thinking:
            print(f"<think>\n{result['think_text']}\n</think>\n")
        else:
            tlen = len(result["think_text"].split())
            print(f"[Think: {tlen} words] (use --show_thinking to display)")
    print(f"Leo: {result['answer_text']}")
    print(f"{'='*60}")

    if args.show_uncertainty and result.get("output_ids") is not None:
        generator.print_uncertainty_map(result["output_ids"], result["uncertainty"])


if __name__ == "__main__":
    main()
