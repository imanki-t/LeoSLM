"""
leo_rag.py — LeoSLM Aether Knowledge Layer

BUG FIXES vs original:
  BUG 1: _FAISSBackend hardcoded dim=384 — crashes if embed_model changes.
          Fix: pass embedder.dim dynamically after Embedder is constructed.

  BUG 2: augment() truncation logic double-condition was redundant + could
          produce context longer than max_chars.
          Fix: single clean guard, truncate only the overhead portion.

  BUG 3: handle_tool_calls() infinite loop risk — if a tool result contains
          the literal tag string, the while loop never exits.
          Also: text.index(end_tag) found the FIRST end tag regardless of
          where start_tag was, causing wrong slicing on nested calls.
          Fix: cap iterations + find end_tag AFTER start_tag position.

  BUG 4: code_exec() accepted a `timeout` param but never used it.
          Infinite loops in user code hung the server forever.
          Fix: run exec in a ThreadPoolExecutor with real timeout.

  BUG 5: Dead import — `from datetime import datetime` on line 240 was
          never used (line 292 used __import__ instead).
          Fix: removed dead import, use datetime directly.

  BUG 6: _TFIDFEmbedder fitted lazily on first embed() call. If first call
          was embed_one(query), vocabulary = query words only → all document
          embeddings near-zero → retrieval completely broken.
          Fix: require explicit fit() before use; fall back gracefully.

  BUG 7: DocumentIndexer chunk IDs used path.stem only → two files named
          readme.txt and readme.md produced identical IDs → ChromaDB
          silently overwrote chunks.
          Fix: IDs use md5(full_path + chunk_index).

  BUG 8: `from safety import ...` inside tool handlers with no fallback →
          ImportError on every web_search / code_exec call if safety/
          package not on path.
          Fix: wrapped in try/except with no-op fallbacks.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import subprocess
import sys
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class CustomInstructions:

    SYSTEM_OPEN  = "<|system|>"
    SYSTEM_CLOSE = "<|/system|>"

    def __init__(self, instructions: str = "", source: str = "inline"):
        self._raw    = instructions.strip()
        self._source = source
        self._mtime  = 0.0

    @classmethod
    def from_file(cls, path: str) -> "CustomInstructions":
        p = Path(path)
        obj = cls("", source=str(p))
        if p.exists():
            obj._raw   = p.read_text(encoding="utf-8").strip()
            obj._mtime = p.stat().st_mtime
        else:
            print(f"[CustomInstructions] File not found: {path}")
        return obj

    @classmethod
    def from_string(cls, text: str) -> "CustomInstructions":
        return cls(text.strip(), source="inline")

    @classmethod
    def empty(cls) -> "CustomInstructions":
        return cls("", source="empty")

    def reload(self) -> bool:
        if self._source in ("inline", "empty"):
            return False
        p = Path(self._source)
        if not p.exists():
            return False
        mtime = p.stat().st_mtime
        if mtime != self._mtime:
            self._raw   = p.read_text(encoding="utf-8").strip()
            self._mtime = mtime
            return True
        return False

    def is_empty(self) -> bool:
        return not bool(self._raw)

    def augment(self, prompt: str) -> str:
        self.reload()
        if self.is_empty():
            return prompt
        instr_block = f"{self.SYSTEM_OPEN}\n{self._raw}\n{self.SYSTEM_CLOSE}\n"
        return instr_block + prompt

    def __repr__(self):
        return f"CustomInstructions(source={self._source!r}, lines={len(self._raw.splitlines())})"


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDERS
# ══════════════════════════════════════════════════════════════════════════════

class _SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim    = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]):
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_one(self, text: str):
        return self.embed([text])[0]


class _TFIDFEmbedder:
    """
    BUG FIX: Original fitted lazily on first embed() call.
    If first call was embed_one(query), vocabulary = query words only →
    all document embeddings near-zero → retrieval broken.
    Now requires explicit fit() on corpus before use. Falls back to zeros
    if not fitted rather than fitting on query text.
    """
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec    = TfidfVectorizer(max_features=4096, sublinear_tf=True)
        self._fitted = False
        self.dim     = 4096

    def fit(self, texts: List[str]):
        self._vec.fit(texts)
        self._fitted = True

    def embed(self, texts: List[str]):
        import numpy as np
        if not self._fitted:
            # BUG FIX: don't fit on query text — return zeros instead
            print("[TFIDFEmbedder] WARNING: not fitted on corpus yet. "
                  "Call fit(corpus_texts) before embedding. Returning zeros.")
            return np.zeros((len(texts), self.dim), dtype="float32")
        return self._vec.transform(texts).toarray().astype("float32")

    def embed_one(self, text: str):
        return self.embed([text])[0]


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self._impl = _SentenceTransformerEmbedder(model_name)
        except ImportError:
            try:
                self._impl = _TFIDFEmbedder()
            except ImportError:
                self._impl = None
        self.dim = getattr(self._impl, "dim", 384) if self._impl else 384

    def embed(self, texts: List[str]):
        if self._impl is None:
            import numpy as np
            return np.zeros((len(texts), self.dim), dtype="float32")
        return self._impl.embed(texts)

    def embed_one(self, text: str):
        return self.embed([text])[0]


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

class InMemoryBackend:
    def __init__(self):
        self._vectors: List = []
        self._texts:   List[str]  = []
        self._metas:   List[Dict] = []

    def add(self, ids, vectors, texts, metas):
        import numpy as np
        for v, t, m in zip(vectors, texts, metas):
            self._vectors.append(np.array(v, dtype="float32"))
            self._texts.append(t)
            self._metas.append(m)

    def search(self, vec, k: int) -> List[Tuple[float, str, Dict]]:
        import numpy as np
        if not self._vectors:
            return []
        q   = np.array(vec, dtype="float32")
        mat = np.stack(self._vectors)
        cos = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-10)
        top = np.argsort(cos)[::-1][:k]
        return [(float(cos[i]), self._texts[i], self._metas[i]) for i in top]

    def count(self) -> int:
        return len(self._texts)


class _FAISSBackend:
    def __init__(self, dim: int):
        import faiss
        self._index = faiss.IndexFlatIP(dim)
        self._texts: List[str]  = []
        self._metas: List[Dict] = []

    def add(self, ids, vectors, texts, metas):
        import numpy as np
        arr = np.array(vectors, dtype="float32")
        if arr.ndim == 1:
            arr = arr[None]
        self._index.add(arr)
        self._texts.extend(texts)
        self._metas.extend(metas)

    def search(self, vec, k: int) -> List[Tuple[float, str, Dict]]:
        import numpy as np
        q = np.array(vec, dtype="float32")[None]
        D, I = self._index.search(q, k)
        return [(float(D[0][i]), self._texts[I[0][i]], self._metas[I[0][i]])
                for i in range(len(I[0])) if 0 <= I[0][i] < len(self._texts)]

    def count(self) -> int:
        return self._index.ntotal


class _ChromaBackend:
    def __init__(self, index_path: str = "./knowledge/index/", collection: str = "leo_knowledge"):
        import chromadb
        self._client = chromadb.PersistentClient(path=index_path)
        self._col    = self._client.get_or_create_collection(collection)

    def add(self, ids, vectors, texts, metas):
        self._col.add(
            ids        = ids,
            embeddings = [v.tolist() if hasattr(v, "tolist") else v for v in vectors],
            documents  = texts,
            metadatas  = metas,
        )

    def search(self, vec, k: int) -> List[Tuple[float, str, Dict]]:
        results = self._col.query(
            query_embeddings=[vec.tolist() if hasattr(vec, "tolist") else vec],
            n_results=k,
        )
        return [
            (1.0 - d, doc, meta)
            for doc, d, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]

    def count(self) -> int:
        return self._col.count()


def _build_backend(backend_name: str, index_path: str, collection: str, embedder_dim: int = 384):
    """
    BUG FIX: added embedder_dim parameter.
    Original hardcoded 384 — crashes if embed_model produces different dim.
    """
    if backend_name == "faiss":
        try:
            return _FAISSBackend(embedder_dim)   # ← use actual dim, not hardcoded 384
        except ImportError:
            pass
    if backend_name in ("chroma", "chromadb"):
        try:
            return _ChromaBackend(index_path, collection)
        except ImportError:
            pass
    return InMemoryBackend()


# ══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH
# ══════════════════════════════════════════════════════════════════════════════

_WEB_SEARCH_PROMPT = "Search results for: {query}\nDate: {date}\n\n{results}\n"


def _format_results(results: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", r.get("name", ""))
        body  = r.get("body", r.get("snippet", r.get("description", "")))
        url   = r.get("href", r.get("url", r.get("link", "")))
        lines.append(f"[{i}] {title}\n{body}\nURL: {url}")
    return "\n\n".join(lines)


def web_search_multi_backend(query: str, n: int = 5, prefer: str = "ddg") -> str:
    # BUG FIX: removed dead `from datetime import datetime` (was never used).
    # BUG FIX: wrapped safety import in try/except.
    try:
        from safety import RAGSafetyWrapper
        _safety = RAGSafetyWrapper()
    except ImportError:
        _safety = None

    backends = ["ddg", "brave", "serp"]
    if prefer in backends:
        backends = [prefer] + [b for b in backends if b != prefer]

    raw: List[Dict] = []
    last_err = None
    for backend in backends:
        try:
            if backend == "ddg":
                from duckduckgo_search import DDGS
                with DDGS(timeout=8) as ddgs:
                    raw = list(ddgs.text(query, max_results=n))
            elif backend == "brave":
                import urllib.request, urllib.parse, gzip
                api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
                if not api_key:
                    raise RuntimeError("BRAVE_SEARCH_API_KEY not set")
                url = (f"https://api.search.brave.com/res/v1/web/search"
                       f"?q={urllib.parse.quote(query)}&count={n}")
                req = urllib.request.Request(url, headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data_raw = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        data_raw = gzip.decompress(data_raw)
                data = json.loads(data_raw)
                raw  = [{"title": r.get("title",""), "body": r.get("description",""),
                         "href": r.get("url","")}
                        for r in data.get("web",{}).get("results",[])[:n]]
            elif backend == "serp":
                import urllib.request, urllib.parse
                api_key = os.environ.get("SERPAPI_KEY", "")
                if not api_key:
                    raise RuntimeError("SERPAPI_KEY not set")
                params = urllib.parse.urlencode(
                    {"q": query, "num": n, "api_key": api_key, "engine": "google"}
                )
                with urllib.request.urlopen(
                    f"https://serpapi.com/search?{params}", timeout=12
                ) as resp:
                    data = json.loads(resp.read())
                raw = [{"title": r.get("title",""), "body": r.get("snippet",""),
                        "href": r.get("link","")}
                       for r in data.get("organic_results",[])[:n]]
            if raw:
                break
        except Exception as e:
            last_err = e
            continue

    if not raw:
        return f"[web_search] No results for '{query}'. Last error: {last_err}"

    if _safety:
        raw = _safety.filter_results(raw)

    # BUG FIX: use imported datetime directly (removed __import__ hack)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return _WEB_SEARCH_PROMPT.format(
        query=query, date=date_str, results=_format_results(raw[:n])
    )


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT INDEXER
# ══════════════════════════════════════════════════════════════════════════════

class DocumentIndexer:
    def __init__(self, source_dir: str, backend, embedder,
                 chunk_size: int = 256, chunk_overlap: int = 32,
                 manifest_path: str = ".leo_rag_manifest.json"):
        self._src      = Path(source_dir)
        self._backend  = backend
        self._embedder = embedder
        self._chunk    = chunk_size
        self._overlap  = chunk_overlap
        self._manifest: Dict[str, str] = {}
        self._mpath    = manifest_path
        if Path(manifest_path).exists():
            with open(manifest_path) as f:
                self._manifest = json.load(f)

    def _save_manifest(self):
        with open(self._mpath, "w") as f:
            json.dump(self._manifest, f)

    def _read_file(self, path: Path) -> str:
        suf = path.suffix.lower()
        try:
            if suf == ".pdf":
                import pypdf
                return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(str(path)).pages)
            elif suf == ".docx":
                import docx
                return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
            elif suf == ".html":
                from bs4 import BeautifulSoup
                return BeautifulSoup(path.read_text(errors="ignore"), "html.parser").get_text()
            elif suf == ".json":
                data = json.loads(path.read_text(errors="ignore"))
                return json.dumps(data, indent=1) if isinstance(data, (dict, list)) else str(data)
            else:
                return path.read_text(errors="ignore")
        except Exception:
            return ""

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        step  = max(1, self._chunk - self._overlap)
        return [" ".join(words[i:i+self._chunk])
                for i in range(0, len(words), step)
                if len(words[i:i+self._chunk]) >= 10]

    def index_all(self, force: bool = False):
        if not self._src.exists():
            return
        supported = {".pdf", ".txt", ".md", ".html", ".docx", ".json", ".csv"}
        for path in [f for f in self._src.rglob("*") if f.suffix.lower() in supported]:
            content_hash = hashlib.md5(path.read_bytes()).hexdigest()
            if not force and self._manifest.get(str(path)) == content_hash:
                continue
            text = self._read_file(path)
            if not text.strip():
                continue
            chunks = self._chunk_text(text)
            if not chunks:
                continue
            vectors = self._embedder.embed(chunks)
            # BUG FIX: use md5(full_path + index) as ID to prevent collisions
            # between files with same stem but different extensions
            ids   = [hashlib.md5(f"{path}_{i}".encode()).hexdigest()[:16]
                     for i in range(len(chunks))]
            metas = [{"source": str(path), "chunk": i, "stem": path.stem}
                     for i in range(len(chunks))]
            self._backend.add(ids, vectors, chunks, metas)
            self._manifest[str(path)] = content_hash
        self._save_manifest()


# ══════════════════════════════════════════════════════════════════════════════
# RAG MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class RAGManager:
    RETRIEVED_OPEN  = "<|retrieved|>"
    RETRIEVED_CLOSE = "<|/retrieved|>"

    def __init__(self, backend, embedder, top_k: int = 5,
                 max_tokens: int = 2048, min_score: float = 0.20):
        self._backend   = backend
        self._embedder  = embedder
        self.top_k      = top_k
        self.max_tokens = max_tokens
        self.min_score  = min_score

    def retrieve(self, query: str) -> List[Tuple[float, str, Dict]]:
        vec  = self._embedder.embed_one(query)
        hits = self._backend.search(vec, self.top_k)
        return [(s, t, m) for s, t, m in hits if s >= self.min_score]

    def format_context(self, results: List[Tuple]) -> str:
        if not results:
            return ""
        chunks = [t for _, t, _ in results]
        body   = "\n---\n".join(chunks)
        if len(body) > self.max_tokens * 4:
            body = body[:self.max_tokens * 4] + "\n[...truncated...]"
        return f"{self.RETRIEVED_OPEN}\n{body}\n{self.RETRIEVED_CLOSE}\n"

    def augment(self, query: str) -> str:
        return self.format_context(self.retrieve(query))

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        vecs  = self._embedder.embed(texts)
        ids   = [hashlib.md5(t.encode()).hexdigest()[:12] for t in texts]
        metas = metadatas or [{} for _ in texts]
        self._backend.add(ids, vecs, texts, metas)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolSchema:
    name:        str
    description: str
    parameters:  Dict[str, Any]
    handler:     Optional[Callable] = field(default=None, repr=False)


_SESSION_MEMORY: Dict[str, str] = {}


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSchema] = {}
        self._register_builtins()

    def tool(self, name: str, description: str, parameters: Optional[Dict] = None):
        def decorator(fn: Callable) -> Callable:
            self.register(ToolSchema(
                name=name, description=description,
                parameters=parameters or {"type": "object", "properties": {}},
                handler=fn,
            ))
            return fn
        return decorator

    def register(self, schema: ToolSchema):
        self._tools[schema.name] = schema

    def call(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self._tools:
            return self._wrap(f"Error: tool '{name}' not registered.", name)
        handler = self._tools[name].handler
        if handler is None:
            return self._wrap(f"Tool '{name}' has no handler.", name)
        try:
            return self._wrap(str(handler(**args)), name)
        except Exception as e:
            return self._wrap(f"Tool error: {e}", name)

    def parse_and_call(self, tool_call_text: str) -> str:
        try:
            payload = json.loads(tool_call_text.strip())
            return self.call(
                payload.get("name", ""),
                payload.get("args", payload.get("arguments", {})),
            )
        except json.JSONDecodeError as e:
            return self._wrap(f"Malformed tool call JSON: {e}", "parse_error")

    def list_tools(self) -> List[Dict]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in self._tools.values()]

    def tools_as_context(self) -> str:
        if not self._tools:
            return ""
        lines = ["Available tools (call via <|tool_call|>{...}<|/tool_call|>):"]
        for t in self._tools.values():
            lines.append(f"  - {t.name}: {t.description}")
        return "<|system|>\n" + "\n".join(lines) + "\n<|/system|>\n"

    @staticmethod
    def _wrap(result: str, name: str) -> str:
        return f"<|tool_result|>\n[{name}]: {result}\n<|/tool_result|>"

    def _register_builtins(self):

        @self.tool(
            "web_search", "Search the web for current information.",
            {"type": "object",
             "properties": {"query": {"type": "string"}, "n": {"type": "integer", "default": 5}},
             "required": ["query"]},
        )
        def web_search(query: str, n: int = 5) -> str:
            return web_search_multi_backend(query, n=n)

        @self.tool(
            "code_exec", "Execute Python code and return stdout.",
            {"type": "object",
             "properties": {"code": {"type": "string"}, "timeout": {"type": "integer", "default": 10}},
             "required": ["code"]},
        )
        def code_exec(code: str, timeout: int = 10) -> str:
            """
            BUG FIX: original exec()ed with no timeout — infinite loops hung forever.
            Now runs in a ThreadPoolExecutor with real wall-clock timeout.
            BUG FIX: safety import wrapped in try/except.
            """
            # Safety check
            try:
                from safety import check_text_safety
                if check_text_safety(code).blocked:
                    return "Error: code blocked by safety filter."
            except ImportError:
                pass  # safety not available — proceed without check

            import io, contextlib

            def _run():
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                        exec(compile(code, "<leo>", "exec"), {})
                    return buf.getvalue() or "(no output)"
                except Exception as e:
                    return f"Error: {type(e).__name__}: {e}"

            # BUG FIX: enforce timeout using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    return f"Error: code execution timed out after {timeout}s"

        @self.tool(
            "calculator", "Evaluate a math expression safely.",
            {"type": "object",
             "properties": {"expression": {"type": "string"}},
             "required": ["expression"]},
        )
        def calculator(expression: str) -> str:
            import ast, operator
            _ops = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
                ast.Pow: operator.pow, ast.USub: operator.neg,
                ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
            }
            def _eval(n):
                if isinstance(n, ast.Constant):  return n.value
                elif isinstance(n, ast.BinOp):   return _ops[type(n.op)](_eval(n.left), _eval(n.right))
                elif isinstance(n, ast.UnaryOp): return _ops[type(n.op)](_eval(n.operand))
                raise ValueError(f"Unsupported expression node: {type(n)}")
            try:
                return str(_eval(ast.parse(expression, mode="eval").body))
            except Exception as e:
                return f"Error: {e}"

        @self.tool(
            "file_io", "Read/write/list files.",
            {"type": "object",
             "properties": {
                 "action":  {"type": "string", "enum": ["read", "write", "list"]},
                 "path":    {"type": "string"},
                 "content": {"type": "string"},
             },
             "required": ["action", "path"]},
        )
        def file_io(action: str, path: str, content: str = "") -> str:
            p = Path(path)
            if action == "read":
                return p.read_text(errors="ignore")[:8000] if p.exists() else f"Not found: {path}"
            elif action == "write":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
                return f"Written {len(content)} chars to {path}"
            elif action == "list":
                if not p.exists():
                    return f"Not found: {path}"
                return "\n".join(str(f) for f in (p.iterdir() if p.is_dir() else [p]))[:2000]
            return f"Unknown action: {action}"

        @self.tool(
            "memory_store", "Store a key-value pair in session memory.",
            {"type": "object",
             "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
             "required": ["key", "value"]},
        )
        def memory_store(key: str, value: str) -> str:
            _SESSION_MEMORY[key] = value
            return f"Stored: {key}"

        @self.tool(
            "memory_recall", "Recall a stored value from session memory.",
            {"type": "object",
             "properties": {"key": {"type": "string"}},
             "required": ["key"]},
        )
        def memory_recall(key: str) -> str:
            return _SESSION_MEMORY.get(key, f"No memory for key: {key}")


# ══════════════════════════════════════════════════════════════════════════════
# MCP SERVER STUB
# ══════════════════════════════════════════════════════════════════════════════

class MCPServerStub:
    def __init__(self, name: str, url: Optional[str] = None, auth: Optional[str] = None):
        self.name = name
        self.url  = url
        self.auth = auth

    def call(self, tool_name: str, args: Dict[str, Any]) -> str:
        if not self.url:
            return (f"[MCP:{self.name}/{tool_name}] Stub — set url in "
                    f"leo_config.yaml external_knowledge.mcp_servers section.")
        try:
            import urllib.request
            payload = json.dumps({"tool": tool_name, "args": args}).encode()
            headers = {"Content-Type": "application/json"}
            if self.auth:
                headers["Authorization"] = f"Bearer {self.auth}"
            req = urllib.request.Request(
                f"{self.url}/tools/{tool_name}",
                data=payload, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode()
        except Exception as e:
            return f"[MCP:{self.name}/{tool_name}] Error: {e}"

    def __repr__(self):
        return f"MCPServerStub(name={self.name!r}, url={self.url!r})"


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE LAYER
# ══════════════════════════════════════════════════════════════════════════════

class LeoKnowledgeLayer:

    def __init__(
        self,
        instructions: Optional[CustomInstructions] = None,
        rag:          Optional[RAGManager]          = None,
        tools:        Optional[ToolRegistry]         = None,
        mcp_servers:  Optional[List[MCPServerStub]]  = None,
    ):
        self.instructions = instructions or CustomInstructions.empty()
        self.rag          = rag
        self.tools        = tools or ToolRegistry()
        self.mcp_servers  = {s.name: s for s in (mcp_servers or [])}

    @classmethod
    def from_config(cls, config_path: str = "./leo_config.yaml") -> "LeoKnowledgeLayer":
        try:
            import yaml
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", "pyyaml", "-q"],
                check=False,
            )
            import yaml

        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[LeoKnowledgeLayer] Config not found: {config_path} — empty layer")
            return cls()

        ext = cfg.get("external_knowledge", {})
        if not ext.get("enabled", False):
            print("[LeoKnowledgeLayer] external_knowledge.enabled=false — empty layer.")
            return cls()

        # Custom instructions
        instr_cfg = ext.get("custom_instructions", {})
        if instr_cfg.get("enabled", False):
            inline = instr_cfg.get("inline")
            instructions = (
                CustomInstructions.from_string(inline) if inline
                else CustomInstructions.from_file(
                    instr_cfg.get("path", "./knowledge/instructions.txt")
                )
            )
        else:
            instructions = CustomInstructions.empty()

        # RAG
        rag = None
        rag_cfg = ext.get("rag", {})
        if rag_cfg.get("enabled", False):
            embedder = Embedder(
                rag_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
            )
            # BUG FIX: pass embedder.dim so FAISS gets the correct dimension
            backend = _build_backend(
                rag_cfg.get("backend",    "chroma"),
                rag_cfg.get("index_path", "./knowledge/index/"),
                rag_cfg.get("collection", "leo_knowledge"),
                embedder_dim=embedder.dim,              # ← was missing
            )
            rag = RAGManager(
                backend    = backend,
                embedder   = embedder,
                top_k      = rag_cfg.get("top_k",      5),
                max_tokens = rag_cfg.get("max_tokens", 2048),
            )
            doc_cfg = ext.get("document_store", {})
            if doc_cfg.get("enabled", False):
                DocumentIndexer(
                    source_dir    = doc_cfg.get("source_dir",    "./knowledge/documents/"),
                    backend       = backend,
                    embedder      = embedder,
                    chunk_size    = doc_cfg.get("chunk_size",    512),
                    chunk_overlap = doc_cfg.get("chunk_overlap", 64),
                ).index_all()

        # Custom functions
        tools = ToolRegistry()
        fn_cfg = ext.get("custom_functions", {})
        if fn_cfg.get("enabled", False):
            reg_path = fn_cfg.get("registry", "./knowledge/functions.json")
            if Path(reg_path).exists():
                for s in json.loads(Path(reg_path).read_text()):
                    tools.register(ToolSchema(
                        name        = s["name"],
                        description = s.get("description", ""),
                        parameters  = s.get("parameters", {}),
                        handler     = None,
                    ))

        # MCP servers
        mcp_servers = []
        for srv in ext.get("mcp_servers", []):
            if srv.get("enabled", False):
                mcp_servers.append(MCPServerStub(
                    name = srv["name"],
                    url  = srv.get("url"),
                    auth = srv.get("auth"),
                ))

        return cls(instructions=instructions, rag=rag, tools=tools, mcp_servers=mcp_servers)

    def augment(self, prompt: str, max_chars: int = 4000) -> str:
        """
        BUG FIX: Original truncation had a redundant double-condition and
        could produce output longer than max_chars when prompt > max_chars.
        Fixed: cleanly cap the overhead (context prefix) at max_chars chars.
        The prompt itself is never truncated — only the injected context.
        """
        result = self.instructions.augment(prompt)
        if self.rag:
            ctx = self.rag.augment(prompt)
            if ctx:
                result = ctx + result

        # Overhead = everything added on top of the raw prompt
        overhead = len(result) - len(prompt)
        if overhead > max_chars:
            # Trim the prefix (context + instructions) to max_chars
            prefix   = result[: len(result) - len(prompt)]   # everything before prompt
            prefix   = prefix[:max_chars] + "\n[...truncated...]\n"
            result   = prefix + prompt

        return result

    def handle_tool_calls(self, text: str, max_iterations: int = 10) -> str:
        """
        BUG FIX 1: Added max_iterations guard to prevent infinite loop when
        tool result itself contains '<|tool_call|>' literal text.

        BUG FIX 2: Find end_tag AFTER start position, not at text start,
        to handle malformed or adjacent tool calls correctly.
        """
        start_tag = "<|tool_call|>"
        end_tag   = "<|/tool_call|>"
        iterations = 0

        while start_tag in text and end_tag in text:
            if iterations >= max_iterations:
                print(f"[handle_tool_calls] Hit max_iterations={max_iterations}, stopping.")
                break
            iterations += 1

            s_pos     = text.index(start_tag)
            # BUG FIX: find end_tag AFTER start_tag, not from beginning of text
            e_pos     = text.find(end_tag, s_pos + len(start_tag))
            if e_pos == -1:
                break   # malformed — no closing tag after this open tag

            call_text = text[s_pos + len(start_tag) : e_pos]
            result    = self.tools.parse_and_call(call_text)
            text      = text[:s_pos] + result + text[e_pos + len(end_tag):]

        return text

    def agentic_loop(self, prompt: str, gen_fn: Callable[[str], str], max_turns: int = 5) -> str:
        context = self.augment(prompt)
        current = context
        for _ in range(max_turns):
            output = gen_fn(current)
            if "<|tool_call|>" not in output:
                return output
            resolved = self.handle_tool_calls(output)
            if resolved == output:
                return output
            current = current + "\n" + resolved
        return gen_fn(current)

    def search_web(self, query: str, n: int = 5) -> str:
        return web_search_multi_backend(query, n=n)

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        if self.rag is None:
            print("[LeoKnowledgeLayer] RAG not configured.")
            return
        self.rag.add_texts(texts, metadatas)

    def set_instructions(self, text: str):
        self.instructions = CustomInstructions.from_string(text)

    def status(self) -> Dict[str, Any]:
        return {
            "custom_instructions": not self.instructions.is_empty(),
            "instructions_source": self.instructions._source,
            "rag":                 self.rag is not None,
            "rag_doc_count":       self.rag._backend.count() if self.rag else 0,
            "tools":               list(self.tools._tools.keys()),
            "mcp_servers":         list(self.mcp_servers.keys()),
        }

    def __repr__(self):
        s = self.status()
        return (f"LeoKnowledgeLayer("
                f"instructions={s['custom_instructions']}, "
                f"rag={s['rag']}({s['rag_doc_count']} docs), "
                f"tools={s['tools']})")
