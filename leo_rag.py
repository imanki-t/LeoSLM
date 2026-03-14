"""
LeoSLM "Aether" — leo_rag.py
==============================
External knowledge integration layer. Completely decoupled from model weights.
Plug this in AFTER training — zero changes to train.py required.

Three orthogonal systems, all optional, all hot-swappable:

  1. CustomInstructions — static operator/user rules prepended to every turn
  2. RAGManager         — retrieval-augmented generation (vector store query)
  3. ToolRegistry       — custom function calling + MCP server stubs

Usage at inference (in generate.py or your serving layer):
    from leo_rag import LeoKnowledgeLayer
    layer = LeoKnowledgeLayer.from_config("./leo_config.yaml")
    prompt = layer.augment(user_message)
    # feed `prompt` to model instead of raw user_message

Backends supported (install the one you use, others stay optional):
    chroma    : pip install chromadb
    faiss     : pip install faiss-cpu
    pinecone  : pip install pinecone-client
    weaviate  : pip install weaviate-client
    qdrant    : pip install qdrant-client
    sentence-transformers (for local embedding): pip install sentence-transformers
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  CUSTOM INSTRUCTIONS  ██████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class CustomInstructions:
    """
    Static text prepended to every conversation as additional system context.

    Use-cases:
    - Persona overrides ("Always respond in French")
    - Domain rules ("Only answer questions about our product")
    - Operator instructions ("Never discuss competitors")
    - User preferences ("Be concise, use bullet points")

    Instructions are prepended INSIDE the <|system|> block so Leo treats
    them as system-level directives, not user input.

    Hot-reload: call .reload() to pick up file changes without restarting.
    """

    SYSTEM_OPEN  = "<|system|>"
    SYSTEM_CLOSE = "<|/system|>"

    def __init__(self, instructions: str = "", source: str = "inline"):
        self._raw     = instructions.strip()
        self._source  = source
        self._mtime   = 0.0

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str) -> "CustomInstructions":
        """Load instructions from a plain-text file. Auto-reloads on change."""
        p = Path(path)
        if not p.exists():
            print(f"[CustomInstructions] File not found: {path} — using empty instructions")
            return cls("", source=path)
        obj = cls(p.read_text(encoding="utf-8"), source=path)
        obj._mtime = p.stat().st_mtime
        return obj

    @classmethod
    def from_string(cls, text: str) -> "CustomInstructions":
        return cls(text, source="inline")

    @classmethod
    def empty(cls) -> "CustomInstructions":
        return cls("", source="none")

    # ── Core ──────────────────────────────────────────────────────────────────

    def reload(self):
        """Reload from file if source is a path and the file has changed."""
        if self._source in ("inline", "none"):
            return
        p = Path(self._source)
        if p.exists():
            mtime = p.stat().st_mtime
            if mtime > self._mtime:
                self._raw   = p.read_text(encoding="utf-8").strip()
                self._mtime = mtime
                print(f"[CustomInstructions] Reloaded from {self._source}")

    def as_system_block(self) -> str:
        """Return the instructions wrapped in Leo's system token tags."""
        if not self._raw:
            return ""
        return f"{self.SYSTEM_OPEN}\n{self._raw}\n{self.SYSTEM_CLOSE}\n"

    def is_empty(self) -> bool:
        return not bool(self._raw.strip())

    def __repr__(self):
        preview = self._raw[:80].replace("\n", " ")
        return f"CustomInstructions(source={self._source!r}, preview={preview!r})"


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  EMBEDDER  █████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class Embedder:
    """
    Thin wrapper around sentence-transformers (or any callable).
    Falls back to a simple TF-IDF bag-of-words if sentence-transformers
    is not installed — so RAG still works without any ML dependency.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model      = None
        self._mode       = "unloaded"

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._mode  = "sentence_transformers"
            print(f"[Embedder] Loaded {self._model_name}")
        except ImportError:
            # Fallback: hash-based pseudo-embedding (for testing / no ML dep)
            self._mode = "hash_fallback"
            print("[Embedder] sentence-transformers not found — using hash fallback "
                  "(install sentence-transformers for real semantic search)")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts. Returns list of float vectors."""
        self._lazy_load()
        if self._mode == "sentence_transformers":
            return self._model.encode(texts, show_progress_bar=False).tolist()
        # Hash fallback: 64-dim pseudo-embedding from sha256
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vecs.append([((b / 255.0) - 0.5) * 2 for b in h[:64]])
        return vecs

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  VECTOR BACKENDS  ██████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class _BaseVectorBackend:
    """Interface all vector backends implement."""

    def add(self, ids: List[str], vectors: List[List[float]],
            texts: List[str], metadatas: Optional[List[dict]] = None): ...

    def query(self, vector: List[float], top_k: int = 5
              ) -> List[Tuple[str, float, dict]]: ...
    # Returns: list of (text, score, metadata)

    def count(self) -> int: ...


class ChromaBackend(_BaseVectorBackend):
    """Chroma vector store backend (pip install chromadb)."""

    def __init__(self, persist_dir: str, collection: str = "leo_knowledge"):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col    = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[ChromaBackend] Collection '{collection}' | {self._col.count()} docs")

    def add(self, ids, vectors, texts, metadatas=None):
        metas = metadatas or [{} for _ in ids]
        self._col.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    def query(self, vector, top_k=5):
        r = self._col.query(query_embeddings=[vector], n_results=top_k,
                            include=["documents", "distances", "metadatas"])
        results = []
        for doc, dist, meta in zip(r["documents"][0], r["distances"][0],
                                    r["metadatas"][0]):
            results.append((doc, 1.0 - dist, meta))   # cosine: distance → similarity
        return results

    def count(self): return self._col.count()


class FaissBackend(_BaseVectorBackend):
    """FAISS backend (pip install faiss-cpu)."""

    def __init__(self, index_path: str, dim: int = 384):
        import faiss, pickle
        self._dim   = dim
        self._path  = index_path
        self._texts : List[str]  = []
        self._metas : List[dict] = []
        idx_file  = Path(index_path) / "index.faiss"
        data_file = Path(index_path) / "data.pkl"
        if idx_file.exists() and data_file.exists():
            self._index = faiss.read_index(str(idx_file))
            with open(data_file, "rb") as f:
                self._texts, self._metas = pickle.load(f)
            print(f"[FaissBackend] Loaded {len(self._texts)} docs from {index_path}")
        else:
            self._index = faiss.IndexFlatIP(dim)   # Inner product (cosine after L2 norm)
            print(f"[FaissBackend] New index at {index_path}")

    def _save(self):
        import faiss, pickle
        Path(self._path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(Path(self._path) / "index.faiss"))
        with open(Path(self._path) / "data.pkl", "wb") as f:
            pickle.dump((self._texts, self._metas), f)

    def add(self, ids, vectors, texts, metadatas=None):
        import numpy as np
        vecs  = np.array(vectors, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
        self._index.add(vecs / norms)
        self._texts.extend(texts)
        self._metas.extend(metadatas or [{} for _ in texts])
        self._save()

    def query(self, vector, top_k=5):
        import numpy as np
        v = np.array([vector], dtype="float32")
        v = v / np.linalg.norm(v).clip(min=1e-8)
        scores, idxs = self._index.search(v, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= 0:
                results.append((self._texts[idx], float(score), self._metas[idx]))
        return results

    def count(self): return self._index.ntotal


class InMemoryBackend(_BaseVectorBackend):
    """
    Pure Python in-memory backend — no dependencies.
    Good for small knowledge bases (<10k chunks) or quick prototyping.
    Cosine similarity via dot product after L2 normalisation.
    """

    def __init__(self):
        self._vecs  : List[List[float]] = []
        self._texts : List[str]         = []
        self._metas : List[dict]        = []

    def _norm(self, v):
        mag = sum(x*x for x in v) ** 0.5
        return [x / (mag + 1e-8) for x in v]

    def add(self, ids, vectors, texts, metadatas=None):
        for v, t, m in zip(vectors, texts, (metadatas or [{} for _ in ids])):
            self._vecs.append(self._norm(v))
            self._texts.append(t)
            self._metas.append(m)

    def query(self, vector, top_k=5):
        qv = self._norm(vector)
        scores = [sum(a*b for a, b in zip(qv, dv)) for dv in self._vecs]
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        return [(self._texts[i], s, self._metas[i]) for i, s in ranked]

    def count(self): return len(self._texts)


def _build_backend(backend: str, index_path: str,
                   collection: str) -> _BaseVectorBackend:
    """Factory: return the right vector backend based on config."""
    if backend == "chroma":
        return ChromaBackend(index_path, collection)
    if backend == "faiss":
        return FaissBackend(index_path)
    if backend == "memory":
        return InMemoryBackend()
    # Pinecone / Weaviate / Qdrant stubs — add your API credentials and uncomment
    if backend == "pinecone":
        raise NotImplementedError(
            "Pinecone backend: set api_key and api_url in leo_config.yaml, "
            "then implement PineconeBackend in leo_rag.py")
    raise ValueError(f"Unknown RAG backend: {backend!r}. "
                     f"Supported: chroma, faiss, memory, pinecone")


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  DOCUMENT INDEXER  █████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class DocumentIndexer:
    """
    Offline pipeline: read files from source_dir → chunk → embed → store.

    Supported file types: pdf, txt, md, html, docx, json, csv
    Call .index_all() once, then use RAGManager for queries.
    Tracks indexed file hashes to skip unchanged files on re-run.
    """

    def __init__(self, source_dir: str, backend: _BaseVectorBackend,
                 embedder: Embedder, chunk_size: int = 512,
                 chunk_overlap: int = 64):
        self._src      = Path(source_dir)
        self._backend  = backend
        self._embedder = embedder
        self._chunk    = chunk_size
        self._overlap  = chunk_overlap
        self._manifest = {}   # path → content_hash (skip unchanged files)

    def _read_file(self, path: Path) -> str:
        """Extract raw text from a file. Add more extractors as needed."""
        ext = path.suffix.lower()
        if ext in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="replace")
        if ext == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return "\n".join(str(item) for item in data)
            return json.dumps(data, indent=2)
        if ext == ".csv":
            return path.read_text(encoding="utf-8", errors="replace")
        if ext == ".html":
            text = path.read_text(encoding="utf-8", errors="replace")
            # Strip HTML tags (basic — install beautifulsoup4 for better parsing)
            try:
                from html.parser import HTMLParser
                class _S(HTMLParser):
                    def __init__(self): super().__init__(); self.parts = []
                    def handle_data(self, d): self.parts.append(d)
                s = _S(); s.feed(text)
                return " ".join(s.parts)
            except Exception:
                return text
        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                print(f"[DocumentIndexer] Install pypdf to index PDFs: pip install pypdf")
                return ""
        if ext == ".docx":
            try:
                import docx
                doc = docx.Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                print(f"[DocumentIndexer] Install python-docx to index .docx: pip install python-docx")
                return ""
        return ""

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping word-level chunks."""
        words  = text.split()
        chunks = []
        step   = max(1, self._chunk - self._overlap)
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i: i + self._chunk]))
        return [c for c in chunks if len(c.split()) >= 10]   # Filter tiny chunks

    def index_all(self, force: bool = False):
        """Index all supported files in source_dir. Skips unchanged files."""
        if not self._src.exists():
            print(f"[DocumentIndexer] Source dir not found: {self._src}")
            return

        supported = {".pdf", ".txt", ".md", ".html", ".docx", ".json", ".csv"}
        files     = [f for f in self._src.rglob("*") if f.suffix.lower() in supported]
        print(f"[DocumentIndexer] Found {len(files)} files in {self._src}")

        total_chunks = 0
        for path in files:
            content_hash = hashlib.md5(path.read_bytes()).hexdigest()
            if not force and self._manifest.get(str(path)) == content_hash:
                continue   # Unchanged — skip

            text   = self._read_file(path)
            if not text.strip():
                continue

            chunks = self._chunk_text(text)
            if not chunks:
                continue

            vectors = self._embedder.embed(chunks)
            ids     = [f"{path.stem}_{i}" for i in range(len(chunks))]
            metas   = [{"source": str(path), "chunk": i} for i in range(len(chunks))]
            self._backend.add(ids, vectors, chunks, metas)
            self._manifest[str(path)] = content_hash
            total_chunks += len(chunks)
            print(f"  Indexed: {path.name} ({len(chunks)} chunks)")

        print(f"[DocumentIndexer] Done. Total chunks in store: {self._backend.count()}")


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  RAG MANAGER  ██████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class RAGManager:
    """
    Retrieval-Augmented Generation for Leo.

    At inference: query → embed → retrieve top-k chunks → format as context block.
    Leo sees retrieved context inside <|retrieved|>...</|retrieved|> tags
    so it knows it's external knowledge, not its own memory.

    These tags are handled here at the prompt level — they are NOT trained
    vocabulary tokens, so no model changes are needed. Leo will treat the block
    as additional context because it reads like a structured system section.
    """

    RETRIEVE_OPEN  = "\n<|retrieved_context|>\n"
    RETRIEVE_CLOSE = "\n<|/retrieved_context|>\n"

    def __init__(self, backend: _BaseVectorBackend, embedder: Embedder,
                 top_k: int = 5, max_tokens: int = 2048,
                 similarity_threshold: float = 0.3):
        self._backend   = backend
        self._embedder  = embedder
        self._top_k     = top_k
        self._max_tok   = max_tokens
        self._threshold = similarity_threshold

    def retrieve(self, query: str) -> List[Tuple[str, float, dict]]:
        """Embed query and retrieve top-k relevant chunks above threshold."""
        vec     = self._embedder.embed_one(query)
        results = self._backend.query(vec, self._top_k)
        return [(text, score, meta) for text, score, meta in results
                if score >= self._threshold]

    def format_context(self, results: List[Tuple[str, float, dict]]) -> str:
        """Format retrieved chunks into a context block for the prompt."""
        if not results:
            return ""
        parts  = []
        budget = self._max_tok * 4   # rough chars ≈ tokens × 4
        used   = 0
        for i, (text, score, meta) in enumerate(results, 1):
            src   = meta.get("source", "unknown")
            chunk = f"[{i}] (source: {Path(src).name}, relevance: {score:.2f})\n{text}"
            if used + len(chunk) > budget:
                break
            parts.append(chunk)
            used += len(chunk)
        body = "\n\n".join(parts)
        return f"{self.RETRIEVE_OPEN}{body}{self.RETRIEVE_CLOSE}"

    def augment(self, query: str) -> str:
        """Return formatted context string for injection into the prompt."""
        results = self.retrieve(query)
        return self.format_context(results)

    def add_text(self, text: str, metadata: Optional[dict] = None):
        """Convenience: add a single text chunk to the store at runtime."""
        vec = self._embedder.embed_one(text)
        uid = hashlib.md5(text.encode()).hexdigest()[:12]
        self._backend.add([uid], [vec], [text], [metadata or {}])

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Add multiple text chunks at runtime."""
        vecs  = self._embedder.embed(texts)
        ids   = [hashlib.md5(t.encode()).hexdigest()[:12] for t in texts]
        metas = metadatas or [{} for _ in texts]
        self._backend.add(ids, vecs, texts, metas)


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  TOOL REGISTRY  ████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolSchema:
    """JSON-schema description of one callable tool (OpenAI function-calling style)."""
    name:        str
    description: str
    parameters:  Dict[str, Any]   # JSON Schema object
    handler:     Optional[Callable] = field(default=None, repr=False)


class ToolRegistry:
    """
    Registry for custom Python functions and MCP server stubs.

    Leo's ACGI gate emits <|tool_call|>{"name": ..., "args": ...}<|/tool_call|>
    at inference. This registry intercepts that output, routes to the right
    handler, and returns the result as <|tool_result|>...<|/tool_result|>.

    Built-in stubs (return NotImplemented until you connect them):
      • web_search   — query a search engine
      • code_exec    — run Python in a sandbox
      • file_io      — read/write files
      • calculator   — evaluate math expressions

    Register your own with @registry.tool(...) or registry.register(...).
    """

    def __init__(self):
        self._tools: Dict[str, ToolSchema] = {}
        self._register_builtins()

    # ── Decorator interface ───────────────────────────────────────────────────

    def tool(self, name: str, description: str,
             parameters: Optional[Dict] = None):
        """Decorator to register a Python function as a Leo tool."""
        def decorator(fn: Callable) -> Callable:
            self.register(ToolSchema(
                name        = name,
                description = description,
                parameters  = parameters or {"type": "object", "properties": {}},
                handler     = fn,
            ))
            return fn
        return decorator

    def register(self, schema: ToolSchema):
        self._tools[schema.name] = schema
        print(f"[ToolRegistry] Registered tool: {schema.name!r}")

    # ── Execution ─────────────────────────────────────────────────────────────

    def call(self, name: str, args: Dict[str, Any]) -> str:
        """
        Call a registered tool. Returns its string output.
        Wraps in <|tool_result|> tags for Leo to consume.
        """
        if name not in self._tools:
            return self._wrap_result(f"Error: tool '{name}' not registered.", name)
        handler = self._tools[name].handler
        if handler is None:
            return self._wrap_result(
                f"Tool '{name}' is registered but has no handler connected.\n"
                f"Connect it in leo_rag.py or pass a handler when calling register().",
                name,
            )
        try:
            result = handler(**args)
            return self._wrap_result(str(result), name)
        except Exception as e:
            return self._wrap_result(f"Tool error: {e}", name)

    def parse_and_call(self, tool_call_text: str) -> str:
        """
        Parse Leo's raw tool-call output and execute it.
        tool_call_text: raw string between <|tool_call|> and <|/tool_call|>
        """
        try:
            payload = json.loads(tool_call_text.strip())
            name    = payload.get("name", "")
            args    = payload.get("args", payload.get("arguments", {}))
            return self.call(name, args)
        except json.JSONDecodeError as e:
            return self._wrap_result(f"Malformed tool call JSON: {e}", "parse_error")

    @staticmethod
    def _wrap_result(result: str, name: str) -> str:
        return f"<|tool_result|>\n[{name}]: {result}\n<|/tool_result|>"

    def list_tools(self) -> List[Dict]:
        """Return all registered tool schemas (for injecting into Leo's context)."""
        return [
            {"name": t.name, "description": t.description,
             "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def tools_as_context(self) -> str:
        """Format available tools as a system context block for Leo."""
        if not self._tools:
            return ""
        lines = ["Available tools (call via <|tool_call|>{...}<|/tool_call|>):"]
        for t in self._tools.values():
            lines.append(f"  • {t.name}: {t.description}")
        return "<|system|>\n" + "\n".join(lines) + "\n<|/system|>\n"

    # ── Built-in stubs ────────────────────────────────────────────────────────

    def _register_builtins(self):
        """Register placeholder stubs for all ACGI tool classes."""

        # ── web_search ────────────────────────────────────────────────────────
        @self.tool(
            name="web_search",
            description="Search the web for current information. Returns top results.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n":     {"type": "integer", "description": "Number of results", "default": 3},
                },
                "required": ["query"],
            },
        )
        def web_search(query: str, n: int = 3) -> str:
            # Stub — connect to DuckDuckGo, SerpAPI, Bing, etc.
            # Example with duckduckgo_search: pip install duckduckgo-search
            try:
                from duckduckgo_search import DDGS
                results = list(DDGS().text(query, max_results=n))
                return "\n".join(
                    f"{r['title']}: {r['body']}" for r in results
                )
            except ImportError:
                return (
                    f"[web_search stub] Install duckduckgo-search to enable: "
                    f"pip install duckduckgo-search\n"
                    f"Or replace this handler with your own search API."
                )

        # ── code_exec ─────────────────────────────────────────────────────────
        @self.tool(
            name="code_exec",
            description="Execute Python code in a sandboxed environment and return output.",
            parameters={
                "type": "object",
                "properties": {
                    "code":    {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Max seconds", "default": 10},
                },
                "required": ["code"],
            },
        )
        def code_exec(code: str, timeout: int = 10) -> str:
            # Basic sandbox via exec with captured stdout
            # For production: use subprocess + docker / e2b / modal
            import io, sys, contextlib, signal
            buf = io.StringIO()
            ns  = {}
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile(code, "<leo_exec>", "exec"), ns)   # noqa: S102
                return buf.getvalue() or "(no output)"
            except Exception as e:
                return f"Error: {e}"

        # ── calculator ────────────────────────────────────────────────────────
        @self.tool(
            name="calculator",
            description="Evaluate a math expression. Safer than code_exec for pure arithmetic.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
        )
        def calculator(expression: str) -> str:
            import math as _math
            allowed_names = {k: v for k, v in vars(_math).items()
                             if not k.startswith("_")}
            allowed_names.update({"abs": abs, "round": round, "int": int, "float": float})
            try:
                result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
                return str(result)
            except Exception as e:
                return f"Calculation error: {e}"

        # ── file_io ───────────────────────────────────────────────────────────
        @self.tool(
            name="file_io",
            description="Read or write a local file. Use mode='read' or mode='write'.",
            parameters={
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "mode":    {"type": "string", "enum": ["read", "write"], "default": "read"},
                    "content": {"type": "string", "description": "Content to write (mode=write only)"},
                },
                "required": ["path"],
            },
        )
        def file_io(path: str, mode: str = "read", content: str = "") -> str:
            p = Path(path)
            if mode == "read":
                return p.read_text(encoding="utf-8") if p.exists() else f"File not found: {path}"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written {len(content)} chars to {path}"

        # ── retrieval (connects to RAGManager) ───────────────────────────────
        @self.tool(
            name="retrieval",
            description="Search Leo's knowledge base for relevant information.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        )
        def retrieval(query: str, top_k: int = 3) -> str:
            return f"[retrieval stub] RAGManager not connected. Pass rag_manager to ToolRegistry."
            # Real implementation: return self._rag_manager.augment(query)
            # Connect via ToolRegistry.connect_rag(rag_manager)

        # ── shell (disabled by default — enable intentionally) ────────────────
        @self.tool(
            name="shell",
            description="DISABLED — Run shell commands. Enable explicitly and add sandbox.",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        )
        def shell(command: str) -> str:
            return "[shell] Tool is disabled for safety. Enable in leo_rag.py."

    def connect_rag(self, rag_manager: "RAGManager"):
        """Wire the retrieval tool to a live RAGManager instance."""
        def _retrieval(query: str, top_k: int = 3) -> str:
            results = rag_manager.retrieve(query)[:top_k]
            return rag_manager.format_context(results)
        self._tools["retrieval"].handler = _retrieval


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  MCP SERVER STUB  ██████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class MCPServerStub:
    """
    Stub for a Model Context Protocol server connection.
    Implement .call(tool_name, args) to connect to a real MCP endpoint.

    MCP spec: https://modelcontextprotocol.io
    Typical MCP servers: filesystem, GitHub, Slack, databases, custom APIs.
    """

    def __init__(self, name: str, url: Optional[str] = None,
                 auth: Optional[str] = None):
        self.name = name
        self.url  = url
        self.auth = auth

    def call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Call a tool on this MCP server.
        Replace this stub with a real HTTP/SSE call to self.url.
        """
        if self.url is None:
            return (
                f"[MCP:{self.name}] Not connected. Set url in leo_config.yaml "
                f"external_knowledge.mcp_servers section."
            )
        # Real MCP call (example — adapt to your MCP server's protocol):
        # import requests
        # resp = requests.post(
        #     f"{self.url}/tools/{tool_name}",
        #     json=args,
        #     headers={"Authorization": f"Bearer {self.auth}"} if self.auth else {},
        # )
        # return resp.text
        return f"[MCP:{self.name}/{tool_name}] Stub — implement real HTTP call above."

    def __repr__(self):
        status = "connected" if self.url else "stub"
        return f"MCPServerStub(name={self.name!r}, status={status})"


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  LEO KNOWLEDGE LAYER  ██████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class LeoKnowledgeLayer:
    """
    Top-level integration class. Combines CustomInstructions + RAGManager +
    ToolRegistry into one object that wraps every Leo inference call.

    Usage:
        layer  = LeoKnowledgeLayer.from_config("./leo_config.yaml")
        prompt = layer.augment(user_message)
        output = leo_generate(prompt)
        # If output contains <|tool_call|>...</|tool_call|>:
        output = layer.handle_tool_calls(output)

    All three sub-systems are optional and independently hot-swappable.
    """

    def __init__(
        self,
        instructions : Optional[CustomInstructions] = None,
        rag          : Optional[RAGManager]          = None,
        tools        : Optional[ToolRegistry]         = None,
        mcp_servers  : Optional[List[MCPServerStub]]  = None,
    ):
        self.instructions = instructions or CustomInstructions.empty()
        self.rag          = rag
        self.tools        = tools or ToolRegistry()
        self.mcp_servers  = {s.name: s for s in (mcp_servers or [])}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str = "./leo_config.yaml") -> "LeoKnowledgeLayer":
        """Build a LeoKnowledgeLayer directly from leo_config.yaml."""
        try:
            import yaml
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except ImportError:
            print("[LeoKnowledgeLayer] Install pyyaml: pip install pyyaml")
            return cls()
        except FileNotFoundError:
            print(f"[LeoKnowledgeLayer] Config not found: {config_path}")
            return cls()

        ext = cfg.get("external_knowledge", {})
        if not ext.get("enabled", False):
            print("[LeoKnowledgeLayer] external_knowledge disabled in config — "
                  "returning empty layer. Set enabled: true to activate.")
            return cls()

        # ── Custom instructions ──────────────────────────────────────────────
        instr_cfg = ext.get("custom_instructions", {})
        if instr_cfg.get("enabled", False):
            inline = instr_cfg.get("inline")
            if inline:
                instructions = CustomInstructions.from_string(inline)
            else:
                instructions = CustomInstructions.from_file(
                    instr_cfg.get("path", "./knowledge/instructions.txt"))
        else:
            instructions = CustomInstructions.empty()

        # ── RAG ───────────────────────────────────────────────────────────────
        rag = None
        rag_cfg = ext.get("rag", {})
        if rag_cfg.get("enabled", False):
            embedder = Embedder(rag_cfg.get("embed_model",
                                             "sentence-transformers/all-MiniLM-L6-v2"))
            backend  = _build_backend(
                rag_cfg.get("backend",    "chroma"),
                rag_cfg.get("index_path", "./knowledge/index/"),
                rag_cfg.get("collection", "leo_knowledge"),
            )
            rag = RAGManager(
                backend    = backend,
                embedder   = embedder,
                top_k      = rag_cfg.get("top_k",      5),
                max_tokens = rag_cfg.get("max_tokens", 2048),
            )
            # Auto-index documents if doc store is configured
            doc_cfg = ext.get("document_store", {})
            if doc_cfg.get("enabled", False):
                indexer = DocumentIndexer(
                    source_dir    = doc_cfg.get("source_dir",    "./knowledge/documents/"),
                    backend       = backend,
                    embedder      = embedder,
                    chunk_size    = doc_cfg.get("chunk_size",    512),
                    chunk_overlap = doc_cfg.get("chunk_overlap", 64),
                )
                indexer.index_all()

        # ── Tools ─────────────────────────────────────────────────────────────
        tools = ToolRegistry()

        # Custom function registry from JSON schema file
        fn_cfg = ext.get("custom_functions", {})
        if fn_cfg.get("enabled", False):
            reg_path = fn_cfg.get("registry", "./knowledge/functions.json")
            if Path(reg_path).exists():
                schemas = json.loads(Path(reg_path).read_text())
                for s in schemas:
                    # No handler — tool is defined but not implemented yet
                    tools.register(ToolSchema(
                        name        = s["name"],
                        description = s.get("description", ""),
                        parameters  = s.get("parameters", {}),
                        handler     = None,
                    ))
                print(f"[LeoKnowledgeLayer] Loaded {len(schemas)} custom function schemas")

        if rag:
            tools.connect_rag(rag)

        # ── MCP servers ───────────────────────────────────────────────────────
        mcp_servers = []
        for srv in ext.get("mcp_servers", []):
            if srv.get("enabled", False):
                stub = MCPServerStub(
                    name = srv["name"],
                    url  = srv.get("url"),
                    auth = srv.get("auth"),
                )
                mcp_servers.append(stub)
                print(f"[LeoKnowledgeLayer] MCP server: {stub}")

        return cls(instructions=instructions, rag=rag,
                   tools=tools, mcp_servers=mcp_servers)

    # ── Inference-time augmentation ───────────────────────────────────────────

    def augment(self, user_message: str,
                inject_tools: bool = True) -> str:
        """
        Build the full augmented prompt for Leo.
        Order: system block → custom instructions → tool list → RAG context → user message
        """
        parts = []

        # Custom operator / user instructions
        self.instructions.reload()   # Hot-reload if file changed
        if not self.instructions.is_empty():
            parts.append(self.instructions.as_system_block())

        # Tool manifest (so Leo knows what tools are available)
        if inject_tools and self.tools:
            tool_ctx = self.tools.tools_as_context()
            if tool_ctx:
                parts.append(tool_ctx)

        # RAG context retrieved for this specific query
        if self.rag:
            rag_ctx = self.rag.augment(user_message)
            if rag_ctx:
                parts.append(rag_ctx)

        parts.append(user_message)
        return "\n".join(parts)

    def handle_tool_calls(self, model_output: str) -> str:
        """
        Post-process Leo's output: execute any <|tool_call|>...</|tool_call|> blocks.
        Replaces each tool call with its result in the output string.
        For multi-turn agentic loops, call this in a loop until no tool calls remain.
        """
        import re
        pattern = re.compile(
            r"<\|tool_call\|>(.*?)<\|/tool_call\|>",
            re.DOTALL,
        )

        def _replace(match):
            payload = match.group(1).strip()
            # Check if it's an MCP call: {"mcp": "server_name", "tool": ..., "args": ...}
            try:
                parsed = json.loads(payload)
                if "mcp" in parsed:
                    srv_name = parsed["mcp"]
                    srv      = self.mcp_servers.get(srv_name)
                    if srv:
                        return srv.call(parsed.get("tool", ""), parsed.get("args", {}))
                    return f"<|tool_result|>MCP server '{srv_name}' not registered.<|/tool_result|>"
            except json.JSONDecodeError:
                pass
            # Standard tool registry call
            return self.tools.parse_and_call(payload)

        return pattern.sub(_replace, model_output)

    def has_pending_tool_calls(self, text: str) -> bool:
        """Check if Leo's output contains unresolved tool calls."""
        return "<|tool_call|>" in text and "<|/tool_call|>" in text

    def agentic_loop(self, user_message: str,
                     generate_fn: Callable[[str], str],
                     max_turns: int = 5) -> str:
        """
        Full agentic inference loop:
        1. Augment prompt with instructions + RAG context
        2. Generate Leo's response
        3. Execute any tool calls in the response
        4. Feed tool results back to Leo for the next turn
        5. Repeat until no pending tool calls or max_turns reached

        Args:
            user_message : the user's input
            generate_fn  : fn(prompt: str) -> str — your Leo inference call
            max_turns    : safety limit on tool-call rounds
        Returns:
            Leo's final text response (after all tool calls resolved)
        """
        prompt = self.augment(user_message)
        turns  = 0

        while turns < max_turns:
            response = generate_fn(prompt)
            if not self.has_pending_tool_calls(response):
                return response
            # Execute tool calls and embed results back into the prompt
            resolved = self.handle_tool_calls(response)
            prompt   = resolved   # Leo sees its own prior output + tool results
            turns   += 1

        return response   # Return whatever we have after max_turns

    # ── Management ───────────────────────────────────────────────────────────

    def add_knowledge(self, texts: List[str],
                      metadatas: Optional[List[dict]] = None):
        """Dynamically add knowledge chunks to the RAG store at runtime."""
        if self.rag is None:
            print("[LeoKnowledgeLayer] RAG not configured. Enable it in leo_config.yaml.")
            return
        self.rag.add_texts(texts, metadatas)
        print(f"[LeoKnowledgeLayer] Added {len(texts)} chunks to knowledge store.")

    def set_instructions(self, text: str):
        """Override custom instructions at runtime (no restart needed)."""
        self.instructions = CustomInstructions.from_string(text)

    def status(self) -> Dict[str, Any]:
        """Return a status dict showing what's connected."""
        return {
            "custom_instructions": not self.instructions.is_empty(),
            "rag":                 self.rag is not None,
            "rag_doc_count":       self.rag._backend.count() if self.rag else 0,
            "tools":               list(self.tools._tools.keys()),
            "mcp_servers":         list(self.mcp_servers.keys()),
        }

    def __repr__(self):
        s = self.status()
        return (
            f"LeoKnowledgeLayer("
            f"instructions={s['custom_instructions']}, "
            f"rag={s['rag']} ({s['rag_doc_count']} docs), "
            f"tools={s['tools']}, "
            f"mcp={s['mcp_servers']})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  QUICK-START EXAMPLES  █████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== Leo Aether — Knowledge Layer Quick-Start ===\n")

    # ── Example 1: Custom instructions only ──────────────────────────────────
    layer = LeoKnowledgeLayer(
        instructions=CustomInstructions.from_string(
            "Always respond in a helpful, concise manner.\n"
            "You are deployed as a coding assistant for Python developers.\n"
            "When unsure, say 'I don't know' rather than guessing."
        )
    )
    print("Example 1 — Custom instructions:")
    augmented = layer.augment("How do I reverse a list in Python?")
    print(augmented[:400])
    print()

    # ── Example 2: In-memory RAG with a few manual chunks ────────────────────
    from leo_rag import InMemoryBackend, Embedder, RAGManager, LeoKnowledgeLayer
    backend  = InMemoryBackend()
    embedder = Embedder()
    rag      = RAGManager(backend, embedder, top_k=2)
    rag.add_texts([
        "Leo Aether is a language model built by Unmuted with 3.1B parameters.",
        "Leo uses Epistemic Confidence Tokens to detect and suppress hallucinations.",
        "Training uses 2.1B tokens across FineWeb-Edu, FineMath, The Stack, and Wikipedia.",
    ])
    layer2 = LeoKnowledgeLayer(rag=rag)
    print("Example 2 — RAG retrieval:")
    ctx = layer2.augment("What are Epistemic Confidence Tokens?")
    print(ctx[:600])
    print()

    # ── Example 3: Tool call parsing ─────────────────────────────────────────
    layer3   = LeoKnowledgeLayer()
    fake_out = (
        'The answer is: <|tool_call|>{"name": "calculator", "args": {"expression": "2 ** 32"}}<|/tool_call|>'
    )
    print("Example 3 — Tool call execution:")
    resolved = layer3.handle_tool_calls(fake_out)
    print(resolved)
    print()

    print("Status:", layer3.status())
    print("\nTo connect a real backend, set external_knowledge.enabled: true in leo_config.yaml")
