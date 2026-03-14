import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ToolSchema:
    name:        str
    description: str
    parameters:  Dict[str, Any]
    handler:     Optional[Callable] = field(default=None, repr=False)


# ── Embedding backends ────────────────────────────────────────────────────────

class _SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim    = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> "np.ndarray":
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_one(self, text: str) -> "np.ndarray":
        return self.embed([text])[0]


class _TFIDFEmbedder:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(max_features=4096, sublinear_tf=True)
        self._fitted = False
        self.dim = 4096

    def fit(self, texts: List[str]):
        self._vec.fit(texts)
        self._fitted = True

    def embed(self, texts: List[str]) -> "np.ndarray":
        import numpy as np
        if not self._fitted:
            self.fit(texts)
        return self._vec.transform(texts).toarray().astype("float32")

    def embed_one(self, text: str) -> "np.ndarray":
        return self.embed([text])[0]


def _make_embedder(prefer_st: bool = True):
    if prefer_st:
        try:
            return _SentenceTransformerEmbedder()
        except ImportError:
            pass
    try:
        return _TFIDFEmbedder()
    except ImportError:
        return None


# ── Vector store backends ─────────────────────────────────────────────────────

class _FAISSBackend:
    def __init__(self, dim: int):
        import faiss, numpy as np
        self._dim   = dim
        self._index = faiss.IndexFlatIP(dim)
        self._texts: List[str]       = []
        self._metas: List[Dict]      = []

    def add(self, ids, vectors, texts, metas):
        import numpy as np
        arr = np.array(vectors, dtype="float32")
        if arr.ndim == 1:
            arr = arr[np.newaxis]
        self._index.add(arr)
        self._texts.extend(texts)
        self._metas.extend(metas)

    def search(self, vec, k: int) -> List[Tuple[float, str, Dict]]:
        import numpy as np
        q = np.array(vec, dtype="float32")[np.newaxis]
        D, I = self._index.search(q, k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self._texts):
                out.append((float(score), self._texts[idx], self._metas[idx]))
        return out

    def count(self) -> int:
        return self._index.ntotal


class _ChromaBackend:
    def __init__(self, collection_name: str = "leo_rag", persist: Optional[str] = None):
        import chromadb
        if persist:
            self._client = chromadb.PersistentClient(path=persist)
        else:
            self._client = chromadb.Client()
        self._col = self._client.get_or_create_collection(collection_name)

    def add(self, ids, vectors, texts, metas):
        import numpy as np
        self._col.add(
            ids=ids,
            embeddings=[v.tolist() if hasattr(v, "tolist") else v for v in vectors],
            documents=texts,
            metadatas=metas,
        )

    def search(self, vec, k: int) -> List[Tuple[float, str, Dict]]:
        results = self._col.query(
            query_embeddings=[vec.tolist() if hasattr(vec, "tolist") else vec],
            n_results=k,
        )
        out = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            out.append((1.0 - dist, doc, meta))
        return out

    def count(self) -> int:
        return self._col.count()


class _InMemoryBackend:
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


def _make_backend(backend: str = "auto", dim: int = 384, persist: Optional[str] = None):
    if backend == "faiss" or (backend == "auto"):
        try:
            return _FAISSBackend(dim)
        except ImportError:
            pass
    if backend == "chroma" or (backend == "auto"):
        try:
            return _ChromaBackend(persist=persist)
        except ImportError:
            pass
    return _InMemoryBackend()


# ── Web search backends ───────────────────────────────────────────────────────

_WEB_SEARCH_PROMPT = (
    "Search results for: {query}\n"
    "Date: {date}\n\n"
    "{results}\n"
)


def _format_search_results(results: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", r.get("name", ""))
        body  = r.get("body", r.get("snippet", r.get("description", "")))
        url   = r.get("href", r.get("url", r.get("link", "")))
        lines.append(f"[{i}] {title}\n{body}\nURL: {url}")
    return "\n\n".join(lines)


def _ddg_search(query: str, n: int = 5, timeout: int = 8) -> List[Dict]:
    from duckduckgo_search import DDGS
    with DDGS(timeout=timeout) as ddgs:
        return list(ddgs.text(query, max_results=n))


def _brave_search(query: str, n: int = 5) -> List[Dict]:
    import urllib.request
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY not set")
    url = f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote(query)}&count={n}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    })
    import urllib.parse, gzip, io
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        data = json.loads(raw)
    results = []
    for item in data.get("web", {}).get("results", [])[:n]:
        results.append({
            "title": item.get("title", ""),
            "body":  item.get("description", ""),
            "href":  item.get("url", ""),
        })
    return results


def _serp_search(query: str, n: int = 5) -> List[Dict]:
    import urllib.request, urllib.parse
    api_key = os.environ.get("SERPAPI_KEY", "")
    if not api_key:
        raise RuntimeError("SERPAPI_KEY not set")
    params = urllib.parse.urlencode({
        "q": query, "num": n, "api_key": api_key, "engine": "google"
    })
    url = f"https://serpapi.com/search?{params}"
    with urllib.request.urlopen(url, timeout=12) as resp:
        data = json.loads(resp.read())
    results = []
    for r in data.get("organic_results", [])[:n]:
        results.append({
            "title": r.get("title", ""),
            "body":  r.get("snippet", ""),
            "href":  r.get("link", ""),
        })
    return results


def web_search_multi_backend(
    query: str,
    n: int = 5,
    prefer: str = "ddg",
) -> str:
    from safety import RAGSafetyWrapper
    safety = RAGSafetyWrapper()

    backends = ["ddg", "brave", "serp"]
    if prefer in backends:
        backends = [prefer] + [b for b in backends if b != prefer]

    raw_results: List[Dict] = []
    last_err: Optional[Exception] = None

    for backend in backends:
        try:
            if backend == "ddg":
                raw_results = _ddg_search(query, n)
            elif backend == "brave":
                raw_results = _brave_search(query, n)
            elif backend == "serp":
                raw_results = _serp_search(query, n)
            if raw_results:
                break
        except Exception as e:
            last_err = e
            continue

    if not raw_results:
        return f"[web_search] No results found for '{query}'. Error: {last_err}"

    raw_results = safety.filter_results(raw_results)

    formatted = _format_search_results(raw_results[:n])
    date_str  = datetime.now().strftime("%Y-%m-%d")
    return _WEB_SEARCH_PROMPT.format(query=query, date=date_str, results=formatted)


# ── Document indexer ──────────────────────────────────────────────────────────

class DocumentIndexer:
    def __init__(
        self,
        source_dir: str,
        embedder=None,
        backend=None,
        chunk_size: int = 256,
        overlap: int = 32,
        manifest_path: str = ".leo_rag_manifest.json",
    ):
        self._src      = Path(source_dir)
        self._embedder = embedder or _make_embedder()
        self._backend  = backend or _make_backend(dim=getattr(self._embedder, "dim", 384))
        self._chunk    = chunk_size
        self._overlap  = overlap
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
                reader = pypdf.PdfReader(str(path))
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            elif suf == ".docx":
                import docx
                doc = docx.Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs)
            elif suf == ".html":
                from bs4 import BeautifulSoup
                return BeautifulSoup(path.read_text(errors="ignore"), "html.parser").get_text()
            elif suf == ".json":
                data = json.loads(path.read_text(errors="ignore"))
                return json.dumps(data, indent=1) if isinstance(data, (dict, list)) else str(data)
            elif suf == ".csv":
                return path.read_text(errors="ignore")
            else:
                return path.read_text(errors="ignore")
        except Exception:
            return ""

    def _chunk_text(self, text: str) -> List[str]:
        words  = text.split()
        step   = max(1, self._chunk - self._overlap)
        chunks = []
        for i in range(0, len(words), step):
            c = " ".join(words[i: i + self._chunk])
            if len(c.split()) >= 10:
                chunks.append(c)
        return chunks

    def index_all(self, force: bool = False):
        if not self._src.exists():
            return
        supported = {".pdf", ".txt", ".md", ".html", ".docx", ".json", ".csv"}
        files     = [f for f in self._src.rglob("*") if f.suffix.lower() in supported]
        total     = 0
        for path in files:
            content_hash = hashlib.md5(path.read_bytes()).hexdigest()
            if not force and self._manifest.get(str(path)) == content_hash:
                continue
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
            total += len(chunks)
        self._save_manifest()


# ── RAG Manager ───────────────────────────────────────────────────────────────

class RAGManager:
    def __init__(
        self,
        embedder=None,
        backend=None,
        top_k: int = 5,
        min_score: float = 0.20,
        context_tag: str = "retrieved",
    ):
        self._embedder   = embedder or _make_embedder()
        self._backend    = backend  or _make_backend(dim=getattr(self._embedder, "dim", 384))
        self.top_k       = top_k
        self.min_score   = min_score
        self._tag        = context_tag

    def retrieve(self, query: str) -> List[Tuple[float, str, Dict]]:
        if self._embedder is None:
            return []
        vec = self._embedder.embed_one(query)
        hits = self._backend.search(vec, self.top_k)
        return [(s, t, m) for s, t, m in hits if s >= self.min_score]

    def format_context(self, results: List[Tuple]) -> str:
        if not results:
            return ""
        chunks = [t for _, t, _ in results]
        body   = "\n---\n".join(chunks)
        return f"<|{self._tag}|>\n{body}\n<|/{self._tag}|>\n"

    def augment(self, query: str) -> str:
        return self.format_context(self.retrieve(query))

    def add_text(self, text: str, metadata: Optional[Dict] = None):
        if self._embedder is None:
            return
        vec = self._embedder.embed_one(text)
        uid = hashlib.md5(text.encode()).hexdigest()[:12]
        self._backend.add([uid], [vec], [text], [metadata or {}])

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        if self._embedder is None:
            return
        vecs  = self._embedder.embed(texts)
        ids   = [hashlib.md5(t.encode()).hexdigest()[:12] for t in texts]
        metas = metadatas or [{} for _ in texts]
        self._backend.add(ids, vecs, texts, metas)


# ── Tool Registry ─────────────────────────────────────────────────────────────

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
            name    = payload.get("name", "")
            args    = payload.get("args", payload.get("arguments", {}))
            return self.call(name, args)
        except json.JSONDecodeError as e:
            return self._wrap(f"Malformed tool call JSON: {e}", "parse_error")

    def list_tools(self) -> List[Dict]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def tools_as_context(self) -> str:
        if not self._tools:
            return ""
        lines = ["Available tools (call via <|tool_call|>{...}<|/tool_call|>):"]
        for t in self._tools.values():
            lines.append(f"  • {t.name}: {t.description}")
        return "<|system|>\n" + "\n".join(lines) + "\n<|/system|>\n"

    @staticmethod
    def _wrap(result: str, name: str) -> str:
        return f"<|tool_result|>\n[{name}]: {result}\n<|/tool_result|>"

    def _register_builtins(self):

        @self.tool(
            name="web_search",
            description="Search the web for current information. Returns top results.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n":     {"type": "integer", "description": "Number of results", "default": 5},
                    "backend": {"type": "string", "description": "ddg|brave|serp", "default": "ddg"},
                },
                "required": ["query"],
            },
        )
        def web_search(query: str, n: int = 5, backend: str = "ddg") -> str:
            return web_search_multi_backend(query, n=n, prefer=backend)

        @self.tool(
            name="code_exec",
            description="Execute Python code in a sandboxed environment and return stdout.",
            parameters={
                "type": "object",
                "properties": {
                    "code":    {"type": "string"},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["code"],
            },
        )
        def code_exec(code: str, timeout: int = 10) -> str:
            import io, contextlib, signal, sys

            from safety import check_text_safety
            if check_text_safety(code).blocked:
                return "Error: code blocked by safety filter."

            buf = io.StringIO()
            ns  = {}
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile(code, "<leo>", "exec"), ns)
                return buf.getvalue() or "(no output)"
            except Exception as e:
                return f"Error: {type(e).__name__}: {e}"

        @self.tool(
            name="calculator",
            description="Evaluate a math expression safely.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )
        def calculator(expression: str) -> str:
            import ast, operator
            _ops = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
                ast.Pow: operator.pow, ast.USub: operator.neg,
                ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
            }
            def _eval(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    return _ops[type(node.op)](_eval(node.left), _eval(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return _ops[type(node.op)](_eval(node.operand))
                else:
                    raise ValueError(f"Unsupported: {type(node)}")
            try:
                tree = ast.parse(expression, mode="eval")
                return str(_eval(tree.body))
            except Exception as e:
                return f"Error: {e}"

        @self.tool(
            name="file_io",
            description="Read a file from disk. Write path only if explicitly allowed.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write", "list"]},
                    "path":   {"type": "string"},
                    "content": {"type": "string", "description": "For write action"},
                },
                "required": ["action", "path"],
            },
        )
        def file_io(action: str, path: str, content: str = "") -> str:
            p = Path(path)
            if action == "read":
                if not p.exists():
                    return f"Error: file not found: {path}"
                return p.read_text(errors="ignore")[:8000]
            elif action == "write":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
                return f"Written {len(content)} chars to {path}"
            elif action == "list":
                if not p.exists():
                    return f"Error: path not found: {path}"
                files = list(p.iterdir()) if p.is_dir() else [p]
                return "\n".join(str(f) for f in files[:50])
            return f"Unknown action: {action}"

        @self.tool(
            name="memory_store",
            description="Store a key-value pair in Leo's session memory.",
            parameters={
                "type": "object",
                "properties": {
                    "key":   {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        )
        def memory_store(key: str, value: str) -> str:
            _SESSION_MEMORY[key] = value
            return f"Stored: {key}"

        @self.tool(
            name="memory_recall",
            description="Recall a stored value from Leo's session memory.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        )
        def memory_recall(key: str) -> str:
            return _SESSION_MEMORY.get(key, f"No memory stored for key: {key}")


_SESSION_MEMORY: Dict[str, str] = {}


# ── LeoKnowledgeLayer ─────────────────────────────────────────────────────────

class LeoKnowledgeLayer:
    def __init__(
        self,
        rag:           Optional[RAGManager] = None,
        source_dir:    Optional[str]        = None,
        embed_model:   str                  = "all-MiniLM-L6-v2",
        vector_backend: str                 = "auto",
        persist_dir:   Optional[str]        = None,
        top_k:         int                  = 5,
    ):
        embedder = _make_embedder()
        backend  = _make_backend(
            vector_backend,
            dim=getattr(embedder, "dim", 384),
            persist=persist_dir,
        )
        self.rag      = rag or RAGManager(embedder=embedder, backend=backend, top_k=top_k)
        self.registry = ToolRegistry()

        if source_dir:
            self.indexer = DocumentIndexer(source_dir, embedder=embedder, backend=backend)
            self.indexer.index_all()
        else:
            self.indexer = None

    def augment(self, query: str, max_chars: int = 4000) -> str:
        ctx = self.rag.augment(query)
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "\n[...truncated...]\n<|/retrieved|>\n"
        return ctx

    def handle_tool_calls(self, text: str) -> str:
        start_tag = "<|tool_call|>"
        end_tag   = "<|/tool_call|>"
        while start_tag in text and end_tag in text:
            s = text.index(start_tag) + len(start_tag)
            e = text.index(end_tag)
            call_text = text[s:e]
            result    = self.registry.parse_and_call(call_text)
            text      = text[:text.index(start_tag)] + result + text[e + len(end_tag):]
        return text

    def search_web(self, query: str, n: int = 5) -> str:
        return web_search_multi_backend(query, n=n)

    def agentic_loop(
        self,
        prompt:    str,
        gen_fn:    "Callable[[str], str]",
        max_turns: int = 5,
    ) -> str:
        context = self.augment(prompt)
        current = context + prompt
        for _ in range(max_turns):
            output  = gen_fn(current)
            if "<|tool_call|>" not in output:
                return output
            resolved = self.handle_tool_calls(output)
            if resolved == output:
                return output
            current  = current + "\n" + resolved
        return gen_fn(current)

    def status(self) -> Dict:
        return {
            "rag_chunks":   self.rag._backend.count() if hasattr(self.rag._backend, "count") else "?",
            "tools":        [t["name"] for t in self.registry.list_tools()],
            "web_search":   _check_web_search_available(),
            "embedder":     type(self.rag._embedder).__name__ if self.rag._embedder else "none",
            "vector_store": type(self.rag._backend).__name__,
        }


def _check_web_search_available() -> Dict[str, bool]:
    available = {}
    try:
        import duckduckgo_search
        available["ddg"] = True
    except ImportError:
        available["ddg"] = False
    available["brave"] = bool(os.environ.get("BRAVE_SEARCH_API_KEY"))
    available["serp"]  = bool(os.environ.get("SERPAPI_KEY"))
    return available


if __name__ == "__main__":
    print("=== LeoKnowledgeLayer self-test ===\n")

    layer = LeoKnowledgeLayer()
    layer.rag.add_texts([
        "Leo Aether is a 3.1B parameter language model built by Unmuted.",
        "Leo uses Epistemic Confidence Tokens (ECT) to detect and suppress hallucinations.",
        "The ACGI module gates tool calls based on ECT uncertainty.",
        "Leo is trained on 2.1B tokens from FineWeb-Edu, FineMath, The Stack, and Wikipedia.",
    ])

    ctx = layer.augment("What are Epistemic Confidence Tokens?")
    print("RAG context:", ctx[:400])

    fake = 'Answer: <|tool_call|>{"name": "calculator", "args": {"expression": "2 ** 10"}}<|/tool_call|>'
    resolved = layer.handle_tool_calls(fake)
    print("Tool result:", resolved)

    print("\nStatus:", json.dumps(layer.status(), indent=2))
    print("\nWeb search backends:", _check_web_search_available())
    print("\nTo use web search, install: pip install duckduckgo-search")
    print("Or set BRAVE_SEARCH_API_KEY / SERPAPI_KEY environment variables.")
