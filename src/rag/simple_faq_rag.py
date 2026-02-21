"""Tiny in-memory RAG helper for the support FAQ.

This module provides simple functions to load a Markdown FAQ, chunk it,
build an in-memory embedding index (list of (text, vector)), and retrieve
the top-k relevant chunks for a query using cosine similarity.

Notes:
- Uses sentence-transformers `all-MiniLM-L6-v2` for embeddings.
- Keeps everything in-memory and lazy-loads the model and index on first use.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import os

VECTOR_DB: List[Tuple[str, List[float]]] = []
_MODEL = None
_LOADED_FAQ_TEXT: Optional[str] = None


def load_faq_markdown(path: str = "data/support_faq.md") -> str:
    """Read the FAQ markdown file and return its full text."""
    abspath = os.path.abspath(path)
    with open(abspath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_faq(text: str) -> List[str]:
    """Split the FAQ into chunks by headings and blank lines.

    Each chunk contains a heading and its bullet points so it's a short
    paragraph suitable to include in prompts.
    """
    chunks: List[str] = []
    current: List[str] = []

    for line in text.splitlines():
        s = line.strip()
        if not s:
            # blank line: end of current chunk
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            continue

        # Treat heading lines (start with #) as start of new chunk
        if s.startswith("#"):
            if current:
                chunks.append("\n".join(current).strip())
            current = [s.lstrip("# ").strip()]
        else:
            current.append(s)

    if current:
        chunks.append("\n".join(current).strip())

    # Filter out any tiny chunks
    return [c for c in chunks if len(c) > 10]


def _ensure_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required for embeddings. Install: pip install sentence-transformers"
            ) from exc

        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def build_faq_index(chunks: List[str]) -> None:
    """Embed each chunk and store (text, vector) tuples in module-level VECTOR_DB."""
    global VECTOR_DB
    model = _ensure_model()
    # Compute embeddings in one batch
    embeddings = model.encode(chunks, convert_to_numpy=True)
    VECTOR_DB = []
    for text, vec in zip(chunks, embeddings):
        VECTOR_DB.append((text, vec.tolist()))


def _cosine_sim(a, b) -> float:
    import math

    # both a and b are lists or numpy arrays
    try:
        import numpy as np

        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        if a.size == 0 or b.size == 0:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float((a @ b) / denom)
    except Exception:
        # fallback
        dot = sum(x * y for x, y in zip(a, b))
        norma = math.sqrt(sum(x * x for x in a))
        normb = math.sqrt(sum(y * y for y in b))
        if norma == 0 or normb == 0:
            return 0.0
        return dot / (norma * normb)


def retrieve_faq_chunks(query: str, k: int = 3) -> List[str]:
    """Return the top-k chunk texts most similar to the query."""
    if not VECTOR_DB:
        return []

    model = _ensure_model()
    q_vec = model.encode([query], convert_to_numpy=True)[0]

    # Score each chunk
    scored: List[Tuple[float, str]] = []
    for text, vec in VECTOR_DB:
        score = _cosine_sim(q_vec, vec)
        scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [t for _, t in scored[:k]]
    return top


def get_rag_context(question: str, k: int = 3) -> str:
    """High-level helper that lazy-loads the FAQ, builds the index once,
    retrieves the top-k chunks, and returns them joined as a string.
    """
    global _LOADED_FAQ_TEXT

    if not VECTOR_DB:
        # load FAQ and build index
        if _LOADED_FAQ_TEXT is None:
            _LOADED_FAQ_TEXT = load_faq_markdown()
        chunks = chunk_faq(_LOADED_FAQ_TEXT)
        build_faq_index(chunks)

    top_chunks = retrieve_faq_chunks(question, k=k)
    # Join with two newlines to keep chunks distinct in prompts
    return "\n\n".join(top_chunks)
