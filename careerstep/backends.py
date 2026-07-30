"""Swappable lexical, embedding, and LLM backends."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np


def _to_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Lexical (TF-IDF + BM25)
# ---------------------------------------------------------------------------


@dataclass
class LexicalBackend:
    """TF-IDF cosine similarity + BM25 retrieval."""

    name: str = "lexical"

    def similarity_matrix(self, queries: Sequence[str], docs: Sequence[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
        all_text = list(queries) + list(docs)
        matrix = vec.fit_transform(all_text)
        q = matrix[: len(queries)]
        d = matrix[len(queries):]
        return cosine_similarity(q, d)

    def rank(self, query: str, docs: Sequence[str], top_k: int = 10) -> List[int]:
        from rank_bm25 import BM25Okapi

        tokenised_corpus = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenised_corpus)
        scores = bm25.get_scores(query.lower().split())
        return list(np.argsort(scores)[::-1][:top_k])


# ---------------------------------------------------------------------------
# Embedding (sentence-transformers)
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingBackend:
    """Local sentence-transformers backend (default: all-MiniLM-L6-v2)."""

    name: str = "embedding"
    model_name: str = field(
        default_factory=lambda: os.environ.get(
            "KHUTWA_EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )

    def _model(self):
        cached = getattr(self, "_cached_model", None)
        if cached is None:
            from sentence_transformers import SentenceTransformer

            cached = SentenceTransformer(self.model_name)
            object.__setattr__(self, "_cached_model", cached)
        return cached

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        embeddings = self._model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _to_array(embeddings)

    def similarity_matrix(self, queries: Sequence[str], docs: Sequence[str]) -> np.ndarray:
        q = self.encode(queries)
        d = self.encode(docs)
        return q @ d.T

    def rank(self, query: str, docs: Sequence[str], top_k: int = 10) -> List[int]:
        sims = self.similarity_matrix([query], docs)[0]
        return list(np.argsort(sims)[::-1][:top_k])


# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible)
# ---------------------------------------------------------------------------


@dataclass
class LLMBackend:
    """OpenAI-compatible chat backend, used only when ``OPENAI_API_KEY`` is set.

    The class falls back to :class:`EmbeddingBackend` for similarity work so
    callers can ask for generation *and* similarity from one object.
    """

    name: str = "llm"
    chat_model: str = field(
        default_factory=lambda: os.environ.get("KHUTWA_CHAT_MODEL", "gpt-4o-mini")
    )
    embed_model: str = field(
        default_factory=lambda: os.environ.get(
            "KHUTWA_LLM_EMBED_MODEL", "text-embedding-3-small"
        )
    )

    def __post_init__(self) -> None:
        self._fallback = EmbeddingBackend()
        self._client = None
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except Exception:
                self._client = None

    def is_live(self) -> bool:
        return self._client is not None

    def chat(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        if not self._client:
            return ""
        resp = self._client.chat.completions.create(
            model=self.chat_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._client:
            return self._fallback.encode(texts)
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        resp = self._client.embeddings.create(model=self.embed_model, input=list(texts))
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-9)
        return vecs

    def similarity_matrix(self, queries: Sequence[str], docs: Sequence[str]) -> np.ndarray:
        q = self.encode(queries)
        d = self.encode(docs)
        return q @ d.T

    def rank(self, query: str, docs: Sequence[str], top_k: int = 10) -> List[int]:
        sims = self.similarity_matrix([query], docs)[0]
        return list(np.argsort(sims)[::-1][:top_k])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(name: Optional[str] = None):
    """Return a backend instance.

    Resolution order:
      1. Explicit ``name`` argument (``lexical`` / ``embedding`` / ``llm``).
      2. ``KHUTWA_BACKEND`` environment variable.
      3. ``llm`` if ``OPENAI_API_KEY`` is set, else ``embedding``.
    """
    name = (name or os.environ.get("KHUTWA_BACKEND") or "").strip().lower()
    if not name:
        name = "llm" if os.environ.get("OPENAI_API_KEY") else "embedding"
    if name == "lexical":
        return LexicalBackend()
    if name == "embedding":
        return EmbeddingBackend()
    if name == "llm":
        backend = LLMBackend()
        if not backend.is_live():
            return EmbeddingBackend()
        return backend
    raise ValueError(f"unknown backend: {name!r}")


__all__ = [
    "LexicalBackend",
    "EmbeddingBackend",
    "LLMBackend",
    "get_backend",
]
