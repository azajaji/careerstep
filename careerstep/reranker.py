"""Cross-encoder reranker over a first-pass ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class CrossEncoderReranker:
    """Two-text relevance scorer with graceful offline fallback."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def _model(self):
        cached = getattr(self, "_cached_model", None)
        if cached is None:
            try:
                from sentence_transformers import CrossEncoder

                cached = CrossEncoder(self.model_name)
            except Exception as exc:  # noqa: BLE001
                print(f"[reranker] disabled, fallback to identity ({exc!r})")
                cached = "DISABLED"
            object.__setattr__(self, "_cached_model", cached)
        return cached

    def is_enabled(self) -> bool:
        return self._model() != "DISABLED"

    def score(self, query: str, candidates: Sequence[str]) -> List[float]:
        model = self._model()
        if model == "DISABLED" or not candidates:
            return [0.0] * len(candidates)
        pairs = [(query, c) for c in candidates]
        return [float(s) for s in model.predict(pairs, batch_size=16, show_progress_bar=False)]

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Return (candidate_index, score) sorted descending."""
        scores = self.score(query, candidates)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, scores[i]) for i in idx]
