"""Semantic-similarity and diversity metrics."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from careerstep.backends import EmbeddingBackend


def semantic_similarity(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    backend=None,
) -> float:
    if not hypotheses or not references:
        return 0.0
    backend = backend or EmbeddingBackend()
    h = backend.encode(list(hypotheses))
    r = backend.encode(list(references))
    sims = h @ r.T
    # Best-match similarity per hypothesis, averaged.
    return float(sims.max(axis=1).mean())


# ---- roadmap-specific composite metrics ----------------------------------


def list_overlap(predicted: Sequence[str], gold: Sequence[str]) -> float:
    p, g = set(predicted), set(gold)
    if not p and not g:
        return 0.0
    return len(p & g) / max(1, len(p | g))


def coverage(predicted_items: Sequence[str], missing_skills: Sequence[str]) -> float:
    if not missing_skills:
        return 0.0
    hits = 0
    pred_text = " ".join(predicted_items).lower()
    for s in missing_skills:
        if s.lower() in pred_text:
            hits += 1
    return hits / len(missing_skills)


def diversity(predicted_items: Sequence[str], *, backend=None) -> float:
    if len(predicted_items) < 2:
        return 1.0
    backend = backend or EmbeddingBackend()
    emb = backend.encode(list(predicted_items))
    sims = emb @ emb.T
    np.fill_diagonal(sims, 0.0)
    n = len(predicted_items)
    mean_sim = sims.sum() / (n * (n - 1))
    return float(max(0.0, 1.0 - mean_sim))
