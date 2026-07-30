"""Feedback coverage and actionability metrics."""

from __future__ import annotations

import re
from typing import Sequence

from careerstep.backends import EmbeddingBackend
from careerstep.feedback import ACTION_VERB_RE


def feedback_coverage(
    predicted_bullets: Sequence[str],
    reference_bullets: Sequence[str],
    *,
    threshold: float = 0.6,
    backend=None,
) -> float:
    if not reference_bullets:
        return 0.0
    if not predicted_bullets:
        return 0.0
    backend = backend or EmbeddingBackend()
    p = backend.encode(list(predicted_bullets))
    r = backend.encode(list(reference_bullets))
    sims = r @ p.T  # for each ref, max similarity to any predicted bullet
    matched = (sims.max(axis=1) >= threshold).sum()
    return float(matched / len(reference_bullets))


def actionability(predicted_bullets: Sequence[str]) -> float:
    if not predicted_bullets:
        return 0.0
    hits = 0
    for b in predicted_bullets:
        if ACTION_VERB_RE.search(b) and len(b.split()) >= 3:
            hits += 1
    return hits / len(predicted_bullets)


_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z+#.]{2,}\b")

# Heuristic stop-list of capitalised words that are not real entities
# (common English, generic CV vocabulary, structural section names).
# Bullets that mention only these are NOT flagged as hallucinated.
_ENTITY_STOPLIST = {
    "CV", "AI", "ATS", "Add", "Include", "Move", "Remove", "Practice",
    "Rewrite", "Expand", "Reduce", "Highlight", "Clarify", "Emphasize",
    "Reorder", "Skills", "Experience", "Education", "Summary", "Projects",
    "Certifications", "Publications", "Awards", "Objective", "Section",
    "STAR", "PDF", "DOCX", "URL", "Khutwa", "Tell", "Walk", "Describe",
    "Explain", "How", "What", "When", "Why", "The", "This", "That", "These",
    "Use", "Avoid", "English", "Arabic",
}


def hallucination_rate(predicted_bullets: Sequence[str], source_text: str) -> float:
    """Heuristic ungrounded-entity rate (higher = more hallucinations).

    A bullet is flagged when it mentions a capitalised token that is
    (a) not in the heuristic stop-list of common English / structural
    words and (b) does not appear in the source text. This is a coarse
    proxy for grounding; it under-counts subtler hallucinations and
    over-counts proper-noun-like words. The metric is reported alongside
    its complement (groundedness = 1 - rate) for transparency.
    """
    if not predicted_bullets:
        return 0.0
    src_low = source_text.lower()
    bad = 0
    for b in predicted_bullets:
        entities = [e for e in _ENTITY_RE.findall(b) if e not in _ENTITY_STOPLIST]
        for e in entities:
            if e.lower() not in src_low:
                bad += 1
                break
    return bad / len(predicted_bullets)


def feedback_summary(
    cases: Sequence[tuple[Sequence[str], Sequence[str], str]],
    *,
    backend=None,
) -> dict:
    """``cases`` is a list of (predicted_bullets, reference_bullets, source_text)."""
    backend = backend or EmbeddingBackend()
    cov, act, hal = [], [], []
    for pred, ref, src in cases:
        cov.append(feedback_coverage(pred, ref, backend=backend))
        act.append(actionability(pred))
        hal.append(hallucination_rate(pred, src))
    n = max(len(cases), 1)
    mean_hal = sum(hal) / n
    return {
        "coverage": sum(cov) / n,
        "actionability": sum(act) / n,
        "hallucination_rate": mean_hal,
        "groundedness": 1.0 - mean_hal,
        "per_case": [
            {"coverage": c, "actionability": a, "hallucination": h}
            for c, a, h in zip(cov, act, hal)
        ],
    }
