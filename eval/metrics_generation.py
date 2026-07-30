"""Semantic-similarity and diversity metrics."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from careerstep.backends import EmbeddingBackend


def bleu(hypothesis: str, references: Sequence[str]) -> float:
    if not hypothesis or not references:
        return 0.0
    import sacrebleu

    res = sacrebleu.sentence_bleu(hypothesis, list(references))
    return float(res.score / 100.0)


def rouge_l(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference, hypothesis)["rougeL"].fmeasure)


def bertscore_f1(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    if not hypotheses or not references:
        return 0.0
    from bert_score import score

    _, _, f1 = score(
        list(hypotheses),
        list(references),
        lang="en",
        rescale_with_baseline=False,
        verbose=False,
    )
    return float(f1.mean().item())


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
