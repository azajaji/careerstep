"""Retrieval metrics: recall@k, MRR, nDCG@k."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def recall_at_k(ranked_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[int], relevant_ids: set[int]) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def retrieval_summary(
    rankings: list[tuple[list[int], set[int]]],
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> dict:
    """``rankings`` is a list of (ranked_doc_ids, set_of_relevant_doc_ids)."""
    out: dict = {"per_query": [], "macro": {}}
    rr_list, ndcg_list = [], []
    recalls = {k: [] for k in ks}
    for ranked, gold in rankings:
        rr = reciprocal_rank(ranked, gold)
        rr_list.append(rr)
        nd = ndcg_at_k(ranked, gold, k=max(ks))
        ndcg_list.append(nd)
        per = {"mrr": rr, "ndcg": nd}
        for k in ks:
            r = recall_at_k(ranked, gold, k)
            recalls[k].append(r)
            per[f"recall@{k}"] = r
        out["per_query"].append(per)

    out["macro"]["mrr"] = float(np.mean(rr_list)) if rr_list else 0.0
    out["macro"]["ndcg"] = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    for k in ks:
        out["macro"][f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
    return out
