"""Tests for the MELO external benchmark.

Runs under pytest or directly: ``python -m tests.test_melo``.
"""

from __future__ import annotations

import numpy as np

from careerstep.seeding import load_seeds
from eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from experiments.exp12_melo_external import (
    _concept,
    _max_jaccard,
    _paired_delta,
    _to_concepts,
)


def test_concept_strips_language_and_variant():
    assert _concept("C001940_en_003") == "C001940"
    assert _concept("C000001_en_000") == "C000001"


def test_to_concepts_dedupes_and_keeps_first_position():
    ranked = ["C2_en_000", "C2_en_001", "C9_en_004", "C2_en_002", "C5_en_000"]
    assert _to_concepts(ranked, 10) == ["C2", "C9", "C5"]


def test_to_concepts_respects_limit_and_is_lazy():
    def gen():
        for i in range(1000):
            yield f"C{i}_en_000"
        raise AssertionError("consumed past the limit")

    assert _to_concepts(gen(), 3) == ["C0", "C1", "C2"]


def test_max_jaccard():
    # identical -> 1.0
    assert _max_jaccard("chief executive", ["chief executive"]) == 1.0
    # takes the best of several gold labels
    assert _max_jaccard("ceo", ["chief executive officer", "ceo"]) == 1.0
    # disjoint -> 0.0
    assert _max_jaccard("welder", ["chief executive officer"]) == 0.0
    # 1 shared of 3 distinct
    assert abs(_max_jaccard("chief cook", ["chief executive"]) - 1 / 3) < 1e-9


def test_single_gold_concept_metrics_are_hit_based():
    gold = {"C7"}
    assert recall_at_k(["C7", "C1"], gold, 1) == 1.0
    assert recall_at_k(["C1", "C7"], gold, 1) == 0.0
    assert recall_at_k(["C1", "C7"], gold, 5) == 1.0
    assert reciprocal_rank(["C1", "C2", "C7"], gold) == 1 / 3
    assert reciprocal_rank(["C1", "C2"], gold) == 0.0
    assert ndcg_at_k(["C7"], gold, 10) == 1.0
    assert abs(ndcg_at_k(["C1", "C7"], gold, 10) - 1 / np.log2(3)) < 1e-9


def test_paired_bootstrap_is_seed_deterministic():
    a = [1.0] * 30 + [0.0] * 70
    b = [0.0] * 100
    r1 = _paired_delta(a, b, np.random.default_rng(20260101))
    r2 = _paired_delta(a, b, np.random.default_rng(20260101))
    assert r1 == r2
    assert abs(r1["mean_delta"] - 0.30) < 1e-9
    assert r1["excludes_zero"] is True


def test_paired_bootstrap_detects_no_difference():
    same = [1.0, 0.0] * 50
    r = _paired_delta(same, same, np.random.default_rng(20260101))
    assert r["mean_delta"] == 0.0
    assert r["excludes_zero"] is False


def test_bootstrap_seed_is_pinned():
    assert load_seeds()["bootstrap_seed"] == 20260101


def test_melo_has_exactly_one_gold_concept_per_query():
    """The scoring collapses surface forms; that is only valid if each query
    has a single correct occupation. Skipped when the cache is cold."""
    from data import CACHE_DIR

    if not (CACHE_DIR / "melo" / "usa_q_en_c_en" / "annotations.tsv").exists():
        print("  (skipped: MELO not cached)")
        return
    from data.melo import load

    _, _, qrels = load("usa_q_en_c_en")
    assert qrels, "no annotations loaded"
    for qid, gold in qrels.items():
        assert len({_concept(g) for g in gold}) == 1, qid


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
