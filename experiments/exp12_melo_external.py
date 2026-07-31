"""External occupation linking on the MELO benchmark (usa_q_en_c_en).

Every other measurement in this project runs on corpora the study team built.
MELO is externally constructed and externally annotated, and it uses the
retrieval formulation the CV/JD aligner already implements: rank a query
against a fixed catalogue with known relevance labels. The three scorers run
zero-shot, with no training or tuning on MELO.

Scope: this evaluates occupation entity linking and catalogue retrieval. It
does not evaluate the questionnaire scoring, the work-value projection, the
composite RoleSuitability score, or recommendation effectiveness.

Scoring is at concept level. Each MELO query has exactly one correct ESCO
occupation; the several annotated corpus rows are surface forms of it
("chief executive officer", "CEO", "chairman"). Ranked element lists are
therefore collapsed to first occurrence per concept before scoring, so
Recall@1 is "the top-ranked occupation is the right one" rather than a
fraction of interchangeable labels.

Overlap quartiles are defined before scoring: for each query, the maximum
token Jaccard against any relevant label, split at its own quartiles.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np

from careerstep.backends import EmbeddingBackend
from careerstep.reranker import CrossEncoderReranker
from careerstep.seeding import load_seeds, set_global_seeds
from data.melo import load
from eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from experiments._io import print_header, save_report

CONFIG = "usa_q_en_c_en"
RERANK_DEPTH = 100
TOP_K = 10
N_RESAMPLES = 10_000


def _concept(corpus_id: str) -> str:
    """C001940_en_003 -> C001940 (the ESCO occupation, not the surface form)."""
    return corpus_id.split("_", 1)[0]


def _to_concepts(ranked_ids: Iterable[str], limit: int) -> List[str]:
    """Collapse an element ranking to distinct concepts, keeping first position."""
    out: List[str] = []
    seen: Set[str] = set()
    for cid in ranked_ids:
        c = _concept(cid)
        if c not in seen:
            seen.add(c)
            out.append(c)
            if len(out) >= limit:
                break
    return out


def _tokens(text: str) -> Set[str]:
    return {t for t in text.lower().split() if t}


def _max_jaccard(query: str, gold_texts: Sequence[str]) -> float:
    q = _tokens(query)
    best = 0.0
    for g in gold_texts:
        t = _tokens(g)
        if q or t:
            best = max(best, len(q & t) / max(1, len(q | t)))
    return best


def _peak_mem_mb() -> float:
    """Peak resident set of this process, in MB. Stdlib only."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        # restype must be declared: the pseudo-handle is truncated otherwise.
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]

        c = _Counters()
        c.cb = ctypes.sizeof(c)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                      ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 1e6
        return float("nan")
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def _per_query(ranked: Sequence[str], gold: Set[str]) -> Dict[str, float]:
    return {
        "recall@1": recall_at_k(ranked, gold, 1),
        "recall@5": recall_at_k(ranked, gold, 5),
        "mrr": reciprocal_rank(ranked, gold),
        f"ndcg@{TOP_K}": ndcg_at_k(ranked, gold, k=TOP_K),
        f"zero_hit@{TOP_K}": 0.0 if any(d in gold for d in ranked[:TOP_K]) else 1.0,
    }


def _boot_ci(values: Sequence[float], rng: np.random.Generator) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    idx = rng.integers(0, v.size, size=(N_RESAMPLES, v.size))
    means = v[idx].mean(axis=1)
    return {"mean": float(v.mean()),
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5))}


def _paired_delta(a: Sequence[float], b: Sequence[float],
                  rng: np.random.Generator) -> Dict[str, float]:
    """Paired bootstrap on the per-query difference a - b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = rng.integers(0, d.size, size=(N_RESAMPLES, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"mean_delta": float(d.mean()), "ci95_low": lo, "ci95_high": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def run() -> dict:
    t_start = time.perf_counter()
    set_global_seeds()
    seeds = load_seeds()
    rng = np.random.default_rng(seeds["bootstrap_seed"])

    queries, corpus, qrels = load(CONFIG)
    qids = [q for q in queries if q in qrels]
    cids = list(corpus)
    ctexts = [corpus[c] for c in cids]
    print(f"  {CONFIG}: {len(qids)} queries, {len(cids)} corpus elements")

    # pre-specified overlap quartiles, computed before any scoring
    overlap = np.array([_max_jaccard(queries[q],
                                     [corpus[c] for c in qrels[q] if c in corpus])
                        for q in qids])
    edges = np.quantile(overlap, [0.25, 0.5, 0.75])
    quartile = np.digitize(overlap, edges)

    from rank_bm25 import BM25Okapi
    t0 = time.perf_counter()
    bm25 = BM25Okapi([c.lower().split() for c in ctexts])
    t_index = time.perf_counter() - t0

    emb = EmbeddingBackend()
    t0 = time.perf_counter()
    cvecs = emb.encode(ctexts)
    t_encode_corpus = time.perf_counter() - t0
    qvecs = emb.encode([queries[q] for q in qids])

    rer = CrossEncoderReranker()
    rows: List[Dict] = []
    cand_recall_100: List[float] = []
    t_bm25 = t_bi = t_rer = 0.0

    for i, qid in enumerate(qids):
        gold_ids = qrels[qid]
        gold = {_concept(g) for g in gold_ids}
        qtext = queries[qid]

        t0 = time.perf_counter()
        s = bm25.get_scores(qtext.lower().split())
        bm_order = np.argsort(-s)
        t_bm25 += time.perf_counter() - t0

        t0 = time.perf_counter()
        sims = cvecs @ qvecs[i]
        d_order = np.argsort(-sims)
        t_bi += time.perf_counter() - t0

        cand = [int(j) for j in d_order[:RERANK_DEPTH]]
        cand_recall_100.append(
            1.0 if any(cids[j] in gold_ids for j in cand) else 0.0)

        t0 = time.perf_counter()
        if rer.is_enabled() and len(cand) > 1:
            scored = rer.rerank(qtext, [ctexts[j] for j in cand], top_k=len(cand))
            reordered = [cand[k] for k, _ in scored]
            rr_scores = [float(v) for _, v in scored][:TOP_K]
        else:
            reordered = cand
            rr_scores = []
        t_rer += time.perf_counter() - t0

        # collapse surface forms to occupations before scoring
        bm_c = _to_concepts((cids[j] for j in bm_order), TOP_K)
        bi_c = _to_concepts((cids[j] for j in d_order), TOP_K)
        rr_c = _to_concepts([cids[j] for j in reordered], TOP_K)

        rows.append({
            "query_id": qid,
            "query": qtext,
            "gold_concept": sorted(gold)[0],
            "n_gold_labels": len(gold_ids),
            "max_jaccard": float(overlap[i]),
            "overlap_quartile": int(quartile[i]) + 1,
            "bm25": {"ranked": bm_c, **_per_query(bm_c, gold)},
            "bi_encoder": {"ranked": bi_c,
                           "scores": [float(sims[j]) for j in d_order[:TOP_K]],
                           **_per_query(bi_c, gold)},
            "bi_plus_reranker": {"ranked": rr_c, "scores": rr_scores,
                                 **_per_query(rr_c, gold)},
        })

    metrics = [k for k in rows[0]["bm25"] if k != "ranked"]
    methods = ("bm25", "bi_encoder", "bi_plus_reranker")

    summary: Dict[str, Dict] = {}
    for m in methods:
        summary[m] = {k: _boot_ci([r[m][k] for r in rows], rng) for k in metrics}
        s = summary[m]
        print(f"     {m:18} R@1={s['recall@1']['mean']:.3f} R@5={s['recall@5']['mean']:.3f} "
              f"MRR={s['mrr']['mean']:.3f} nDCG@10={s[f'ndcg@{TOP_K}']['mean']:.3f} "
              f"zero@10={s[f'zero_hit@{TOP_K}']['mean']:.3f}")

    paired = {}
    for a, b in (("bi_encoder", "bm25"), ("bi_plus_reranker", "bi_encoder"),
                 ("bi_plus_reranker", "bm25")):
        paired[f"{a}_vs_{b}"] = {
            k: _paired_delta([r[a][k] for r in rows], [r[b][k] for r in rows], rng)
            for k in ("recall@1", "recall@5", "mrr", f"ndcg@{TOP_K}")}

    def _slice(idx: Sequence[int]) -> Dict:
        j = [rows[i]["max_jaccard"] for i in idx]
        return {
            "n": len(idx),
            "jaccard_min": float(min(j)), "jaccard_max": float(max(j)),
            "mean_max_jaccard": float(np.mean(j)),
            **{m: {"mrr": float(np.mean([rows[i][m]["mrr"] for i in idx])),
                   "recall@1": float(np.mean([rows[i][m]["recall@1"] for i in idx]))}
               for m in methods},
        }

    slices = {}
    for q in (1, 2, 3, 4):
        idx = [i for i, r in enumerate(rows) if r["overlap_quartile"] == q]
        if idx:
            slices[f"q{q}"] = _slice(idx)

    # More than a quarter of queries share no token with any gold label, so the
    # 25th-percentile edge sits at zero and the lowest quartile is empty. That
    # zero-overlap group is the sharpest test of lexical matching, and its
    # boundary is fixed by the text alone, so it is also reported on its own.
    zero_idx = [i for i, r in enumerate(rows) if r["max_jaccard"] == 0.0]
    if zero_idx:
        slices["zero_overlap"] = _slice(zero_idx)

    for name, s in slices.items():
        print(f"     {name:13} n={s['n']:3} J=[{s['jaccard_min']:.3f},{s['jaccard_max']:.3f}]  "
              f"BM25 MRR={s['bm25']['mrr']:.3f}  bi={s['bi_encoder']['mrr']:.3f}  "
              f"rerank={s['bi_plus_reranker']['mrr']:.3f}")

    return {
        "benchmark": "MELO (Retyk et al. 2024), MIT licence",
        "config": CONFIG,
        "zero_shot": True,
        "n_queries": len(qids),
        "n_corpus": len(cids),
        "rerank_depth": RERANK_DEPTH,
        "candidate_recall@100": _boot_ci(cand_recall_100, rng),
        "summary": summary,
        "paired_bootstrap": paired,
        "overlap_quartiles": slices,
        "runtime_seconds": {
            "bm25_index": t_index, "corpus_encode": t_encode_corpus,
            "bm25_search": t_bm25, "bi_encoder_search": t_bi,
            "cross_encoder_rerank": t_rer,
            "total": time.perf_counter() - t_start,
        },
        "peak_memory_mb": _peak_mem_mb(),
        "per_query": rows,
    }


if __name__ == "__main__":
    print_header("Experiment 12 - External occupation linking (MELO)")
    payload = run()
    path = save_report("exp12_melo_external", payload)
    print(f"\nSaved {path}")
