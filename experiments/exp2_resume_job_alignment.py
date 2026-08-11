"""CV / job-description alignment under BM25, bi-encoder, and reranker."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from data.loaders import (
    load_jds,
    load_resume_jd_pairs,
    load_resume_jd_pairs_ar,
    load_resumes,
)
from eval.metrics_retrieval import retrieval_summary
from eval.stats import paired_t_test, wilcoxon_signed_rank
from experiments._io import print_header, save_report
from careerstep.alignment import CVJobAligner
from careerstep.backends import EmbeddingBackend, LexicalBackend
from careerstep.reranker import CrossEncoderReranker
from careerstep.seeding import set_global_seeds


# Lexical-overlap stratification
def _tokenset(text: str) -> set:
    return {t for t in text.lower().split() if len(t) > 2}


def _overlap(a: str, b: str) -> float:
    A, B = _tokenset(a), _tokenset(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _stratify(values: List[float]) -> List[str]:
    """Quantile-based low/mid/high stratification."""
    if not values:
        return []
    arr = np.asarray(values)
    q1, q2 = np.quantile(arr, [1 / 3, 2 / 3])
    return ["low" if v <= q1 else ("high" if v >= q2 else "mid") for v in arr]


# Evaluation helpers
def _rank_dense_then_rerank(
    aligner: CVJobAligner,
    rer: CrossEncoderReranker,
    resumes: List[str],
    jds: List[str],
    top_k: int = 50,
) -> List[List[int]]:
    """Bi-encoder top_k -> cross-encoder rerank -> sort. Returns rank lists."""
    first_pass = aligner.rank(resumes, jds, top_k=min(top_k, len(jds)))
    out: List[List[int]] = []
    for q_idx, result in enumerate(first_pass):
        cand_ids = result.ranked_doc_ids
        if not rer.is_enabled() or len(cand_ids) <= 1:
            out.append(cand_ids)
            continue
        cand_docs = [jds[i] for i in cand_ids]
        rer_scored = rer.rerank(resumes[q_idx], cand_docs, top_k=len(cand_docs))
        out.append([cand_ids[i] for i, _ in rer_scored])
    return out


def _evaluate_rankings(
    rankings: List[List[int]],
    resume_ids: List[int],
    relevant_pos: dict,
    label: str,
) -> dict:
    pairs_eval = []
    for rid, ranks in zip(resume_ids, rankings):
        gold = relevant_pos.get(int(rid), set())
        if not gold:
            continue
        pairs_eval.append((ranks, gold))
    summary = retrieval_summary(pairs_eval, ks=(1, 5, 10))
    print(
        f"  {label}: R@1={summary['macro']['recall@1']:.4f}, "
        f"R@5={summary['macro']['recall@5']:.4f}, "
        f"MRR={summary['macro']['mrr']:.4f}, "
        f"nDCG={summary['macro']['ndcg']:.4f}"
    )
    return summary


# English track
def _run_english_track(rer: CrossEncoderReranker) -> dict:
    resumes = load_resumes().sort_values("resume_id").reset_index(drop=True)
    jds = load_jds().sort_values("jd_id").reset_index(drop=True)
    pairs = load_resume_jd_pairs()

    jd_id_index = {int(jid): pos for pos, jid in enumerate(jds["jd_id"])}

    # Exact-instance labels: the single sampled positive per CV. Retained only
    # as a secondary diagnostic; it treats the other same-role descriptions as
    # errors and therefore understates role-level retrieval.
    exact: dict = {}
    for _, row in pairs.iterrows():
        if row["label"] == 1:
            exact.setdefault(int(row["resume_id"]), set()).add(int(row["jd_id"]))
    exact_pos = {
        rid: {jd_id_index[j] for j in jset if j in jd_id_index}
        for rid, jset in exact.items()
    }

    # Role-level labels (headline): every description written for the CV's own
    # role counts as relevant. Labels come from the generator's role field and
    # are fixed before any scorer runs.
    jd_role = {int(r["jd_id"]): str(r["title"]) for _, r in jds.iterrows()}
    role_positions: dict = {}
    for jid, role in jd_role.items():
        role_positions.setdefault(role, set()).add(jd_id_index[jid])
    relevant_pos = {
        int(r["resume_id"]): set(role_positions.get(str(r["category"]), set()))
        for _, r in resumes.iterrows()
    }
    relevant_pos = {k: v for k, v in relevant_pos.items() if v}

    resume_texts = resumes["text"].tolist()
    resume_ids = resumes["resume_id"].tolist()
    jd_texts = jds["text"].tolist()

    # ---- BM25 ----
    bm25_aligner = CVJobAligner(backend=LexicalBackend())
    bm25_ranks = [
        r.ranked_doc_ids
        for r in bm25_aligner.rank(resume_texts, jd_texts, top_k=len(jd_texts))
    ]
    summ_bm25 = _evaluate_rankings(bm25_ranks, resume_ids, relevant_pos, "baseline_bm25")
    exact_bm25 = _evaluate_rankings(bm25_ranks, resume_ids, exact_pos, "baseline_bm25")

    # ---- bi-encoder ----
    bi_aligner = CVJobAligner(backend=EmbeddingBackend())
    bi_ranks = [
        r.ranked_doc_ids
        for r in bi_aligner.rank(resume_texts, jd_texts, top_k=len(jd_texts))
    ]
    summ_bi = _evaluate_rankings(bi_ranks, resume_ids, relevant_pos, "khutwa_bi_encoder")
    exact_bi = _evaluate_rankings(bi_ranks, resume_ids, exact_pos, "khutwa_bi_encoder")

    # ---- bi + reranker ----
    bi_plus = _rank_dense_then_rerank(bi_aligner, rer, resume_texts, jd_texts, top_k=50)
    summ_rr = _evaluate_rankings(bi_plus, resume_ids, relevant_pos, "khutwa_bi_plus_reranker")
    exact_rr = _evaluate_rankings(bi_plus, resume_ids, exact_pos, "khutwa_bi_plus_reranker")

    # ---- stratification (audit) ----
    # For each resume with a gold JD, compute lexical overlap with that JD,
    # then bucket queries low/mid/high and recompute MRR per bucket.
    overlap_vals: List[float] = []
    strat_indices: List[int] = []
    for idx, rid in enumerate(resume_ids):
        gold = exact_pos.get(int(rid), set())
        if not gold:
            continue
        gold_jd = jd_texts[next(iter(gold))]
        overlap_vals.append(_overlap(resume_texts[idx], gold_jd))
        strat_indices.append(idx)
    buckets = _stratify(overlap_vals)

    def _bucket_mrr(rankings: List[List[int]]) -> dict:
        from collections import defaultdict

        accum: defaultdict = defaultdict(list)
        for sidx, bucket in zip(strat_indices, buckets):
            ranks = rankings[sidx]
            gold = exact_pos.get(int(resume_ids[sidx]), set())
            if not gold:
                continue
            rr = 0.0
            for r_pos, jd_pos in enumerate(ranks, 1):
                if jd_pos in gold:
                    rr = 1.0 / r_pos
                    break
            accum[bucket].append(rr)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in accum.items()}

    stratified = {
        "baseline_bm25": _bucket_mrr(bm25_ranks),
        "khutwa_bi_encoder": _bucket_mrr(bi_ranks),
        "khutwa_bi_plus_reranker": _bucket_mrr(bi_plus),
    }
    print("  stratified MRR by lexical overlap (low/mid/high):")
    for sys_name, b in stratified.items():
        print(f"    {sys_name}: low={b.get('low', 0):.3f}  mid={b.get('mid', 0):.3f}  high={b.get('high', 0):.3f}")

    # ---- significance tests ----
    rr_bm25 = [q["mrr"] for q in summ_bm25["per_query"]]
    rr_rr = [q["mrr"] for q in summ_rr["per_query"]]
    rr_bi = [q["mrr"] for q in summ_bi["per_query"]]

    n_rel = len(next(iter(relevant_pos.values()))) if relevant_pos else 0
    return {
        "relevance_scheme": {
            "headline": "role_level",
            "role_level_relevant_per_query": n_rel,
            "n_documents": len(jd_texts),
            "recall_ceiling@1": (1.0 / n_rel) if n_rel else 0.0,
            "recall_ceiling@5": (min(5, n_rel) / n_rel) if n_rel else 0.0,
            "note": ("role_level marks every description written for the query "
                     "CV's role as relevant. exact_instance marks only the one "
                     "sampled description, so the other same-role descriptions "
                     "count as errors; it measures template-instance recovery, "
                     "not role alignment."),
        },
        "exact_instance": {
            "baseline_bm25": exact_bm25["macro"],
            "khutwa_bi_encoder": exact_bi["macro"],
            "khutwa_bi_plus_reranker": exact_rr["macro"],
        },
        "baseline_bm25": summ_bm25,
        "khutwa_bi_encoder": summ_bi,
        "khutwa_bi_plus_reranker": summ_rr,
        "stratified_mrr_by_overlap": stratified,
        "tests": {
            "paired_t_rr_reranker_vs_bm25": paired_t_test(rr_rr, rr_bm25).to_dict(),
            "wilcoxon_rr_reranker_vs_bm25": wilcoxon_signed_rank(rr_rr, rr_bm25).to_dict(),
            "paired_t_rr_bi_vs_bm25": paired_t_test(rr_bi, rr_bm25).to_dict(),
        },
        "reranker_enabled": rer.is_enabled(),
    }


# Arabic track
def _run_arabic_track(rer: CrossEncoderReranker) -> dict:
    """Treat each Arabic CV as a query against the bag of all Arabic JDs.

    The matched JD for each CV is the positive; all other JDs are negatives.
    Reports the same R@k / MRR / nDCG as the English track.
    """
    pairs = load_resume_jd_pairs_ar()
    if pairs.empty:
        return {"status": "skipped"}

    resume_texts = pairs["resume"].tolist()
    jd_texts = pairs["job_description"].tolist()
    resume_ids = list(range(len(resume_texts)))
    # Role-level labels: every Arabic description written for the CV's own role
    # counts as relevant. The earlier version marked only the same-row
    # description relevant, so the other five same-role descriptions were
    # scored as errors. The exact-row set is kept as a secondary diagnostic.
    roles = pairs["role"].tolist()
    by_role = {}
    for i, r in enumerate(roles):
        by_role.setdefault(r, set()).add(i)
    relevant_pos = {i: set(by_role[roles[i]]) for i in resume_ids}
    exact_pos = {i: {i} for i in resume_ids}

    bm25_aligner = CVJobAligner(backend=LexicalBackend())
    bm25_ranks = [
        r.ranked_doc_ids
        for r in bm25_aligner.rank(resume_texts, jd_texts, top_k=len(jd_texts))
    ]
    summ_bm25 = _evaluate_rankings(bm25_ranks, resume_ids, relevant_pos, "ar_baseline_bm25")

    bi_aligner = CVJobAligner(backend=EmbeddingBackend())
    bi_ranks = [
        r.ranked_doc_ids
        for r in bi_aligner.rank(resume_texts, jd_texts, top_k=len(jd_texts))
    ]
    summ_bi = _evaluate_rankings(bi_ranks, resume_ids, relevant_pos, "ar_khutwa_bi_encoder")

    bi_plus = _rank_dense_then_rerank(bi_aligner, rer, resume_texts, jd_texts, top_k=20)
    summ_rr = _evaluate_rankings(bi_plus, resume_ids, relevant_pos, "ar_khutwa_bi_plus_reranker")

    exact_bm25 = _evaluate_rankings(bm25_ranks, resume_ids, exact_pos, "ar_baseline_bm25")
    exact_bi = _evaluate_rankings(bi_ranks, resume_ids, exact_pos, "ar_khutwa_bi_encoder")
    exact_rr = _evaluate_rankings(bi_plus, resume_ids, exact_pos, "ar_khutwa_bi_plus_reranker")

    rr_bm25 = [q["mrr"] for q in summ_bm25["per_query"]]
    rr_rr = [q["mrr"] for q in summ_rr["per_query"]]

    n_rel = len(next(iter(relevant_pos.values()))) if relevant_pos else 0
    return {
        "relevance_scheme": {
            "headline": "role_level",
            "role_level_relevant_per_query": n_rel,
            "n_documents": len(jd_texts),
            "recall_ceiling@1": (1.0 / n_rel) if n_rel else 0.0,
            "recall_ceiling@5": (min(5, n_rel) / n_rel) if n_rel else 0.0,
        },
        "exact_instance": {
            "baseline_bm25": exact_bm25["macro"],
            "khutwa_bi_encoder": exact_bi["macro"],
            "khutwa_bi_plus_reranker": exact_rr["macro"],
        },
        "baseline_bm25": summ_bm25,
        "khutwa_bi_encoder": summ_bi,
        "khutwa_bi_plus_reranker": summ_rr,
        "n_pairs": len(resume_texts),
        "tests": {
            "paired_t_rr_reranker_vs_bm25": paired_t_test(rr_rr, rr_bm25).to_dict(),
            "wilcoxon_rr_reranker_vs_bm25": wilcoxon_signed_rank(rr_rr, rr_bm25).to_dict(),
        },
    }


def run() -> dict:
    set_global_seeds()
    rer = CrossEncoderReranker()
    en = _run_english_track(rer)
    print("\n  -- Arabic track --")
    ar = _run_arabic_track(rer)
    return {"english": en, "arabic": ar}


if __name__ == "__main__":
    print_header("Experiment 2 - Resume<->JD alignment (rigorous, bilingual)")
    payload = run()
    path = save_report("exp2_resume_job_alignment", payload)
    print(f"\nSaved {path}")
