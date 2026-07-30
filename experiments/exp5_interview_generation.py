"""Interview-question generation against generic and retrieval baselines."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from data.loaders import load_interview_questions, load_occupations
from eval.metrics_generation import diversity, semantic_similarity
from eval.stats import summarize, wilcoxon_signed_rank
from experiments._io import print_header, save_report
from careerstep.backends import EmbeddingBackend
from careerstep.interview import InterviewQuestionGenerator
from careerstep.seeding import set_global_seeds


_GENERIC_QUESTIONS = [
    "Tell me about yourself.",
    "What are your strengths?",
    "What are your weaknesses?",
    "Why are you interested in this role?",
    "Where do you see yourself in five years?",
    "Why should we hire you?",
    "Tell me about a time you faced a challenge at work.",
    "What is your biggest professional achievement?",
    "Why are you leaving your current job?",
    "Do you have any questions for us?",
]


def _generic_baseline(role: str, k: int) -> List[str]:
    """Role-agnostic question set, padded to size k."""
    pool = _GENERIC_QUESTIONS
    out: List[str] = []
    while len(out) < k:
        out.extend(pool)
    return out[:k]


def _retrieval_baseline(
    role: str, role_skills: List[str], corpus: List[str], backend: EmbeddingBackend, k: int
) -> List[str]:
    """Top-k HR-corpus questions by similarity to a role+skills query."""
    if not corpus:
        return []
    query = role + " " + " ".join(role_skills)
    qv = backend.encode([query])[0]
    cv = backend.encode(corpus)
    qv = qv / (np.linalg.norm(qv) + 1e-9)
    cv = cv / (np.linalg.norm(cv, axis=1, keepdims=True) + 1e-9)
    sims = cv @ qv
    top = np.argsort(-sims)[:k]
    return [corpus[i] for i in top]


def _coverage(hyps: List[str], skills: List[str]) -> float:
    if not skills:
        return 0.0
    joined = " ".join(h.lower() for h in hyps)
    hits = sum(1 for s in skills if s.lower() in joined)
    return hits / len(skills)


def run(*, k: int = 10) -> dict:
    set_global_seeds()
    qs = load_interview_questions()
    occ = load_occupations()
    backend = EmbeddingBackend()
    gen = InterviewQuestionGenerator()

    role_to_refs: Dict[str, List[str]] = defaultdict(list)
    for _, row in qs.iterrows():
        role_to_refs[row["role"]].append(row["question"])
    global_corpus = qs["question"].astype(str).tolist()

    per_system_records: Dict[str, List[Dict]] = {
        "generic": [],
        "retrieval": [],
        "khutwa": [],
    }
    per_role_log: List[Dict] = []

    for _, row in occ.iterrows():
        role = row["role"]
        refs = role_to_refs.get(role, [])
        if not refs:
            continue
        skills = list(row["skills"])

        # Generate for each system.
        generic = _generic_baseline(role, k)
        retrieval = _retrieval_baseline(role, skills, global_corpus, backend, k)
        khutwa = [
            q.text
            for q in gen.generate(
                role=role, skills=skills, n_behavioral=4, n_technical=k - 4
            )
        ]
        # Trim or pad to exactly k.
        khutwa = (khutwa + [""] * k)[:k]

        log_entry: Dict = {"role": role, "skills": skills}
        for name, hyps in (("generic", generic), ("retrieval", retrieval), ("khutwa", khutwa)):
            sim = semantic_similarity(hyps, refs, backend=backend) if hyps else 0.0
            div = diversity(hyps, backend=backend) if len([h for h in hyps if h]) > 1 else 0.0
            cov = _coverage(hyps, skills)
            entry = {
                "semantic_similarity": sim,
                "diversity": div,
                "coverage": cov,
            }
            per_system_records[name].append(entry)
            log_entry[name] = entry
            log_entry[f"{name}_questions"] = hyps
        per_role_log.append(log_entry)

    aggregate: Dict[str, Dict] = {}
    for name, rows in per_system_records.items():
        aggregate[name] = {
            metric: summarize([r[metric] for r in rows])
            for metric in ("semantic_similarity", "diversity", "coverage")
        }

    # Significance: Khutwa vs each baseline on semantic similarity.
    sig: Dict[str, dict] = {}
    khutwa_sim = [r["semantic_similarity"] for r in per_system_records["khutwa"]]
    for name in ("generic", "retrieval"):
        base = [r["semantic_similarity"] for r in per_system_records[name]]
        sig[f"khutwa_vs_{name}_semantic"] = wilcoxon_signed_rank(khutwa_sim, base).to_dict()

    payload = {
        "systems": aggregate,
        "tests": sig,
        "per_role_log": per_role_log,
        "k": k,
        "note": "BLEU and ROUGE-L are intentionally omitted; see paper for rationale.",
    }
    print("\n  -- per-system aggregate --")
    for name, m in aggregate.items():
        print(
            f"    {name:9s}: sem-sim={m['semantic_similarity']['mean']:.3f}  "
            f"diversity={m['diversity']['mean']:.3f}  "
            f"coverage={m['coverage']['mean']:.3f}"
        )
    return payload


if __name__ == "__main__":
    print_header("Experiment 5 - Interview question generation (baselines, diversity)")
    payload = run()
    path = save_report("exp5_interview_generation", payload)
    print(f"\nSaved {path}")
