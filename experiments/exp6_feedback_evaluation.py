"""Feedback grounding under the naive and context-aware rules."""

from __future__ import annotations

from typing import List

from data.loaders import load_feedback_references, load_occupations
from eval.grounding_ner import context_aware_grounding_summary, grounding_summary
from eval.metrics_feedback import feedback_summary
from eval.stats import summarize
from experiments._io import print_header, save_report
from careerstep.cv_optimizer import CVOptimizer
from careerstep.cv_parser import parse_cv
from careerstep.seeding import set_global_seeds
from careerstep.skill_gap import SkillGapDetector, SkillVocabulary


def _split_refs(text: str) -> List[str]:
    return [s.strip() for s in text.split(";") if s.strip()]


def run() -> dict:
    set_global_seeds()
    refs = load_feedback_references()
    occ = load_occupations()
    role_to_skills = {row["role"]: list(row["skills"]) for _, row in occ.iterrows()}
    vocab = SkillVocabulary.from_pairs(
        [(s, []) for skills in role_to_skills.values() for s in skills]
    )

    optimizer = CVOptimizer()
    detector = SkillGapDetector(vocab, mode="semantic", threshold=0.55)

    pred_cases = []         # original NER + heuristic shape
    pred_cases_ctx = []     # context-aware NER shape (adds JD/required-skills as reference)
    legacy_cases = []       # heuristic harness shape

    for _, row in refs.iterrows():
        role = row["role"]
        cv_text = row["cv_text"]
        parsed = parse_cv(cv_text)
        report = optimizer.score(parsed, target_keywords=role_to_skills.get(role, [])[:10])
        sg = detector.detect(cv_text, role, role_to_skills.get(role, []))
        bullets = report.suggestions + [f"Learn {s}." for s in sg.missing[:4]]
        reference = _split_refs(row["reference_feedback"])
        # Reference text for context-aware grounding = role + required-skill list
        # (a proxy for the JD until per-row JD text is added to the corpus).
        ref_text = role + "\n" + ", ".join(role_to_skills.get(role, []))
        pred_cases.append({"predicted": bullets, "source": cv_text})
        pred_cases_ctx.append({"predicted": bullets, "source": cv_text, "reference": ref_text})
        legacy_cases.append((bullets, reference, cv_text))

    # --- NER grounding (original metric: every entity must be in CV) ---
    ner = grounding_summary(pred_cases)

    # --- Context-aware NER grounding (new): CLAIM in CV, RECOMMENDATION exempt,
    #     REFERENCE in JD/required-skills. Turns the original metric's
    #     "learn X" false positive into a measured class.
    ner_ctx = context_aware_grounding_summary(pred_cases_ctx)

    # --- Heuristic (kept for audit comparison only) ---
    heuristic = feedback_summary(legacy_cases)

    payload = {
        # Primary metric: entity grounding.
        "ner_grounding_rate": ner["grounding_rate"],
        "ner_hallucination_rate": ner["hallucination_rate"],
        "ner_ungrounded_by_type": ner.get("ungrounded_by_type", {}),
        "ner_n_bullets": ner.get("n_bullets", 0),
        # New: context-aware NER (CLAIM / RECOMMENDATION / REFERENCE).
        "ner_context_aware_grounding_rate": ner_ctx["grounding_rate"],
        "ner_context_aware_hallucination_rate": ner_ctx["hallucination_rate"],
        "ner_context_aware_per_context": ner_ctx.get("per_context", {}),
        "ner_context_aware_ungrounded_by_type_and_context":
            ner_ctx.get("ungrounded_by_type_and_context", {}),
        "ner_context_aware_n_bullets": ner_ctx.get("n_bullets", 0),
        # Coverage / actionability (unchanged from before).
        "coverage": heuristic["coverage"],
        "actionability": heuristic["actionability"],
        "coverage_summary": summarize([c["coverage"] for c in heuristic["per_case"]]),
        "actionability_summary": summarize([c["actionability"] for c in heuristic["per_case"]]),
        # Heuristic kept for the audit table.
        "heuristic_hallucination_rate": heuristic["hallucination_rate"],
        "heuristic_groundedness": heuristic.get("groundedness", 1.0 - heuristic["hallucination_rate"]),
    }
    print(
        f"\n  NER grounding rate (original, N4)   : {payload['ner_grounding_rate']:.3f}\n"
        f"  Context-aware NER grounding rate    : {payload['ner_context_aware_grounding_rate']:.3f}\n"
        f"  ungrounded by type (original)       : {payload['ner_ungrounded_by_type']}\n"
        f"  per-context block (new)             : {payload['ner_context_aware_per_context']}\n"
        f"  heuristic hallucination (audit)     : {payload['heuristic_hallucination_rate']:.3f}\n"
        f"  coverage                            : {payload['coverage']:.3f}\n"
        f"  actionability                       : {payload['actionability']:.3f}"
    )
    return payload


if __name__ == "__main__":
    print_header("Experiment 6 - AI feedback evaluation (NER-grounded)")
    payload = run()
    path = save_report("exp6_feedback_evaluation", payload)
    print(f"\nSaved {path}")
