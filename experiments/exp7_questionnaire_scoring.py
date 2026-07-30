"""Deterministic verification of the 25-item CSMQ scoring path.

Checks reverse-keying, item aggregation, rescaling to [0,1], and item-level
sensitivity against values derived analytically from the scoring rule. No
human data and no psychometric claim: this establishes software correctness,
not reliability or validity.
"""

from __future__ import annotations

from typing import Dict, List

from careerstep.career_positioning import ORIENTATIONS, score_csmq
from careerstep.seeding import set_global_seeds
from data.loaders import load_csmq_questionnaire
from experiments._io import print_header, save_report

LO, HI = 1, 5
MID = (LO + HI) / 2.0
# One item moved one Likert point shifts its orientation by 1/5 * 1/(HI-LO).
ITEM_STEP = (1.0 / 5.0) * (1.0 / (HI - LO))


def _responses(questionnaire, value_for) -> Dict[str, int]:
    return {it.item_id: value_for(it) for it in questionnaire.items}


def _uniform(questionnaire, v: int) -> Dict[str, int]:
    return _responses(questionnaire, lambda it: v)


def _dominant(questionnaire, orientation: str) -> Dict[str, int]:
    """Max agreement with one orientation, min with the rest, reverse-aware."""
    def val(it):
        target = it.orientation == orientation
        agree = HI if target else LO
        return (LO + HI) - agree if it.reverse else agree
    return _responses(questionnaire, val)


def _cases(questionnaire) -> List[dict]:
    cases: List[dict] = []

    for o in ORIENTATIONS:
        cases.append({
            "name": f"dominant_{o}",
            "responses": _dominant(questionnaire, o),
            "expect_dominant": o,
            "expect_scores": {o: 1.0},
        })

    two = ORIENTATIONS[0], ORIENTATIONS[2]
    def val_two(it):
        agree = HI if it.orientation in two else LO
        return (LO + HI) - agree if it.reverse else agree
    cases.append({
        "name": "two_equally_dominant",
        "responses": _responses(questionnaire, val_two),
        "expect_tie": list(two),
        "expect_scores": {two[0]: 1.0, two[1]: 1.0},
    })

    cases.append({
        "name": "all_neutral",
        "responses": _uniform(questionnaire, int(MID)),
        "expect_scores": {o: 0.5 for o in ORIENTATIONS},
    })
    # Reverse-keyed items invert, so uniform extremes are NOT all-0 or all-1:
    # each orientation has 4 forward + 1 reverse item -> 4/5 and 1/5.
    cases.append({
        "name": "all_minimum",
        "responses": _uniform(questionnaire, LO),
        "expect_scores": {o: 0.2 for o in ORIENTATIONS},
    })
    cases.append({
        "name": "all_maximum",
        "responses": _uniform(questionnaire, HI),
        "expect_scores": {o: 0.8 for o in ORIENTATIONS},
    })

    # Contradictory within one orientation: its forward items max, reverse max
    # too (i.e. the respondent agrees with both a statement and its negation).
    target = ORIENTATIONS[1]
    base = _uniform(questionnaire, int(MID))
    contra = dict(base)
    for it in questionnaire.items:
        if it.orientation == target:
            contra[it.item_id] = HI
    cases.append({
        "name": "contradictory_within_orientation",
        "responses": contra,
        # 4 forward at 1.0 + 1 reverse at 0.0 -> 0.8
        "expect_scores": {target: 0.8},
    })

    # One-point perturbations, forward and reverse-keyed.
    fwd = next(it for it in questionnaire.items if not it.reverse)
    rev = next(it for it in questionnaire.items if it.reverse)
    for label, it, delta in (("perturb_forward_item", fwd, +1),
                             ("perturb_reverse_item", rev, +1)):
        pert = dict(base)
        pert[it.item_id] = int(MID) + delta
        expected = 0.5 + (ITEM_STEP if not it.reverse else -ITEM_STEP)
        cases.append({
            "name": label,
            "responses": pert,
            "expect_scores": {it.orientation: round(expected, 10)},
            "note": f"{it.item_id} ({'reverse' if it.reverse else 'forward'}) {int(MID)}->{int(MID)+delta}",
        })

    def val_mixed(it):
        agree = HI if ORIENTATIONS.index(it.orientation) % 2 == 0 else LO
        return (LO + HI) - agree if it.reverse else agree
    cases.append({
        "name": "mixed_high_low",
        "responses": _responses(questionnaire, val_mixed),
        "expect_scores": {o: (1.0 if i % 2 == 0 else 0.0)
                          for i, o in enumerate(ORIENTATIONS)},
    })
    return cases


def run() -> dict:
    set_global_seeds()
    q = load_csmq_questionnaire()
    reverse_ids = [it.item_id for it in q.items if it.reverse]

    results = []
    for case in _cases(q):
        profile = score_csmq(case["responses"], q)
        scores = {o: round(v, 10) for o, v in profile.scores.items()}
        checks, ok = [], True

        for o, want in case.get("expect_scores", {}).items():
            hit = abs(scores[o] - want) < 1e-9
            ok &= hit
            checks.append({"orientation": o, "expected": want,
                           "actual": scores[o], "pass": hit})

        dom_expected = case.get("expect_dominant")
        if dom_expected is not None:
            hit = profile.dominant() == dom_expected
            ok &= hit
            checks.append({"dominant_expected": dom_expected,
                           "dominant_actual": profile.dominant(), "pass": hit})

        if "expect_tie" in case:
            top = max(scores.values())
            tied = sorted(o for o, v in scores.items() if abs(v - top) < 1e-9)
            hit = tied == sorted(case["expect_tie"])
            ok &= hit
            checks.append({"tie_expected": sorted(case["expect_tie"]),
                           "tie_actual": tied, "pass": hit})

        results.append({"case": case["name"], "note": case.get("note", ""),
                        "scores": scores, "checks": checks, "pass": ok})

    # determinism: identical input scored twice must agree exactly
    rerun = score_csmq(_uniform(q, int(MID)), q).scores
    again = score_csmq(_uniform(q, int(MID)), q).scores
    deterministic = rerun == again
    results.append({"case": "determinism_repeat_execution", "note": "",
                    "scores": {}, "checks": [{"pass": deterministic}],
                    "pass": deterministic})

    n_pass = sum(1 for r in results if r["pass"])
    for r in results:
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['case']}"
              + (f"   ({r['note']})" if r["note"] else ""))
    print(f"\n  {n_pass}/{len(results)} cases pass")

    return {
        "n_items": len(q.items),
        "scale": [q.scale_min, q.scale_max],
        "reverse_keyed_items": reverse_ids,
        "analytical_item_step": ITEM_STEP,
        "n_cases": len(results),
        "n_pass": n_pass,
        "all_pass": n_pass == len(results),
        "cases": results,
    }


if __name__ == "__main__":
    print_header("Experiment 7 - CSMQ item-level scoring verification")
    payload = run()
    path = save_report("exp7_questionnaire_scoring", payload)
    print(f"\nSaved {path}")
