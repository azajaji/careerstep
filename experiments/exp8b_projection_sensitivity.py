"""Sensitivity of the benchmark to perturbations of the projection matrix W."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

import careerstep.career_positioning as cp
from data.loaders import (
    load_csmq_questionnaire,
    load_onet_work_values,
    load_saudi_cyber_roles,
)
from careerstep.career_positioning import (
    ORIENTATIONS,
    OrientationProfile,
    RoleRecommender,
)
from careerstep.career_positioning_benchmark import (
    lexical_skill_readiness,
    mrr as ranking_mrr,
    rank_csmq_only,
    rank_khutwa,
    rank_skills_only,
    recall_at_k,
    role_to_required_skills,
    simulate_profiles,
)
from careerstep.seeding import set_global_seeds
from careerstep.benchmark_cohort import build_cohort, cohort_fingerprint
from experiments._io import save_report

# The original, manuscript matrix (independent of any later monkeypatch).
W0 = np.array(cp._PROJECTION, dtype=float, copy=True)


def _l1_normalise_rows(W: np.ndarray) -> np.ndarray:
    """Match the manuscript invariant: each row's absolute weights sum to 1."""
    out = W.copy()
    for i in range(out.shape[0]):
        s = np.abs(out[i]).sum()
        if s > 0:
            out[i] = out[i] / s
    return out


def _perturb(W: np.ndarray, sigma: float, np_rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to every weight, then re-L1-normalise each row.

    Zero entries are perturbed too, so a perturbation can introduce a new
    (small) cross-loading or flip a small weight's sign -- a fair stress
    test of whether the exact zero/sign pattern is load-bearing.
    """
    noise = np_rng.normal(0.0, sigma, size=W.shape)
    return _l1_normalise_rows(W + noise)


def _perturb_relative(W: np.ndarray, frac: float, np_rng: np.random.Generator) -> np.ndarray:
    """Scale every NON-ZERO weight by an independent uniform factor in
    [1-frac, 1+frac] (the reviewer's literal "vary weights +/-20%"), then
    re-L1-normalise each row. Structural zeros stay zero here."""
    factor = 1.0 + np_rng.uniform(-frac, frac, size=W.shape)
    out = W * factor
    return _l1_normalise_rows(out)


def _self_consistency(rec: RoleRecommender) -> Dict[str, float]:
    role_ids = list(rec.centroid_df.index)
    top1 = top3 = top5 = 0
    rr = 0.0
    for rid in role_ids:
        centroid = rec.centroid_df.loc[rid].to_dict()
        profile = OrientationProfile(scores={o: float(centroid[o]) for o in ORIENTATIONS})
        ranked = [r.role_id for r in rec.recommend(profile, top_k=len(role_ids))]
        pos = ranked.index(rid) + 1 if rid in ranked else None
        if pos:
            rr += 1.0 / pos
            top1 += int(pos <= 1)
            top3 += int(pos <= 3)
            top5 += int(pos <= 5)
    n = len(role_ids)
    return {"mrr": rr / n, "top1": top1 / n, "top3": top3 / n, "top5": top5 / n}


def _ranking(rec: RoleRecommender, roles_df, *, n_profiles: int,
             py_rng: random.Random, np_rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    """Lexical-scorer ranking benchmark (lexical == full on this structured-token
    benchmark, per the manuscript), so this reproduces the headline composite."""
    profiles = build_cohort(rec, roles_df)
    required = role_to_required_skills(roles_df)

    def _lex(user, req):
        return lexical_skill_readiness(user, req)

    rankers = {
        "skills_only": lambda p: rank_skills_only(p, rec, required, _lex),
        "csmq_only": lambda p: rank_csmq_only(p, rec),
        "csmq_khutwa": lambda p: rank_khutwa(p, rec, roles_df, required, _lex),
    }
    acc: Dict[str, Dict[str, List[float]]] = {
        name: {"recall@3": [], "recall@5": [], "mrr": []} for name in rankers
    }
    for prof in profiles:
        for name, fn in rankers.items():
            ranked = fn(prof)
            acc[name]["recall@3"].append(recall_at_k(ranked, prof.acceptable_role_ids, 3))
            acc[name]["recall@5"].append(recall_at_k(ranked, prof.acceptable_role_ids, 5))
            acc[name]["mrr"].append(ranking_mrr(ranked, prof.acceptable_role_ids))
    return {name: {m: float(np.mean(v)) for m, v in metr.items()} for name, metr in acc.items()}


def _build_recommender(W: np.ndarray, roles_df, wv_df) -> RoleRecommender:
    """Rebuild the recommender under projection matrix W (monkeypatch the module
    global used in RoleRecommender.__post_init__)."""
    cp._PROJECTION = W
    return RoleRecommender(roles_df=roles_df, wv_df=wv_df)


def _advance_like_exp8_noise_curve(np_rng: np.random.Generator, n_roles: int) -> None:
    """Advance the RNG exactly as exp8 does before simulate_profiles.

    exp8 consumes the same Generator in its noise curve first, so the profile
    draws depend on that prior consumption. Values are discarded; only the
    state advances."""
    for sigma in (0.00, 0.05, 0.10, 0.20, 0.30):
        for _ in range(n_roles):
            for _ in range(20):
                np_rng.normal(0.0, sigma, size=(5,))


def _frozen_cases(W_base: np.ndarray, roles_df, wv_df, seeds, n_profiles: int = 120):
    """Build the benchmark cases once, under the baseline projection.

    The earlier version of this experiment regenerated profiles and
    acceptable-role labels from whichever W was under test. Because
    ``simulate_profiles`` sets each profile to its latent role's centroid and
    defines acceptable roles as the nearest centroids, both the inputs and the
    labels moved with the parameter being perturbed, so the result could not
    distinguish robustness from self-consistency. The cases are now fixed here
    and reused unchanged for every perturbation; only the scorer varies."""
    rec = _build_recommender(W_base, roles_df, wv_df)
    py_rng = random.Random(seeds["python_random_seed"])
    return build_cohort(rec, roles_df)


def _rank_fixed_cases(rec: RoleRecommender, roles_df, profiles) -> Dict[str, Dict[str, float]]:
    """Score pre-built cases under ``rec``; labels come from the frozen cases."""
    required = role_to_required_skills(roles_df)

    def _lex(user, req):
        return lexical_skill_readiness(user, req)

    rankers = {
        "skills_only": lambda p: rank_skills_only(p, rec, required, _lex),
        "csmq_only": lambda p: rank_csmq_only(p, rec),
        "csmq_khutwa": lambda p: rank_khutwa(p, rec, roles_df, required, _lex),
    }
    acc: Dict[str, Dict[str, List[float]]] = {
        name: {"recall@3": [], "recall@5": [], "mrr": []} for name in rankers
    }
    for prof in profiles:
        for name, fn in rankers.items():
            ranked = fn(prof)
            acc[name]["recall@3"].append(recall_at_k(ranked, prof.acceptable_role_ids, 3))
            acc[name]["recall@5"].append(recall_at_k(ranked, prof.acceptable_role_ids, 5))
            acc[name]["mrr"].append(ranking_mrr(ranked, prof.acceptable_role_ids))
    return {name: {m: float(np.mean(v)) for m, v in metr.items()}
            for name, metr in acc.items()}


def _evaluate_once(W: np.ndarray, roles_df, wv_df, seeds, profiles) -> Dict:
    """Evaluate perturbed W on the frozen cases in ``profiles``.

    ``self_consistency`` still back-projects the perturbed centroids, which is
    intrinsic to that measure; the ranking arm now uses fixed inputs/labels."""
    rec = _build_recommender(W, roles_df, wv_df)
    sc = _self_consistency(rec)
    rk = _rank_fixed_cases(rec, roles_df, profiles)
    return {"self_consistency": sc, "ranking": rk}


def _band(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "p2.5": float(np.percentile(a, 2.5)),
        "p97.5": float(np.percentile(a, 97.5)),
    }


def run(n_trials: int = 200, sigmas=(0.05, 0.10)) -> Dict:
    seeds = set_global_seeds()
    roles = load_saudi_cyber_roles()
    wv = load_onet_work_values()
    _ = load_csmq_questionnaire()  # parity with exp8 load order

    # Build the benchmark cases once, under the baseline projection, and reuse
    # them for every perturbation. Only the scorer changes across trials.
    cases = _frozen_cases(W0, roles, wv, seeds, n_profiles=120)

    # Baseline (unperturbed W) -- must reproduce the published headline.
    baseline = _evaluate_once(W0, roles, wv, seeds, cases)

    # Perturbation regimes. The perturbation RNG is seeded once and advanced
    # across trials, so the W' draws are reproducible.
    pert_rng = np.random.default_rng(20260623)
    regimes: Dict[str, Dict] = {}
    for sigma in sigmas:
        sc_keys = ["mrr", "top1", "top3", "top5"]
        rk_targets = {
            "csmq_only": ["recall@3", "recall@5", "mrr"],
            "csmq_khutwa": ["recall@3", "recall@5", "mrr"],
            "skills_only": ["recall@5"],
        }
        sc_acc = {k: [] for k in sc_keys}
        rk_acc = {r: {m: [] for m in ms} for r, ms in rk_targets.items()}
        n_sign_flips = []
        for _t in range(n_trials):
            Wp = _perturb(W0, sigma, pert_rng)
            # Count how many weights changed sign vs the L1-normalised original.
            base_norm = _l1_normalise_rows(W0)
            n_sign_flips.append(int(np.sum(np.sign(Wp) != np.sign(base_norm))))
            res = _evaluate_once(Wp, roles, wv, seeds, cases)
            for k in sc_keys:
                sc_acc[k].append(res["self_consistency"][k])
            for r, ms in rk_targets.items():
                for m in ms:
                    rk_acc[r][m].append(res["ranking"][r][m])
        regimes[f"sigma_{sigma}"] = {
            "n_trials": n_trials,
            "mean_sign_flips_per_matrix": float(np.mean(n_sign_flips)),
            "self_consistency": {k: _band(v) for k, v in sc_acc.items()},
            "ranking": {r: {m: _band(v) for m, v in ms.items()} for r, ms in rk_acc.items()},
        }

    # Literal relative +/-20% regime (the reviewer's wording): each non-zero
    # weight scaled by uniform[0.8, 1.2], structural zeros preserved.
    for frac in (0.20,):
        sc_keys = ["mrr", "top1", "top3", "top5"]
        rk_targets = {
            "csmq_only": ["recall@3", "recall@5", "mrr"],
            "csmq_khutwa": ["recall@3", "recall@5", "mrr"],
            "skills_only": ["recall@5"],
        }
        sc_acc = {k: [] for k in sc_keys}
        rk_acc = {r: {m: [] for m in ms} for r, ms in rk_targets.items()}
        for _t in range(n_trials):
            Wp = _perturb_relative(W0, frac, pert_rng)
            res = _evaluate_once(Wp, roles, wv, seeds, cases)
            for k in sc_keys:
                sc_acc[k].append(res["self_consistency"][k])
            for r, ms in rk_targets.items():
                for m in ms:
                    rk_acc[r][m].append(res["ranking"][r][m])
        regimes[f"relative_{frac}"] = {
            "n_trials": n_trials,
            "description": "each non-zero weight scaled by uniform[1-frac,1+frac]",
            "self_consistency": {k: _band(v) for k, v in sc_acc.items()},
            "ranking": {r: {m: _band(v) for m, v in ms.items()} for r, ms in rk_acc.items()},
        }

    # Restore the module global so nothing downstream is affected.
    cp._PROJECTION = W0
    return {"baseline": baseline, "perturbation": regimes,
            "notes": {
                "W_invariant": "rows L1-normalised (abs weights sum to 1)",
                "skills_only_is_W_invariant": True,
                "scorer": "lexical (== full composite on this structured-token benchmark)",
            }}


if __name__ == "__main__":
    payload = run()
    save_report("exp8_projection_sensitivity", payload)

    def fmt(band):
        return "mean={:.3f} sd={:.3f} [{:.3f},{:.3f}] min={:.3f}".format(
            band["mean"], band["sd"], band["p2.5"], band["p97.5"], band["min"])

    b = payload["baseline"]
    print("=== baseline (unperturbed W, lexical scorer) ===")
    print("  self-consistency: top5={top5:.3f} top3={top3:.3f} mrr={mrr:.3f}".format(**b["self_consistency"]))
    print("  csmq_only   Hit@3={:.3f}".format(b["ranking"]["csmq_only"]["recall@3"]))
    print("  csmq_khutwa Hit@5={:.3f} mrr={:.3f}".format(
        b["ranking"]["csmq_khutwa"]["recall@5"], b["ranking"]["csmq_khutwa"]["mrr"]))
    print("  skills_only Hit@5={:.3f}".format(b["ranking"]["skills_only"]["recall@5"]))
    for name, reg in payload["perturbation"].items():
        flips = reg.get("mean_sign_flips_per_matrix")
        flip_str = f", mean sign-flips/matrix={flips:.1f}" if flips is not None else ""
        print(f"\n=== {name}  (N={reg['n_trials']}{flip_str}) ===")
        sc = reg["self_consistency"]
        print("  self-consistency top5: " + fmt(sc["top5"]))
        print("  self-consistency top3: " + fmt(sc["top3"]))
        rk = reg["ranking"]
        print("  csmq_khutwa Hit@5:     " + fmt(rk["csmq_khutwa"]["recall@5"]))
        print("  csmq_khutwa MRR:       " + fmt(rk["csmq_khutwa"]["mrr"]))
        print("  csmq_only   Hit@3:     " + fmt(rk["csmq_only"]["recall@3"]))
    print("Saved results/exp8_projection_sensitivity.json")
