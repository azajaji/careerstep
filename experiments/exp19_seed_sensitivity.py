"""Does the influence result depend on the profile generator's seed?

Every table in the ranking family is computed on one 120-profile cohort drawn
from one seed. That cohort is frozen so the tables are comparable, but a single
draw cannot show whether the reported influence ordering is a property of the
scorer or of that draw.

This experiment repeats the Experiment 8f measures over independent cohorts and
reports the distribution of each quantity. Nothing here replaces the headline
table; it bounds how far the headline could move under resampling of the
generator alone. Generator parameters, catalog, and weights are held fixed, so
the intervals do not cover those choices.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import stats

from careerstep.benchmark_cohort import build_cohort
from careerstep.career_positioning import RoleRecommender, _cosine
from careerstep.career_positioning_benchmark import (
    FEASIBILITY, W_FEAS, W_SKILL, W_VALUE,
    lexical_skill_readiness, role_to_required_skills,
)
from careerstep.seeding import set_global_seeds
from careerstep.tie_aware import max_set
from data.loaders import load_onet_work_values, load_saudi_cyber_roles
from experiments._io import print_header, save_report

CRITERIA = ("value_fit", "skill_readiness", "feasibility")
WEIGHTS = {"value_fit": W_VALUE, "skill_readiness": W_SKILL, "feasibility": W_FEAS}
N_SEEDS = 20


def _criterion_matrices(profiles, rec, required, metas, role_ids):
    out = []
    for p in profiles:
        rows = []
        for i, rid in enumerate(role_ids):
            rows.append([
                _cosine(p.csmq_vector, rec.centroids[i]),
                lexical_skill_readiness(p.known_skills, required[rid]),
                FEASIBILITY.get((p.level, str(metas.loc[rid, "seniority"]).lower()), 0.5),
            ])
        out.append(np.asarray(rows, dtype=float))
    return out


def run() -> dict:
    set_global_seeds()
    roles_df = load_saudi_cyber_roles()
    rec = RoleRecommender(roles_df, load_onet_work_values())
    required = role_to_required_skills(roles_df)
    metas = roles_df.set_index("role_id")
    role_ids = list(rec.centroid_df.index)
    w = np.array([WEIGHTS[c] for c in CRITERIA], dtype=float)

    per_seed: Dict[str, Dict[str, List[float]]] = {
        c: {"dispersion": [], "top1": [], "strict": [], "decides": [], "tau": []}
        for c in CRITERIA
    }

    for s in range(N_SEEDS):
        profiles = build_cohort(rec, roles_df, seed_offset=s)
        mats = _criterion_matrices(profiles, rec, required, metas, role_ids)

        shares = []
        for X in mats:
            sd = X.std(axis=0, ddof=1)
            ws = w * sd
            shares.append(ws / ws.sum() if ws.sum() > 0 else np.zeros_like(ws))
        shares = np.asarray(shares)

        rank = lambda X: np.argsort(-(X @ w), kind="stable")
        for j, c in enumerate(CRITERIA):
            top1, strict, taus = [], [], []
            dec = 0
            npairs = 0
            for X in mats:
                full = rank(X)
                Y = X.copy()
                Y[:, j] = Y[:, j].mean()
                got = rank(Y)
                top1.append(0.0 if full[0] == got[0] else 1.0)
                strict.append(0.0 if full[0] in set(max_set(Y @ w).tolist()) else 1.0)
                taus.append(stats.kendalltau(X @ w, Y @ w, variant="b").statistic)
                d = X[:, None, :] - X[None, :, :]
                iu = np.triu_indices(X.shape[0], k=1)
                diffs = d[iu] * w
                total = diffs.sum(axis=1)
                others = np.delete(diffs, j, axis=1).sum(axis=1)
                dec += int(np.sum((np.sign(diffs[:, j]) == np.sign(total))
                                  & (np.abs(diffs[:, j]) > np.abs(others))))
                npairs += len(total)
            per_seed[c]["dispersion"].append(float(shares[:, j].mean()))
            per_seed[c]["top1"].append(float(np.mean(top1)))
            per_seed[c]["strict"].append(float(np.mean(strict)))
            per_seed[c]["tau"].append(float(np.mean(taus)))
            per_seed[c]["decides"].append(float(dec / npairs))

    def summ(v):
        a = np.asarray(v, dtype=float)
        return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
                "ci95_low": float(np.percentile(a, 2.5)),
                "ci95_high": float(np.percentile(a, 97.5))}

    result = {c: {k: summ(v) for k, v in d.items()} for c, d in per_seed.items()}

    # Does the ordering itself hold across seeds?
    disp_rank_stable = all(
        per_seed["feasibility"]["dispersion"][i] > per_seed["skill_readiness"]["dispersion"][i]
        > per_seed["value_fit"]["dispersion"][i] for i in range(N_SEEDS))
    value_lowest_top1 = all(
        per_seed["value_fit"]["top1"][i] < min(per_seed["skill_readiness"]["top1"][i],
                                               per_seed["feasibility"]["top1"][i])
        for i in range(N_SEEDS))

    payload = {
        "n_seeds": N_SEEDS,
        "n_profiles_per_seed": 120,
        "weights": {c: WEIGHTS[c] for c in CRITERIA},
        "per_criterion": result,
        "ordering_stability": {
            "dispersion_order_feasibility_gt_skill_gt_value_in_all_seeds": bool(disp_rank_stable),
            "value_fit_lowest_top1_change_in_all_seeds": bool(value_lowest_top1),
        },
        "notes": {
            "scope": ("resampling of the profile generator only; the catalog, "
                      "generator parameters, and weights are fixed, so these "
                      "intervals do not cover those choices"),
        },
    }

    print_header("Experiment 19 - Seed sensitivity of the influence result")
    print(f"  {N_SEEDS} independent 120-profile cohorts")
    for c in CRITERIA:
        r = result[c]
        print(f"  {c:<16} disp {r['dispersion']['mean']:.3f} "
              f"[{r['dispersion']['min']:.3f}, {r['dispersion']['max']:.3f}]   "
              f"top-1 {r['top1']['mean']:.3f} "
              f"[{r['top1']['min']:.3f}, {r['top1']['max']:.3f}]   "
              f"strict {r['strict']['mean']:.3f}   "
              f"decides {r['decides']['mean']:.3f}")
    print(f"  dispersion ordering holds in all seeds: {disp_rank_stable}")
    print(f"  value-fit lowest top-1 in all seeds:    {value_lowest_top1}")
    save_report("exp19_seed_sensitivity", payload)
    return payload


if __name__ == "__main__":
    run()
