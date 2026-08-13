"""How much does each criterion actually move the ranking?

The pooled statistic reported earlier, w_j sigma_j over all profile-by-role
pairs, is a dispersion heuristic and not a share of variance. It ignores
covariance between criteria, and it pools across profiles even though ranking
happens separately within each profile. A criterion that varies a lot between
students but little between the roles offered to one student cannot reorder
that student's shortlist.

This experiment replaces it with measurements taken at the level where the
decision is made:

1. Within-profile dispersion. For each profile, the standard deviation of each
   criterion across the candidate roles, weighted by its coefficient and
   normalized across criteria, then averaged over profiles.

2. Leave-one-criterion-out. Hold a criterion at its within-profile mean so it
   cannot discriminate, re-rank, and measure the damage with Kendall's tau,
   top-1 change rate, and top-5 overlap against the full ranking.

3. Pairwise score margins. For every ordered pair of roles inside a profile,
   the contribution w_j [x_j(a) - x_j(b)] each criterion makes to the margin
   that separates them, reported as a mean absolute contribution and as the
   share of pairs whose ordering that criterion alone would decide.
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
from scipy import stats

from careerstep.career_positioning import RoleRecommender, _cosine
from careerstep.career_positioning_benchmark import (
    FEASIBILITY, W_FEAS, W_SKILL, W_VALUE,
    lexical_skill_readiness, role_to_required_skills, simulate_profiles,
)
from careerstep.seeding import load_seeds, set_global_seeds
from careerstep.tie_aware import max_set, rank_random_tiebreak
from careerstep.benchmark_cohort import build_cohort, cohort_fingerprint
from data.loaders import load_onet_work_values, load_saudi_cyber_roles
from experiments._io import print_header, save_report

CRITERIA = ("value_fit", "skill_readiness", "feasibility")
WEIGHTS = {"value_fit": W_VALUE, "skill_readiness": W_SKILL, "feasibility": W_FEAS}


def run() -> dict:
    set_global_seeds()
    seeds = load_seeds()
    roles_df = load_saudi_cyber_roles()
    rec = RoleRecommender(roles_df, load_onet_work_values())
    required = role_to_required_skills(roles_df)
    metas = roles_df.set_index("role_id")
    role_ids = list(rec.centroid_df.index)

    profiles = build_cohort(rec, roles_df)

    # Criterion matrix per profile: rows are candidate roles, columns criteria.
    per_profile: List[np.ndarray] = []
    for p in profiles:
        rows = []
        for i, rid in enumerate(role_ids):
            vf = _cosine(p.csmq_vector, rec.centroids[i])
            sk = lexical_skill_readiness(p.known_skills, required[rid])
            fe = FEASIBILITY.get((p.level, str(metas.loc[rid, "seniority"]).lower()), 0.5)
            rows.append([vf, sk, fe])
        per_profile.append(np.asarray(rows, dtype=float))

    w = np.array([WEIGHTS[c] for c in CRITERIA], dtype=float)

    # 1. Within-profile weighted dispersion, normalized per profile.
    shares = []
    raw_sd = []
    for X in per_profile:
        sd = X.std(axis=0, ddof=1)
        raw_sd.append(sd)
        ws = w * sd
        shares.append(ws / ws.sum() if ws.sum() > 0 else np.zeros_like(ws))
    shares = np.asarray(shares)
    raw_sd = np.asarray(raw_sd)
    within = {
        c: {
            "within_profile_sd_mean": float(raw_sd[:, j].mean()),
            "weighted_dispersion_share_mean": float(shares[:, j].mean()),
            "weighted_dispersion_share_sd": float(shares[:, j].std(ddof=1)),
        }
        for j, c in enumerate(CRITERIA)
    }

    # 2. Leave-one-criterion-out.
    def rank(X: np.ndarray) -> np.ndarray:
        return np.argsort(-(X @ w), kind="stable")

    loco: Dict[str, dict] = {}
    tie_rng = np.random.default_rng(20260101)
    for j, c in enumerate(CRITERIA):
        taus, taubs, top1, ov5, strict, maxset = [], [], [], [], [], []
        for X in per_profile:
            full = rank(X)
            Y = X.copy()
            Y[:, j] = Y[:, j].mean()           # criterion can no longer discriminate
            got = rank(Y)
            pos_full = np.empty(len(full), dtype=int)
            pos_got = np.empty(len(got), dtype=int)
            pos_full[full] = np.arange(len(full))
            pos_got[got] = np.arange(len(got))
            taus.append(stats.kendalltau(pos_full, pos_got).statistic)
            # tau-b on the scores themselves, which is defined under ties.
            taubs.append(stats.kendalltau(X @ w, Y @ w, variant="b").statistic)
            top1.append(0.0 if full[0] == got[0] else 1.0)
            a, b = set(full[:5].tolist()), set(got[:5].tolist())
            ov5.append(len(a & b) / len(a | b))
            # Tie-free statement: is the original leader still a maximum?
            ms = max_set(Y @ w)
            maxset.append(len(ms))
            strict.append(0.0 if full[0] in set(ms.tolist()) else 1.0)

        # Expected top-1 change under random tie-breaking.
        exp_top1 = []
        for _ in range(200):
            hits = []
            for X in per_profile:
                Y = X.copy()
                Y[:, j] = Y[:, j].mean()
                lead_full = rank_random_tiebreak(X @ w, tie_rng)[0]
                lead_abl = rank_random_tiebreak(Y @ w, tie_rng)[0]
                hits.append(0.0 if lead_full == lead_abl else 1.0)
            exp_top1.append(float(np.mean(hits)))
        exp_top1 = np.asarray(exp_top1)

        # Catalog-order sensitivity of the reported-style rate.
        perm_rates = []
        perm_rng = np.random.default_rng(7)
        for _ in range(200):
            order = perm_rng.permutation(per_profile[0].shape[0])
            hits = []
            for X in per_profile:
                Xp = X[order]
                Yp = Xp.copy()
                Yp[:, j] = Yp[:, j].mean()
                hits.append(0.0 if rank(Xp)[0] == rank(Yp)[0] else 1.0)
            perm_rates.append(float(np.mean(hits)))

        loco[c] = {
            "kendall_tau_mean": float(np.mean(taus)),
            "kendall_tau_b_mean": float(np.mean(taubs)),
            "top1_change_rate": float(np.mean(top1)),
            "strict_top_set_loss": float(np.mean(strict)),
            "mean_tied_maxima_after_ablation": float(np.mean(maxset)),
            "expected_top1_change": float(exp_top1.mean()),
            "expected_top1_ci95": [float(np.percentile(exp_top1, 2.5)),
                                   float(np.percentile(exp_top1, 97.5))],
            "catalog_order_top1_min": float(np.min(perm_rates)),
            "catalog_order_top1_max": float(np.max(perm_rates)),
            "top5_jaccard_mean": float(np.mean(ov5)),
        }

    # 3. Pairwise margin contributions inside each profile.
    contrib_abs = np.zeros(len(CRITERIA))
    decisive = np.zeros(len(CRITERIA))
    n_pairs = 0
    for X in per_profile:
        d = X[:, None, :] - X[None, :, :]           # role a minus role b
        iu = np.triu_indices(X.shape[0], k=1)
        diffs = d[iu] * w                            # weighted contributions
        total = diffs.sum(axis=1)
        contrib_abs += np.abs(diffs).sum(axis=0)
        # A criterion is decisive for a pair when it alone matches the sign of
        # the total margin and exceeds the combined magnitude of the others.
        for j in range(len(CRITERIA)):
            others = np.delete(diffs, j, axis=1).sum(axis=1)
            decisive[j] += np.sum((np.sign(diffs[:, j]) == np.sign(total))
                                  & (np.abs(diffs[:, j]) > np.abs(others)))
        n_pairs += len(total)
    margins = {
        c: {
            "mean_abs_contribution": float(contrib_abs[j] / n_pairs),
            "share_of_pairs_decided": float(decisive[j] / n_pairs),
        }
        for j, c in enumerate(CRITERIA)
    }

    payload = {
        "n_profiles": len(profiles),
        "n_roles": len(role_ids),
        "n_within_profile_pairs": int(n_pairs),
        "weights": {c: WEIGHTS[c] for c in CRITERIA},
        "within_profile_dispersion": within,
        "leave_one_criterion_out": loco,
        "pairwise_margins": margins,
        "notes": {
            "measure": ("normalized weighted dispersion, not a share of "
                        "variance; it ignores covariance and is reported "
                        "alongside rank-based measures for that reason"),
            "unit": ("all statistics are computed within a profile, because "
                     "ranking happens among the roles offered to one student"),
            "loco": ("a criterion is held at its within-profile mean so that it "
                     "cannot discriminate; damage is measured against the full "
                     "ranking"),
        },
    }

    print_header("Experiment 8f - Criterion influence on the ranking")
    print(f"  {'criterion':<18}{'w':>6}{'within-sd':>11}{'disp share':>12}"
          f"{'tau (LOCO)':>12}{'top1 chg':>10}{'decides':>9}")
    for c in CRITERIA:
        l = loco[c]
        print(f"  {c:<18}{WEIGHTS[c]:>6.2f}"
              f"{within[c]['within_profile_sd_mean']:>11.4f}"
              f"{within[c]['weighted_dispersion_share_mean']:>12.3f}"
              f"{l['kendall_tau_mean']:>12.3f}"
              f"{l['top1_change_rate']:>10.3f}"
              f"{margins[c]['share_of_pairs_decided']:>9.3f}")
        print(f"      tie-aware: tau-b {l['kendall_tau_b_mean']:.3f}  "
              f"strict loss {l['strict_top_set_loss']:.4f}  "
              f"E[top-1] {l['expected_top1_change']:.3f} "
              f"[{l['expected_top1_ci95'][0]:.3f}, {l['expected_top1_ci95'][1]:.3f}]  "
              f"catalog-order {l['catalog_order_top1_min']:.3f}-{l['catalog_order_top1_max']:.3f}  "
              f"mean tied maxima {l['mean_tied_maxima_after_ablation']:.3f}")
    save_report("exp8f_criterion_influence", payload)
    return payload


if __name__ == "__main__":
    run()
