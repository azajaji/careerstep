"""Does calibrating the criterion scales repair the scorer?

Experiment 8f shows the nominal coefficients do not describe influence, because
the three criteria occupy incomparable ranges. The obvious repair is to put
them on a common scale before weighting. This experiment applies that repair
and measures what it fixes and what it does not.

Two calibrations are tested, both computed within a profile's candidate set,
which is the set the ranking is over:

  z      standardize each criterion to zero mean and unit variance
  range  rescale each criterion to [0, 1] by its observed min and max

Under either one every criterion has the same spread, so the normalized
weighted dispersion of Experiment 8f collapses to the weights themselves. That
is arithmetic, not a finding, and it is reported only as a check that the
calibration does what it claims. The measured questions are the two that
arithmetic does not settle:

A. Does realized influence follow the nominal weights once the scales match?
   Re-run the leave-one-criterion-out and pairwise-margin measures on the
   calibrated composite. Weight fidelity is the point of the repair.

B. Does the repaired scorer rank better? It hands the largest share of the
   ordering to value-fit, and Experiment 15 finds that the representation
   behind value-fit carries no occupational structure a random matrix on the
   same support would not. The internal benchmark cannot settle validity,
   because its acceptable-role labels are built from the same orientation
   centroids the value-fit ranker scores with, so it is biased in favour of a
   value-heavy ranker. Both the recovery numbers and that bias are reported.

The two calibrations also carry a cost worth naming: normalizing within the
candidate set makes a role's score depend on which other roles are offered, so
adding or removing a candidate can reorder the rest. The rank-reversal rate
under candidate dropout is measured for that reason.
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
from scipy import stats

from careerstep.career_positioning import RoleRecommender, _cosine
from careerstep.career_positioning_benchmark import (
    FEASIBILITY, W_FEAS, W_SKILL, W_VALUE,
    lexical_skill_readiness, mrr, ndcg_at_k, recall_at_k,
    role_to_required_skills, simulate_profiles,
)
from careerstep.seeding import load_seeds, set_global_seeds
from careerstep.benchmark_cohort import build_cohort, cohort_fingerprint
from data.loaders import load_onet_work_values, load_saudi_cyber_roles
from experiments._io import print_header, save_report

CRITERIA = ("value_fit", "skill_readiness", "feasibility")
WEIGHTS = {"value_fit": W_VALUE, "skill_readiness": W_SKILL, "feasibility": W_FEAS}
N_DROPOUT_TRIALS = 20


def _calibrate(X: np.ndarray, mode: str) -> np.ndarray:
    """Put the columns of one profile's criterion matrix on a common scale."""
    if mode == "none":
        return X
    if mode == "z":
        sd = X.std(axis=0, ddof=1)
        return (X - X.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)
    if mode == "range":
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = hi - lo
        return (X - lo) / np.where(span > 1e-12, span, 1.0)
    raise ValueError(mode)


def _influence(per_profile: List[np.ndarray], w: np.ndarray, mode: str) -> dict:
    """Experiment 8f's measures, recomputed on the calibrated composite."""
    def rank(X):
        return np.argsort(-(X @ w), kind="stable")

    shares, taus, top1, ov5 = [], [[] for _ in CRITERIA], [[] for _ in CRITERIA], \
        [[] for _ in CRITERIA]
    decisive = np.zeros(len(CRITERIA))
    n_pairs = 0
    for X0 in per_profile:
        X = _calibrate(X0, mode)
        sd = X.std(axis=0, ddof=1)
        ws = w * sd
        shares.append(ws / ws.sum() if ws.sum() > 0 else np.zeros_like(ws))

        full = rank(X)
        pos_full = np.empty(len(full), dtype=int)
        pos_full[full] = np.arange(len(full))
        for j in range(len(CRITERIA)):
            Y = X.copy()
            Y[:, j] = Y[:, j].mean()
            got = rank(Y)
            pos_got = np.empty(len(got), dtype=int)
            pos_got[got] = np.arange(len(got))
            taus[j].append(stats.kendalltau(pos_full, pos_got).statistic)
            top1[j].append(0.0 if full[0] == got[0] else 1.0)
            a, b = set(full[:5].tolist()), set(got[:5].tolist())
            ov5[j].append(len(a & b) / len(a | b))

        d = X[:, None, :] - X[None, :, :]
        iu = np.triu_indices(X.shape[0], k=1)
        diffs = d[iu] * w
        total = diffs.sum(axis=1)
        for j in range(len(CRITERIA)):
            others = np.delete(diffs, j, axis=1).sum(axis=1)
            decisive[j] += np.sum((np.sign(diffs[:, j]) == np.sign(total))
                                  & (np.abs(diffs[:, j]) > np.abs(others)))
        n_pairs += len(total)

    shares = np.asarray(shares)
    return {
        c: {
            "weighted_dispersion_share_mean": float(shares[:, j].mean()),
            "kendall_tau_mean": float(np.mean(taus[j])),
            "top1_change_rate": float(np.mean(top1[j])),
            "top5_jaccard_mean": float(np.mean(ov5[j])),
            "share_of_pairs_decided": float(decisive[j] / n_pairs),
        }
        for j, c in enumerate(CRITERIA)
    }


def _recovery(per_profile, profiles, role_ids, w, mode) -> dict:
    """Hit-rate, MRR and nDCG of the calibrated composite on the benchmark.

    Every condition uses the same stable tie-break, so the calibration is the
    only thing that varies. The shipped benchmark leaves ties to an unstable
    sort; the uncalibrated row here therefore differs from the composite row of
    the ranking table by one or two profiles out of 120, which is the size of
    the tie-breaking ambiguity rather than a disagreement.
    """
    r1 = r3 = r5 = m = nd = 0.0
    for X0, p in zip(per_profile, profiles):
        order = np.argsort(-(_calibrate(X0, mode) @ w), kind="stable")
        ranked = [role_ids[i] for i in order]
        acc = p.acceptable_role_ids
        r1 += recall_at_k(ranked, acc, 1) > 0
        r3 += recall_at_k(ranked, acc, 3) > 0
        r5 += recall_at_k(ranked, acc, 5) > 0
        m += mrr(ranked, acc)
        nd += ndcg_at_k(ranked, acc, 5)
    n = len(profiles)
    return {"hit_rate_at_1": r1 / n, "hit_rate_at_3": r3 / n,
            "hit_rate_at_5": r5 / n, "mrr": m / n, "ndcg_at_5": nd / n}


def _dropout_reversals(per_profile, w, mode, rng) -> float:
    """How often removing one non-leading candidate changes the leading role.

    Under an uncalibrated additive score this is zero by construction: each
    role's score is independent of the others. Calibrating within the candidate
    set gives that independence up, and this measures the price.
    """
    changed = total = 0
    for X0 in per_profile:
        base = np.argsort(-(_calibrate(X0, mode) @ w), kind="stable")
        leader = base[0]
        for _ in range(N_DROPOUT_TRIALS):
            drop = int(rng.integers(0, X0.shape[0]))
            if drop == leader:
                continue
            keep = [i for i in range(X0.shape[0]) if i != drop]
            sub = np.argsort(-(_calibrate(X0[keep], mode) @ w), kind="stable")
            changed += keep[sub[0]] != leader
            total += 1
    return float(changed / total) if total else 0.0


def run() -> dict:
    set_global_seeds()
    seeds = load_seeds()
    roles_df = load_saudi_cyber_roles()
    rec = RoleRecommender(roles_df, load_onet_work_values())
    required = role_to_required_skills(roles_df)
    metas = roles_df.set_index("role_id")
    role_ids = list(rec.centroid_df.index)

    profiles = build_cohort(rec, roles_df)

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
    rng = np.random.default_rng(seeds["numpy_seed"])

    influence: Dict[str, dict] = {}
    recovery: Dict[str, dict] = {}
    reversal: Dict[str, float] = {}
    for mode in ("none", "z", "range"):
        influence[mode] = _influence(per_profile, w, mode)
        recovery[mode] = _recovery(per_profile, profiles, role_ids, w, mode)
        reversal[mode] = _dropout_reversals(per_profile, w, mode, rng)

    # How far the calibrated ranking moves from the shipped one.
    moved = {}
    for mode in ("z", "range"):
        top1, jac = [], []
        for X0 in per_profile:
            a = np.argsort(-(_calibrate(X0, "none") @ w), kind="stable")
            b = np.argsort(-(_calibrate(X0, mode) @ w), kind="stable")
            top1.append(0.0 if a[0] == b[0] else 1.0)
            sa, sb = set(a[:5].tolist()), set(b[:5].tolist())
            jac.append(len(sa & sb) / len(sa | sb))
        moved[mode] = {"top1_change_vs_uncalibrated": float(np.mean(top1)),
                       "top5_jaccard_vs_uncalibrated": float(np.mean(jac))}

    payload = {
        "n_profiles": len(profiles),
        "n_roles": len(role_ids),
        "weights": {c: WEIGHTS[c] for c in CRITERIA},
        "influence_by_calibration": influence,
        "recovery_by_calibration": recovery,
        "ranking_shift_vs_uncalibrated": moved,
        "dropout_reversal_rate": reversal,
        "notes": {
            "arithmetic": ("under either calibration every criterion has the same "
                           "within-profile spread, so the normalized weighted "
                           "dispersion equals the weights by construction; it is "
                           "reported as a check, not as a result"),
            "measured": ("the leave-one-criterion-out and margin measures are not "
                         "fixed by construction and are the test of weight fidelity"),
            "benchmark_bias": ("acceptable-role labels are the latent role and its "
                               "two nearest orientation-centroid neighbours, so the "
                               "benchmark favours a value-heavy ranker; recovery "
                               "numbers here bound nothing about role-fit validity"),
            "cost": ("calibrating within the candidate set makes a role's score "
                     "depend on the other candidates, so dropping one candidate can "
                     "change the leader; the uncalibrated score cannot do this"),
        },
    }

    print_header("Experiment 16 - Scale calibration of the composite")
    for mode in ("none", "z", "range"):
        print(f"  calibration = {mode}")
        print(f"    {'criterion':<18}{'w':>6}{'disp':>8}{'tau':>8}"
              f"{'top1 chg':>10}{'decides':>9}")
        for c in CRITERIA:
            d = influence[mode][c]
            print(f"    {c:<18}{WEIGHTS[c]:>6.2f}"
                  f"{d['weighted_dispersion_share_mean']:>8.3f}"
                  f"{d['kendall_tau_mean']:>8.3f}"
                  f"{d['top1_change_rate']:>10.3f}"
                  f"{d['share_of_pairs_decided']:>9.3f}")
        r = recovery[mode]
        print(f"    recovery: hit@1={r['hit_rate_at_1']:.3f} "
              f"hit@3={r['hit_rate_at_3']:.3f} hit@5={r['hit_rate_at_5']:.3f} "
              f"MRR={r['mrr']:.3f} nDCG@5={r['ndcg_at_5']:.3f}")
        print(f"    dropout reversal rate: {reversal[mode]:.3f}")
    for mode, d in moved.items():
        print(f"  {mode} vs uncalibrated: top-1 changes "
              f"{d['top1_change_vs_uncalibrated']:.3f}, "
              f"top-5 Jaccard {d['top5_jaccard_vs_uncalibrated']:.3f}")
    save_report("exp16_scale_calibration", payload)
    return payload


if __name__ == "__main__":
    run()
