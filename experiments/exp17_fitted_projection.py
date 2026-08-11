"""How much occupational structure does the theory-guided projection give up?

Experiment 15 shows the designed matrix separates O*NET related-occupation
pairs worse than the raw Work Values it consumes, and no better than a random
matrix on the same support. That says the design does not help. It does not say
whether a 5-dimensional projection could help, or how much the specific
coefficients cost.

This experiment answers both by fitting a 5x6 matrix to the separation
objective itself and evaluating it on occupations held out from the fit.

  Split      the rated occupations are split in half by a seeded permutation.
             A pair enters the fit set only when both endpoints are in the fit
             half, and the evaluation set only when both are in the held-out
             half, so no occupation is seen on both sides.

  Objective  a smooth surrogate for the AUC: related pairs should have a
             higher cosine than random pairs. Optimized with L-BFGS-B from two
             starting points, the designed matrix and a fixed random draw, so
             the result cannot be an artifact of initialization.

  Reference  the raw 6-dimensional Work Values under the same cosine, which is
             what the pipeline would use with no projection at all. It is a
             reference and not a ceiling: a linear map changes the similarity
             measure rather than the information, so a fitted projection can
             separate better than its own input even though it cannot add
             information to it.

An unsupervised 5-dimensional reduction is included to separate two questions
that the designed matrix conflates: whether compressing six dimensions to five
costs anything, and whether these particular coefficients cost anything.

Every AUC here is exact (Mann-Whitney), computed over the held-out pairs only,
so the numbers are internally comparable but not identical to the resampled
full-sample figures of Experiment 15.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from careerstep.seeding import set_global_seeds
from data.loaders import (
    load_onet_interests,
    load_onet_related_occupations,
    load_onet_work_values_full,
)
from experiments._io import print_header, save_report

RIASEC = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising",
          "Conventional"]
ONET_WV_ELEMENTS = ["Achievement", "Independence", "Recognition",
                    "Relationships", "Support", "Working Conditions"]
DESIGNED_W = np.array([
    [0.60, 0.00, 0.40, 0.00, 0.00, 0.00],
    [0.00, -0.20, 0.00, 0.00, 0.50, 0.30],
    [0.00, 0.80, 0.00, 0.00, 0.00, 0.20],
    [0.50, 0.30, 0.00, 0.00, 0.00, 0.20],
    [0.00, 0.00, 0.00, 0.40, 0.20, 0.40],
])
SEED = 20260101


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _cos_pairs(X: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    U = _unit(X)
    return np.einsum("ij,ij->i", U[i], U[j])


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Exact AUC by rank statistic, ties counted as half."""
    r = stats.rankdata(np.concatenate([pos, neg]))
    u = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _fit(X: np.ndarray, pi, pj, ni, nj, w0: np.ndarray) -> np.ndarray:
    """Fit the projection on the fit-half pairs.

    The objective is the standardized mean difference between related-pair and
    random-pair cosines, which is smooth, costs one pass over the pairs, and is
    monotone in the AUC for well-behaved score distributions. Cosine is
    invariant to scaling the whole matrix, so the objective has one flat
    direction; L-BFGS-B tolerates it and the evaluation is unaffected.
    """
    def loss(flat):
        P = X @ flat.reshape(5, 6).T
        a = _cos_pairs(P, pi, pj)
        b = _cos_pairs(P, ni, nj)
        pooled = np.sqrt(0.5 * (a.var() + b.var()) + 1e-12)
        return float(-(a.mean() - b.mean()) / pooled)

    res = minimize(loss, w0.ravel(), method="L-BFGS-B",
                   options={"maxiter": 400, "ftol": 1e-12})
    return res.x.reshape(5, 6)


def run() -> dict:
    set_global_seeds()
    rng = np.random.default_rng(SEED)

    wv = load_onet_work_values_full()
    inte = load_onet_interests()
    rel = load_onet_related_occupations()

    wv_ex = wv[(wv["Scale ID"] == "EX") & (wv["Element Name"].isin(ONET_WV_ELEMENTS))]
    Wt = wv_ex.pivot_table(index="O*NET-SOC Code", columns="Element Name",
                           values="Data Value", aggfunc="first")[ONET_WV_ELEMENTS].dropna()
    ri = inte[(inte["Scale ID"] == "OI") & (inte["Element Name"].isin(RIASEC))]
    R = ri.pivot_table(index="O*NET-SOC Code", columns="Element Name",
                       values="Data Value", aggfunc="first")[RIASEC].dropna()

    socs = list(Wt.index.intersection(R.index))
    X = Wt.loc[socs].to_numpy(dtype=float) / 7.0
    pos_of = {s: i for i, s in enumerate(socs)}

    # Disjoint occupation split.
    perm = rng.permutation(len(socs))
    half = len(socs) // 2
    fit_set, held_set = set(perm[:half].tolist()), set(perm[half:].tolist())

    pairs = rel[rel["O*NET-SOC Code"].isin(pos_of)
                & rel["Related O*NET-SOC Code"].isin(pos_of)]
    ai = pairs["O*NET-SOC Code"].map(pos_of).to_numpy()
    aj = pairs["Related O*NET-SOC Code"].map(pos_of).to_numpy()
    keep = ai != aj
    ai, aj = ai[keep], aj[keep]

    def split_pairs(members):
        m = np.array([a in members and b in members for a, b in zip(ai, aj)])
        return ai[m], aj[m]

    fpi, fpj = split_pairs(fit_set)
    hpi, hpj = split_pairs(held_set)

    def random_pairs(members, n):
        idx = np.array(sorted(members))
        a = rng.choice(idx, size=n)
        b = rng.choice(idx, size=n)
        ok = a != b
        return a[ok], b[ok]

    fni, fnj = random_pairs(fit_set, len(fpi) * 2)
    hni, hnj = random_pairs(held_set, len(hpi) * 2)

    def held_auc(P: np.ndarray) -> float:
        return _auc(_cos_pairs(P, hpi, hpj), _cos_pairs(P, hni, hnj))

    def fit_half_auc(P: np.ndarray) -> float:
        return _auc(_cos_pairs(P, fpi, fpj), _cos_pairs(P, fni, fnj))

    # Baselines, all scored on the held-out pairs only.
    results = {
        "raw_work_values_6d": held_auc(X),
        "designed_W_5d": held_auc(X @ DESIGNED_W.T),
    }

    # An unsupervised 5-dimensional reduction, to separate "fitting helps" from
    # "any sensible 5-d map helps".
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    results["pca_5d"] = held_auc(X @ Vt[:5].T)

    # Fitted maps, from two starting points.
    fitted, fit_auc = {}, {"designed_W_5d": fit_half_auc(X @ DESIGNED_W.T)}
    for name, w0 in (("from_designed", DESIGNED_W),
                     ("from_random", rng.normal(size=(5, 6)) * 0.3)):
        Wf = _fit(X, fpi, fpj, fni, fnj, w0)
        fitted[name] = Wf
        results[f"fitted_5d_{name}"] = held_auc(X @ Wf.T)
        fit_auc[f"fitted_5d_{name}"] = fit_half_auc(X @ Wf.T)

    # Matched random null on the held-out pairs, for the same comparison
    # Experiment 15 makes on the full set.
    signs, support = np.sign(DESIGNED_W), np.sign(DESIGNED_W) != 0
    nulls = np.empty(2000, dtype=float)
    for i in range(len(nulls)):
        M = signs * np.abs(rng.normal(size=(5, 6))) * support
        M = M / np.maximum(np.abs(M).sum(axis=1, keepdims=True), 1e-12)
        nulls[i] = held_auc(X @ M.T)
    results["null_matched_mean"] = float(nulls.mean())

    reference = results["raw_work_values_6d"]
    designed = results["designed_W_5d"]
    best_fit = max(results["fitted_5d_from_designed"],
                   results["fitted_5d_from_random"])
    deltas = {k: float(results[k] - designed)
              for k in ("raw_work_values_6d", "pca_5d", "null_matched_mean",
                        "fitted_5d_from_designed", "fitted_5d_from_random")}

    payload = {
        "onet_release": "28.0",
        "n_occupations": len(socs),
        "split": {
            "n_fit_occupations": len(fit_set),
            "n_held_out_occupations": len(held_set),
            "n_fit_related_pairs": int(len(fpi)),
            "n_held_out_related_pairs": int(len(hpi)),
            "rule": ("a pair enters a side only when both endpoints are on that "
                     "side, so no occupation appears in both fit and evaluation"),
        },
        "held_out_auc": results,
        "fit_half_auc": fit_auc,
        "auc_minus_designed": deltas,
        "fitted_matrices": {k: v.tolist() for k, v in fitted.items()},
        "notes": {
            "estimator": ("exact Mann-Whitney AUC over held-out pairs; Experiment "
                          "15 reports a resampled estimate over all pairs, so the "
                          "two are comparable in ordering but not digit for digit"),
            "reference": ("the raw 6-dimensional Work Values under the same cosine "
                          "are the no-projection reference, not an upper bound: a "
                          "linear map changes the similarity measure, so a fitted "
                          "projection can separate better than its own input "
                          "without adding information to it"),
            "generalization": ("fit_half_auc is reported beside held_out_auc so the "
                               "gap between them is visible"),
            "claim": ("fitting shows what the representation could carry, not that "
                      "a fitted projection is a valid measure of a student's work "
                      "values or of role fit"),
        },
    }

    print_header("Experiment 17 - A projection fitted to external structure")
    s = payload["split"]
    print(f"  occupations {len(socs)} split {s['n_fit_occupations']}/"
          f"{s['n_held_out_occupations']}; related pairs "
          f"{s['n_fit_related_pairs']} fit / {s['n_held_out_related_pairs']} held out")
    for k in ("raw_work_values_6d", "designed_W_5d", "null_matched_mean",
              "pca_5d", "fitted_5d_from_designed", "fitted_5d_from_random"):
        print(f"  {k:<28} held-out AUC = {results[k]:.3f}")
    for k, v in deltas.items():
        print(f"  {k:<28} designed{v:+.3f}")
    print(f"  best fitted held-out = {best_fit:.3f}, no-projection "
          f"reference = {reference:.3f}")
    for k, v in fit_auc.items():
        print(f"  fit-half AUC {k:<24} = {v:.3f}")
    save_report("exp17_fitted_projection", payload)
    return payload


if __name__ == "__main__":
    run()
