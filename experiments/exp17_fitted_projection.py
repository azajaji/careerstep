"""How much occupational structure does the theory-guided projection give up?

Experiment 15 shows the designed matrix separates O*NET related-occupation
pairs no better than a random matrix on the same support. That says the design
does not help. It does not say whether a 5-dimensional linear map could help,
or how much the specific coefficients cost.

This experiment answers both by fitting a 5x6 matrix to the separation
objective and evaluating it on occupations held out from the fit. It is a
capacity diagnostic, not a repair of the CSMQ mapping: the fit is
unconstrained, so it does not preserve the designed sparsity, signs, row
normalization, or the production clipping step, and its five output axes carry
no CSMQ meaning. Treat the fitted number as an upper reference for what a
linear map of these ratings can do, and nothing more.

  Splits     the rated occupations are split in half repeatedly by seeded
             permutations. A pair enters a side only when both endpoints are on
             that side, so no occupation is seen in both fit and evaluation.
             Every quantity is reported as a mean over splits with a percentile
             interval, because a single split understates uncertainty.

  Negatives  random unordered pairs that are not related pairs, drawn afresh
             within each split's evaluation half.

  Objective  the standardized mean difference between related-pair and
             random-pair cosines, a smooth surrogate for the AUC, optimized
             with L-BFGS-B from two starting points.

  Reference  the raw 6-dimensional Work Values under the same cosine, which is
             what the pipeline would use with no projection. It is a reference
             and not a ceiling: a linear map changes the similarity measure
             rather than the information, so a fitted projection can separate
             better than its own input without adding information to it.

Every AUC is exact (Mann-Whitney).
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
N_SPLITS = 20
N_NULL_PER_SPLIT = 200


def _unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _cos(X, i, j):
    U = _unit(X)
    return np.einsum("ij,ij->i", U[i], U[j])


def _auc(pos, neg):
    r = stats.rankdata(np.concatenate([pos, neg]))
    u = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _fit(X, pi, pj, ni, nj, w0):
    def loss(flat):
        P = X @ flat.reshape(5, 6).T
        a, b = _cos(P, pi, pj), _cos(P, ni, nj)
        pooled = np.sqrt(0.5 * (a.var() + b.var()) + 1e-12)
        return float(-(a.mean() - b.mean()) / pooled)
    res = minimize(loss, w0.ravel(), method="L-BFGS-B",
                   options={"maxiter": 400, "ftol": 1e-12})
    return res.x.reshape(5, 6)


def _summary(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "ci95_low": float(np.percentile(a, 2.5)),
            "ci95_high": float(np.percentile(a, 97.5)),
            "n_splits": int(len(a))}


def run() -> dict:
    set_global_seeds()

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
    # O*NET Extent (EX) runs 1..7; min-max to [0, 1] is (x - 1) / 6.
    X = (Wt.loc[socs].to_numpy(dtype=float) - 1.0) / 6.0
    pos_of = {s: i for i, s in enumerate(socs)}

    pairs = rel[rel["O*NET-SOC Code"].isin(pos_of)
                & rel["Related O*NET-SOC Code"].isin(pos_of)]
    a_ = pairs["O*NET-SOC Code"].map(pos_of).to_numpy()
    b_ = pairs["Related O*NET-SOC Code"].map(pos_of).to_numpy()
    keep = a_ != b_
    canon = np.unique(np.stack([np.minimum(a_[keep], b_[keep]),
                                np.maximum(a_[keep], b_[keep])], 1), axis=0)
    ai, aj = canon[:, 0], canon[:, 1]
    positive = set(map(tuple, canon.tolist()))

    signs, support = np.sign(DESIGNED_W), np.sign(DESIGNED_W) != 0

    acc = {k: [] for k in ("raw_work_values_6d", "designed_W_5d", "pca_5d",
                           "fitted_5d_from_designed", "fitted_5d_from_random",
                           "null_matched_mean")}
    fit_half = {k: [] for k in ("designed_W_5d", "fitted_5d_from_designed")}
    n_fit_pairs, n_held_pairs = [], []

    for sp in range(N_SPLITS):
        rng = np.random.default_rng(SEED + sp)
        perm = rng.permutation(len(socs))
        half = len(socs) // 2
        fit_set, held_set = set(perm[:half].tolist()), set(perm[half:].tolist())

        def side(members):
            m = np.array([u in members and v in members for u, v in zip(ai, aj)])
            return ai[m], aj[m]

        fpi, fpj = side(fit_set)
        hpi, hpj = side(held_set)
        n_fit_pairs.append(len(fpi))
        n_held_pairs.append(len(hpi))

        def negatives(members, need):
            idx = np.array(sorted(members))
            out = set()
            while len(out) < need:
                u = rng.choice(idx, size=2 * need)
                v = rng.choice(idx, size=2 * need)
                for p, q in zip(u.tolist(), v.tolist()):
                    if p == q:
                        continue
                    key = (min(p, q), max(p, q))
                    if key in positive or key in out:
                        continue
                    out.add(key)
                    if len(out) >= need:
                        break
            arr = np.array(sorted(out))
            return arr[:, 0], arr[:, 1]

        fni, fnj = negatives(fit_set, len(fpi))
        hni, hnj = negatives(held_set, len(hpi))

        held = lambda P: _auc(_cos(P, hpi, hpj), _cos(P, hni, hnj))
        fitl = lambda P: _auc(_cos(P, fpi, fpj), _cos(P, fni, fnj))

        # PCA is a fitted representation, so its basis and its centering both
        # come from the fit half. Two things matter here. Taking the SVD over
        # all occupations would let the held-out half inform the basis it is
        # scored on. And PCA scores are (X - mu) V^T, not X V^T: cosine is
        # translation-sensitive, so omitting the mean does not give a PCA
        # transform at all.
        Xf = X[np.array(sorted(fit_set))]
        mu = Xf.mean(axis=0)
        _, _, Vtf = np.linalg.svd(Xf - mu, full_matrices=False)
        PCA5 = Vtf[:5]

        acc["raw_work_values_6d"].append(held(X))
        acc["designed_W_5d"].append(held(X @ DESIGNED_W.T))
        acc["pca_5d"].append(held((X - mu) @ PCA5.T))
        fit_half["designed_W_5d"].append(fitl(X @ DESIGNED_W.T))

        for name, w0 in (("fitted_5d_from_designed", DESIGNED_W),
                         ("fitted_5d_from_random", rng.normal(size=(5, 6)) * 0.3)):
            Wf = _fit(X, fpi, fpj, fni, fnj, w0)
            acc[name].append(held(X @ Wf.T))
            if name == "fitted_5d_from_designed":
                fit_half[name].append(fitl(X @ Wf.T))

        nulls = []
        for _ in range(N_NULL_PER_SPLIT):
            M = signs * np.abs(rng.normal(size=(5, 6))) * support
            M = M / np.maximum(np.abs(M).sum(axis=1, keepdims=True), 1e-12)
            nulls.append(held(X @ M.T))
        acc["null_matched_mean"].append(float(np.mean(nulls)))

    summ = {k: _summary(v) for k, v in acc.items()}
    gap = _summary(np.asarray(acc["fitted_5d_from_designed"])
                   - np.asarray(acc["designed_W_5d"]))
    pca_gap = _summary(np.asarray(acc["pca_5d"]) - np.asarray(acc["designed_W_5d"]))

    payload = {
        "onet_release": "28.0",
        "n_occupations": len(socs),
        "n_splits": N_SPLITS,
        "split": {
            "rule": ("a pair enters a side only when both endpoints are on that "
                     "side; splits are repeated so the interval reflects split "
                     "variation rather than one draw"),
            "mean_fit_related_pairs": float(np.mean(n_fit_pairs)),
            "mean_held_out_related_pairs": float(np.mean(n_held_pairs)),
            "negatives": "unordered non-related pairs drawn within each half",
        },
        "held_out_auc": summ,
        "fit_half_auc": {k: _summary(v) for k, v in fit_half.items()},
        "paired_gap_fitted_minus_designed": gap,
        "paired_gap_pca_minus_designed": pca_gap,
        "notes": {
            "status": ("capacity diagnostic, not a repair: the fit is "
                       "unconstrained, discards the designed sparsity, signs and "
                       "row normalization, omits the production clipping step, "
                       "and its axes carry no CSMQ interpretation"),
            "estimator": "exact Mann-Whitney AUC on held-out pairs",
            "reference": ("the raw 6-dimensional Work Values under the same "
                          "cosine are the no-projection reference, not an upper "
                          "bound"),
        },
    }

    print_header("Experiment 17 - Fitted-projection capacity diagnostic")
    print(f"  {len(socs)} occupations, {N_SPLITS} repeated half-splits; "
          f"mean held-out related pairs {np.mean(n_held_pairs):.0f}")
    for k in ("raw_work_values_6d", "designed_W_5d", "null_matched_mean",
              "pca_5d", "fitted_5d_from_designed", "fitted_5d_from_random"):
        s = summ[k]
        print(f"  {k:<28} {s['mean']:.3f}  [{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]")
    print(f"  paired gap fitted-designed  {gap['mean']:+.3f} "
          f"[{gap['ci95_low']:+.3f}, {gap['ci95_high']:+.3f}]")
    print(f"  paired gap PCA-designed     {pca_gap['mean']:+.3f} "
          f"[{pca_gap['ci95_low']:+.3f}, {pca_gap['ci95_high']:+.3f}]")
    save_report("exp17_fitted_projection", payload)
    return payload


if __name__ == "__main__":
    run()
