"""External structure checks on the CSMQ projection, using public O*NET data only.

Every other experiment runs on the 25-role cybersecurity catalog or on corpora
the study team constructed. This one uses labels O*NET produced, over every
occupation O*NET rates, and asks four questions that need no participant.

A. Generality of the compression. Section VI-B shows the value-fit cosine is
   nearly constant across the 25-role catalog. Projecting every rated O*NET
   occupation through the same fixed matrix tests whether that is a property of
   the projection or of the catalog.

B. External structure. O*NET publishes curated related-occupation lists. If the
   projection preserves occupational structure, related pairs should sit closer
   in CSMQ space than random pairs. Reported as an AUC.

C. The controls that make B interpretable. Chance is the wrong baseline. The
   informative comparisons are the unprojected Work Values the projection
   consumes, and random matrices of the same shape. Without those, an AUC above
   0.5 says only that the input carried some structure, not that the projection
   contributed any.

D. Convergent agreement. O*NET rates Interests (RIASEC) on the same
   occupations, independently of Work Values.

The projection is applied through ``project_work_values_to_csmq`` so the
experiment exercises the shipped code path rather than a reimplementation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from careerstep.career_positioning import project_work_values_to_csmq
from careerstep.seeding import set_global_seeds
from data.loaders import (
    load_onet_interests,
    load_onet_related_occupations,
    load_onet_work_values_full,
)
from experiments._io import print_header, save_report

RIASEC = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising",
          "Conventional"]
# O*NET prints Work-Value element names in title case; the framework keys them
# in snake case. Listed in the framework's key order so the projection receives
# its six dimensions the right way round.
ONET_WV_ELEMENTS = ["Achievement", "Independence", "Recognition",
                    "Relationships", "Support", "Working Conditions"]
N_RANDOM_W = 10_000
SEED = 20260101


def _cosine_matrix(X: np.ndarray) -> np.ndarray:
    n = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return n @ n.T


def _centered(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=1, keepdims=True)


def run() -> dict:
    set_global_seeds()
    rng = np.random.default_rng(SEED)

    wv = load_onet_work_values_full()
    inte = load_onet_interests()
    rel = load_onet_related_occupations()

    wv_ex = wv[(wv["Scale ID"] == "EX") & (wv["Element Name"].isin(ONET_WV_ELEMENTS))]
    W = wv_ex.pivot_table(index="O*NET-SOC Code", columns="Element Name",
                          values="Data Value", aggfunc="first")
    W = W[ONET_WV_ELEMENTS].dropna()

    ri = inte[(inte["Scale ID"] == "OI") & (inte["Element Name"].isin(RIASEC))]
    R = ri.pivot_table(index="O*NET-SOC Code", columns="Element Name",
                       values="Data Value", aggfunc="first")[RIASEC].dropna()

    socs = list(W.index.intersection(R.index))
    # O*NET Extent (EX) runs 1..7; min-max to [0, 1] is (x - 1) / 6.
    raw_wv = (W.loc[socs].to_numpy(dtype=float) - 1.0) / 6.0
    C = np.array([
        project_work_values_to_csmq(W.loc[s].to_numpy(dtype=float)).as_vector()
        for s in socs
    ], dtype=float)
    Rv = R.loc[socs].to_numpy(dtype=float)

    Ccos = _cosine_matrix(C)
    iu = np.triu_indices(len(socs), k=1)
    raw = Ccos[iu]
    Ccen_cos = _cosine_matrix(_centered(C))
    cen = Ccen_cos[iu]

    compression = {
        "n_occupations": len(socs),
        "n_pairs": int(len(raw)),
        "csmq_entry_min": float(C.min()),
        "csmq_entry_max": float(C.max()),
        "raw_cosine_mean": float(raw.mean()),
        "raw_cosine_sd": float(raw.std(ddof=1)),
        "raw_cosine_min": float(raw.min()),
        "share_pairs_above_0_95": float((raw > 0.95).mean()),
        "centered_cosine_mean": float(cen.mean()),
        "centered_cosine_sd": float(cen.std(ddof=1)),
        "centered_cosine_min": float(cen.min()),
        "sd_ratio_centered_over_raw": float(cen.std(ddof=1) / raw.std(ddof=1)),
    }

    pos_of = {s: i for i, s in enumerate(socs)}
    pairs = rel[rel["O*NET-SOC Code"].isin(pos_of)
                & rel["Related O*NET-SOC Code"].isin(pos_of)]
    a_ = pairs["O*NET-SOC Code"].map(pos_of).to_numpy()
    b_ = pairs["Related O*NET-SOC Code"].map(pos_of).to_numpy()
    keep = a_ != b_
    a_, b_ = a_[keep], b_[keep]

    # O*NET lists relatedness directionally and repeats it, so canonicalize
    # each pair and keep it once.
    canon = np.unique(np.stack([np.minimum(a_, b_), np.maximum(a_, b_)], 1), axis=0)
    pi, pj = canon[:, 0], canon[:, 1]
    positive = set(map(tuple, canon.tolist()))

    # Negatives are unordered pairs that are not related pairs. Drawing without
    # this exclusion puts positives in the negative set and understates every
    # representation's separation.
    need = len(pi)

    def draw_negatives(gen):
        """One balanced sample of unlisted pairs, excluding every listed pair."""
        neg = set()
        while len(neg) < need:
            x = gen.integers(0, len(socs), size=2 * need)
            y = gen.integers(0, len(socs), size=2 * need)
            for u, v in zip(x.tolist(), y.tolist()):
                if u == v:
                    continue
                key = (min(u, v), max(u, v))
                if key in positive or key in neg:
                    continue
                neg.add(key)
                if len(neg) >= need:
                    break
        arr = np.array(sorted(neg))
        return arr[:, 0], arr[:, 1]

    ni_, nj_ = draw_negatives(rng)

    def _auc_from(a: np.ndarray, b: np.ndarray) -> float:
        """Exact Mann-Whitney AUC; ties count as half."""
        r = stats.rankdata(np.concatenate([a, b]))
        u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0
        return float(u / (len(a) * len(b)))

    def auc(M: np.ndarray) -> float:
        return _auc_from(M[pi, pj], M[ni_, nj_])

    def _auc_weighted(a, wa, b, wb) -> float:
        """Mann-Whitney AUC with pair weights; ties count as half.

        Sorting the negatives once and taking prefix sums makes this O(n log n)
        rather than the O(|a||b|) of an explicit pairwise comparison, which
        matters because the vertex bootstrap runs it several hundred times.
        """
        order = np.argsort(b, kind="stable")
        bs, wbs = b[order], wb[order]
        cum = np.concatenate([[0.0], np.cumsum(wbs)])
        lo = np.searchsorted(bs, a, side="left")
        hi = np.searchsorted(bs, a, side="right")
        less = cum[lo]
        equal = cum[hi] - cum[lo]
        num = float(np.sum(wa * (less + 0.5 * equal)))
        den = float(wa.sum() * wb.sum())
        return num / den if den > 0 else float("nan")

    def vertex_bootstrap(mats: dict, draws: int = 400) -> dict:
        """Vertex bootstrap that preserves draw multiplicity.

        Occupations are drawn with replacement and a pair carries weight
        m_i * m_j, so an occupation drawn several times counts several times.
        Collapsing multiplicity to presence, as a simpler mask would, discards
        that weight and narrows the spread. All representations are evaluated on
        the same draw, so differences between them are paired within draw.
        """
        boot_rng = np.random.default_rng(SEED)
        n = len(socs)
        keys = list(mats)
        per = {k: [] for k in keys}
        paired = {k: [] for k in keys if k != "raw_work_values"}
        for _ in range(draws):
            m = np.bincount(boot_rng.integers(0, n, size=n), minlength=n).astype(float)
            wp, wn = m[pi] * m[pj], m[ni_] * m[nj_]
            if (wp > 0).sum() < 50 or (wn > 0).sum() < 50:
                continue
            vals = {k: _auc_weighted(M[pi, pj], wp, M[ni_, nj_], wn)
                    for k, M in mats.items()}
            for k, v in vals.items():
                per[k].append(v)
            for k in paired:
                paired[k].append(vals["raw_work_values"] - vals[k])

        def pct(v):
            a = np.asarray(v, dtype=float)
            return {"mean": float(a.mean()),
                    "ci95_low": float(np.percentile(a, 2.5)),
                    "ci95_high": float(np.percentile(a, 97.5)),
                    "n_draws": int(a.size)}

        return {"marginal": {k: pct(v) for k, v in per.items()},
                "paired_raw_minus": {k: pct(v) for k, v in paired.items()}}

    def auc_clustered_ci(M: np.ndarray, draws: int = 400) -> list:
        """Occupation-clustered bootstrap: resample occupations, keep the pairs
        whose endpoints both survive. Pairs sharing an endpoint are not
        independent, so resampling pairs directly would be too narrow."""
        boot_rng = np.random.default_rng(SEED)
        n = len(socs)
        vals = []
        for _ in range(draws):
            take = boot_rng.integers(0, n, size=n)
            mult = np.bincount(take, minlength=n)
            keep_p = (mult[pi] > 0) & (mult[pj] > 0)
            keep_n = (mult[ni_] > 0) & (mult[nj_] > 0)
            if keep_p.sum() < 50 or keep_n.sum() < 50:
                continue
            vals.append(_auc_from(M[pi[keep_p], pj[keep_p]],
                                  M[ni_[keep_n], nj_[keep_n]]))
        v = np.asarray(vals, dtype=float)
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    representations = {
        "raw_work_values": raw_wv,
        "projected_csmq": C,
        "riasec_interests": Rv,
    }
    separation = {"n_related_pairs_canonical": int(len(pi)),
                  "n_negative_pairs": int(len(ni_)),
                  "negatives_exclude_positives": True,
                  "auc_estimator": "exact Mann-Whitney"}
    for name, X in representations.items():
        Mc = _cosine_matrix(X)
        separation[name] = {
            "dimensions": int(X.shape[1]),
            "auc_cosine": auc(Mc),
            "auc_cosine_presence_mask_range": auc_clustered_ci(Mc),
            "auc_centered_cosine": auc(_cosine_matrix(_centered(X))),
        }

    # The projection is compared with the raw ratings on the same pairs, so the
    # contrast is paired. Separate marginal ranges do not answer whether the
    # projection loses structure; the within-draw difference does.
    separation["vertex_bootstrap"] = vertex_bootstrap(
        {k: _cosine_matrix(X) for k, X in representations.items()})

    # The unlisted comparison pairs are themselves a sample. Holding one draw
    # fixed would hide how much of each AUC depends on which unlisted pairs
    # happened to be selected, so the whole comparison is repeated over fresh
    # draws and the spread is reported.
    def negative_draw_sensitivity(mats: dict, draws: int = 20) -> dict:
        gen = np.random.default_rng(SEED + 991)
        per = {k: [] for k in mats}
        gaps = []
        for _ in range(draws):
            qi, qj = draw_negatives(gen)
            vals = {k: _auc_from(M[pi, pj], M[qi, qj]) for k, M in mats.items()}
            for k, v in vals.items():
                per[k].append(v)
            gaps.append(vals["raw_work_values"] - vals["projected_csmq"])

        def summ(v):
            a = np.asarray(v, dtype=float)
            return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                    "min": float(a.min()), "max": float(a.max()),
                    "n_draws": int(a.size)}

        return {"per_representation": {k: summ(v) for k, v in per.items()},
                "raw_minus_projected": summ(gaps)}

    separation["negative_draw_sensitivity"] = negative_draw_sensitivity(
        {k: _cosine_matrix(X) for k, X in representations.items()})

    # Null distributions. "Arbitrary" needs defining, so three are reported,
    # and together they separate the two things a hand-built matrix asserts:
    # which work-value dimensions feed an orientation, and how much each
    # contributes.
    #
    #   dense_normal   every entry ~ N(0,1), rows scaled to unit L1 norm,
    #                  signs free. Asks whether the design matters at all.
    #   matched        the designed zero pattern and signs held fixed, only the
    #                  magnitudes redrawn. Asks whether the coefficients matter
    #                  given the structure.
    #   permuted       the designed magnitudes and signs kept exactly, but the
    #                  positions permuted within each row, so each orientation
    #                  keeps its number of positive and negative loadings and
    #                  their sizes while the work-value dimensions they land on
    #                  are arbitrary. Asks whether the conceptual assignment of
    #                  dimensions to orientations carries the structure.
    designed_W = np.array([
        [0.60, 0.00, 0.40, 0.00, 0.00, 0.00],
        [0.00, -0.20, 0.00, 0.00, 0.50, 0.30],
        [0.00, 0.80, 0.00, 0.00, 0.00, 0.20],
        [0.50, 0.30, 0.00, 0.00, 0.00, 0.20],
        [0.00, 0.00, 0.00, 0.40, 0.20, 0.40],
    ])
    signs = np.sign(designed_W)
    support = signs != 0
    designed = separation["projected_csmq"]["auc_centered_cosine"]

    def l1_rows(M):
        return M / np.maximum(np.abs(M).sum(axis=1, keepdims=True), 1e-12)

    def permute_rows(M):
        """Keep each row's multiset of coefficients, move where they sit."""
        out = np.empty_like(M)
        for r in range(M.shape[0]):
            out[r] = M[r][rng.permutation(M.shape[1])]
        return out

    for name, draw in (
        ("dense_normal", lambda: l1_rows(rng.normal(size=(5, 6)))),
        ("matched_support_and_sign",
         lambda: l1_rows(signs * np.abs(rng.normal(size=(5, 6))) * support)),
        ("permuted_support", lambda: permute_rows(designed_W)),
    ):
        vals = np.empty(N_RANDOM_W, dtype=float)
        clipped = np.empty(N_RANDOM_W, dtype=float)
        for i in range(N_RANDOM_W):
            Praw = raw_wv @ draw().T
            Pm = np.clip(Praw, 0.0, 1.0)
            # Clipping is part of the production projector, so the null gets it
            # too; recording how often it binds says whether it is doing work.
            clipped[i] = float(np.mean(Praw != Pm))
            vals[i] = auc(_cosine_matrix(_centered(Pm)))
        # One-sided empirical p-value with the standard +1 correction.
        p = float((np.sum(vals >= designed) + 1) / (N_RANDOM_W + 1))
        separation[f"null_{name}"] = {
            "n_draws": int(N_RANDOM_W),
            "sampling": {
                "dense_normal":
                    "entries ~ N(0,1), rows scaled to unit L1 norm, signs free",
                "matched_support_and_sign":
                    "designed zero pattern and signs held fixed, magnitudes "
                    "~ |N(0,1)|, rows scaled to unit L1 norm",
                "permuted_support":
                    "the designed coefficients of each row permuted across the "
                    "six work-value dimensions; sparsity, signs, magnitudes and "
                    "row norm are exactly the designed ones",
            }[name],
            "auc_mean": float(vals.mean()),
            "auc_sd": float(vals.std(ddof=1)),
            "auc_p5": float(np.percentile(vals, 5)),
            "auc_p95": float(np.percentile(vals, 95)),
            "designed_auc": float(designed),
            "percentile_of_designed": float((vals < designed).mean()),
            "empirical_p_designed_not_better": p,
            "p_value_rule": "(1 + #{null >= designed}) / (n_draws + 1), one-sided",
            "clipped_entry_share_mean": float(clipped.mean()),
            "clipped_entry_share_max": float(clipped.max()),
        }

    # Same measurement for the designed matrix, so the two are comparable.
    _P_designed_raw = raw_wv @ designed_W.T
    separation["clipping"] = {
        "designed_clipped_entry_share": float(
            np.mean(_P_designed_raw != np.clip(_P_designed_raw, 0.0, 1.0))),
        "note": "share of projected entries the [0,1] clip moves, over 874 occupations",
    }

    Rcos = _cosine_matrix(_centered(Rv))[iu]
    idx = rng.choice(len(raw), size=min(200_000, len(raw)), replace=False)
    convergent = {
        "n_pairs_sampled": int(len(idx)),
        "pearson_raw_csmq_vs_riasec": float(np.corrcoef(raw[idx], Rcos[idx])[0, 1]),
        "pearson_centered_csmq_vs_riasec": float(np.corrcoef(cen[idx], Rcos[idx])[0, 1]),
    }

    payload = {
        "onet_release": "28.0",
        "compression": compression,
        "separation": separation,
        "convergent": convergent,
        "notes": {
            "scope": ("all O*NET occupations rated on both Work Values and "
                      "Interests; no participant data of any kind"),
            "auc": ("probability that a curated related pair is closer than a "
                    "randomly drawn pair; 0.5 is chance, but chance is not the "
                    "informative baseline"),
            "baselines": ("raw_work_values is the input the projection consumes, "
                          "so it bounds what the projection could preserve; "
                          "random_projection_null holds the representation type "
                          "fixed and varies only the matrix"),
            "criterion": ("O*NET relatedness is built largely from shared skills, "
                          "knowledge, and activities rather than shared work "
                          "values, which bears on the RIASEC comparison but not "
                          "on raw-versus-projected, where both sides are the "
                          "same construct"),
        },
    }

    print_header("Experiment 15 - External structure checks on the projection")
    c = compression
    print(f"  occupations={c['n_occupations']}  pairs={c['n_pairs']}")
    print(f"  raw cosine mean={c['raw_cosine_mean']:.4f} sd={c['raw_cosine_sd']:.4f}")
    print(f"  share of pairs above 0.95: {c['share_pairs_above_0_95']:.3f}")
    print(f"  centering widens spread {c['sd_ratio_centered_over_raw']:.1f}x")
    for name in ("raw_work_values", "projected_csmq", "riasec_interests"):
        s = separation[name]
        print(f"  {name:<20} dims={s['dimensions']} "
              f"AUC={s['auc_cosine']:.3f} centered={s['auc_centered_cosine']:.3f}")
    for key in ("null_dense_normal", "null_matched_support_and_sign",
                "null_permuted_support"):
        n = separation[key]
        print(f"  {key:<30} mean={n['auc_mean']:.3f} sd={n['auc_sd']:.3f} "
              f"p95={n['auc_p95']:.3f} pctile={n['percentile_of_designed']:.2f} "
              f"p={n['empirical_p_designed_not_better']:.4f}")
    save_report("exp15_external_structure", payload)
    return payload


if __name__ == "__main__":
    run()
