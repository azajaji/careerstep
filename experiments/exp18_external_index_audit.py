"""The effective-influence audit applied to a composite scorer we did not build.

Every other experiment here audits our own artifact, which leaves open whether
the procedure says anything about scorers designed by other people. This one
runs the same measures on the Human Development Index, a published composite
that ranks countries and states its weights explicitly.

HDI is a useful second case for two reasons. Its weights are equal by design,
one third on each of health, education, and income, so the question is not
whether a large coefficient is honoured but whether equal coefficients buy
equal influence. And it is a geometric mean, so the audit has to be taken on
the scale where the scorer is additive:

    ln HDI = (1/3) ln I_health + (1/3) ln I_education + (1/3) ln I_income

Ranking by HDI and by ln HDI are the same ranking, so the measures of
Experiment 8f apply unchanged to the log-scale criteria with weights 1/3.

The dimension indices are recomputed from UNDP's published goalposts rather
than taken from a file, and the resulting HDI is checked against the published
value before anything is audited. If that check fails the audit is measuring
our own arithmetic instead of theirs.

Two decision sets are reported. The global ranking is the one HDI is used for.
The regional rankings mirror the unit used in the rest of this study, where a
ranking is over the alternatives one decision maker actually compares.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from careerstep.seeding import set_global_seeds
from experiments._io import print_header, save_report

URL = ("https://hdr.undp.org/sites/default/files/2023-24_HDR/"
       "HDR23-24_Composite_indices_complete_time_series.csv")
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "hdr"
CSV = CACHE / "hdr_composite.csv"
YEAR = 2022

# UNDP goalposts, HDR 2023-24 technical note. GNI per capita is 2017 PPP $.
LE_MIN, LE_MAX = 20.0, 85.0
EYS_MAX, MYS_MAX = 18.0, 15.0
GNI_MIN, GNI_MAX = 100.0, 75000.0

CRITERIA = ("health", "education", "income")
WEIGHTS = {c: 1.0 / 3.0 for c in CRITERIA}


def _fetch() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    if not CSV.exists():
        print(f"  downloading {URL}")
        urllib.request.urlretrieve(URL, CSV)
    return pd.read_csv(CSV, encoding="latin-1")


def _dimension_indices(d: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the three HDI dimension indices from the published goalposts."""
    out = pd.DataFrame(index=d.index)
    out["health"] = (d[f"le_{YEAR}"] - LE_MIN) / (LE_MAX - LE_MIN)
    eys = np.minimum(d[f"eys_{YEAR}"] / EYS_MAX, 1.0)
    mys = np.minimum(d[f"mys_{YEAR}"] / MYS_MAX, 1.0)
    out["education"] = (eys + mys) / 2.0
    out["income"] = ((np.log(d[f"gnipc_{YEAR}"]) - np.log(GNI_MIN))
                     / (np.log(GNI_MAX) - np.log(GNI_MIN)))
    return out.clip(lower=1e-9, upper=1.0)


def _audit(sets: List[np.ndarray], w: np.ndarray) -> Dict[str, dict]:
    """Experiment 8f's measures over one or more decision sets, on the log scale."""
    shares, taus, top1, ov10 = [], [[] for _ in CRITERIA], [[] for _ in CRITERIA], \
        [[] for _ in CRITERIA]
    decisive = np.zeros(len(CRITERIA))
    n_pairs = 0

    def rank(X):
        return np.argsort(-(X @ w), kind="stable")

    for X in sets:
        sd = X.std(axis=0, ddof=1)
        ws = w * sd
        shares.append(ws / ws.sum() if ws.sum() > 0 else np.zeros_like(ws))

        full = rank(X)
        pos_full = np.empty(len(full), dtype=int)
        pos_full[full] = np.arange(len(full))
        k = min(10, len(full))
        for j in range(len(CRITERIA)):
            Y = X.copy()
            Y[:, j] = Y[:, j].mean()
            got = rank(Y)
            pos_got = np.empty(len(got), dtype=int)
            pos_got[got] = np.arange(len(got))
            taus[j].append(stats.kendalltau(pos_full, pos_got).statistic)
            top1[j].append(0.0 if full[0] == got[0] else 1.0)
            a, b = set(full[:k].tolist()), set(got[:k].tolist())
            ov10[j].append(len(a & b) / len(a | b))

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
            "top10_jaccard_mean": float(np.mean(ov10[j])),
            "share_of_pairs_decided": float(decisive[j] / n_pairs),
        }
        for j, c in enumerate(CRITERIA)
    }


def run() -> dict:
    set_global_seeds()
    raw = _fetch()

    need = [f"{p}_{YEAR}" for p in ("hdi", "le", "eys", "mys", "gnipc")]
    d = raw.dropna(subset=need).copy()
    # The file carries aggregate rows (regions, world) alongside countries;
    # those have no ISO3 country code of the usual form.
    d = d[d["iso3"].astype(str).str.len() == 3]
    idx = _dimension_indices(d)

    recomputed = (idx["health"] * idx["education"] * idx["income"]) ** (1.0 / 3.0)
    published = d[f"hdi_{YEAR}"].to_numpy(dtype=float)
    err = np.abs(recomputed.to_numpy(dtype=float) - published)
    reproduction = {
        "n_countries": int(len(d)),
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
        "pearson_r": float(np.corrcoef(recomputed, published)[0, 1]),
        "share_within_0_001": float((err <= 0.001).mean()),
    }

    L = np.log(idx[list(CRITERIA)].to_numpy(dtype=float))
    w = np.array([WEIGHTS[c] for c in CRITERIA], dtype=float)

    global_audit = _audit([L], w)

    regions = d["region"].astype(str)
    reg_sets, reg_names = [], []
    for name, grp in d.groupby(regions):
        if name in ("nan", "") or len(grp) < 8:
            continue
        reg_sets.append(np.log(_dimension_indices(grp)[list(CRITERIA)]
                               .to_numpy(dtype=float)))
        reg_names.append(name)
    regional_audit = _audit(reg_sets, w)

    payload = {
        "scorer": "Human Development Index",
        "source": "UNDP Human Development Report 2023-24 composite indices",
        "url": URL,
        "year": YEAR,
        "weights": WEIGHTS,
        "reproduction_check": reproduction,
        "global_decision_set": global_audit,
        "regional_decision_sets": {
            "n_regions": len(reg_sets),
            "regions": reg_names,
            "sizes": [int(len(x)) for x in reg_sets],
            "measures": regional_audit,
        },
        "notes": {
            "additive_scale": ("HDI is a geometric mean, so the audit is taken on "
                               "log dimension indices, where the scorer is additive "
                               "with weights 1/3 and the ranking is unchanged"),
            "goalposts": ("dimension indices are recomputed from UNDP's published "
                          "goalposts and checked against the published HDI before "
                          "the audit, so the object under audit is theirs"),
            "unit": ("the global set is how HDI is used; the regional sets mirror "
                     "the within-decision-set unit used elsewhere in this study"),
        },
    }

    print_header("Experiment 18 - The audit applied to a published external index")
    r = reproduction
    print(f"  reproduced HDI for {r['n_countries']} countries: "
          f"max abs error {r['max_abs_error']:.4f}, "
          f"{r['share_within_0_001'] * 100:.1f}% within 0.001, r={r['pearson_r']:.5f}")
    for label, res in (("global", global_audit), ("regional", regional_audit)):
        print(f"  {label} decision set(s)")
        print(f"    {'criterion':<12}{'w':>6}{'disp':>8}{'tau':>8}"
              f"{'top1 chg':>10}{'decides':>9}")
        for c in CRITERIA:
            v = res[c]
            print(f"    {c:<12}{WEIGHTS[c]:>6.2f}"
                  f"{v['weighted_dispersion_share_mean']:>8.3f}"
                  f"{v['kendall_tau_mean']:>8.3f}"
                  f"{v['top1_change_rate']:>10.3f}"
                  f"{v['share_of_pairs_decided']:>9.3f}")
    save_report("exp18_external_index_audit", payload)
    return payload


if __name__ == "__main__":
    run()
