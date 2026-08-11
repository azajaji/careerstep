"""Figure: the layered composite scorer and the contract between its layers.

The representation layer fixes how far each criterion varies over a decision
set; the aggregation layer sets the coefficients. Nothing in the interface
between them carries the range, which is the defect this article measures.

Layout only. The criterion ranges printed in the representation band are the
observed values from results/exp8e_component_scale.json.

    python figures/make_mechanism_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "exp8e_component_scale.json"
OUT = Path(__file__).resolve().parent / "fig_mechanism.pdf"

INK = "#1a1a1a"
GREY = "#8a8a8a"
AUDIT = "#c00000"
BAND = "#f4f4f4"
FILL_IN = "#ffffff"
FILL_CORE = "#e8eef5"
FILL_AUDIT = "#fbeaea"


def box(ax, x, y, w, h, text, fill=FILL_IN, edge=GREY, lw=0.8, fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                linewidth=lw, edgecolor=edge, facecolor=fill,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, linespacing=1.30)
    return (x, y, w, h)


def band(ax, y, h, label, rqs):
    ax.add_patch(FancyBboxPatch((0.005, y), 0.99, h,
                                boxstyle="square,pad=0", linewidth=0,
                                facecolor=BAND, zorder=1))
    # The band label sits above the flow, so it is masked against any arrow
    # that happens to pass behind it.
    mask = dict(facecolor=BAND, edgecolor="none", pad=1.6)
    ax.text(0.018, y + h - 0.016, label, ha="left", va="top", fontsize=6.4,
            color="#666666", zorder=6, fontweight="bold", bbox=mask)
    if rqs:
        ax.text(0.982, y + h - 0.016, rqs, ha="right", va="top", fontsize=6.2,
                color=AUDIT, zorder=6, bbox=mask)


def arrow(ax, src, dst, color=GREY, lw=0.9):
    ax.add_patch(FancyArrowPatch(src, dst, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, shrinkA=0,
                                 shrinkB=0, zorder=2))


def main() -> None:
    d = json.loads(RESULT.read_text(encoding="utf8"))["components"]
    rng_txt = {k: f"range {d[k]['range']:.3f}"
               for k in ("value_fit", "skill_readiness", "feasibility")}

    fig, ax = plt.subplots(figsize=(7.16, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    band(ax, 0.800, 0.200, "ELICITATION", "")
    band(ax, 0.420, 0.355, "REPRESENTATION", "RQ2   RQ3   RQ5")
    band(ax, 0.150, 0.165, "AGGREGATION", "RQ1   RQ4")
    band(ax, 0.005, 0.115, "RANKING", "")

    # Elicitation: one source per side, so the fan into the producers is clean.
    box(ax, 0.045, 0.830, 0.420, 0.108,
        "Student\n25 Likert items $\\cdot$ known skills $\\cdot$ training stage")
    box(ax, 0.535, 0.830, 0.420, 0.108,
        "Catalog role\nO*NET Work Values $\\cdot$ required skills $\\cdot$ seniority")

    # Representation: three criterion producers, each with its observed range.
    vf = box(ax, 0.045, 0.560, 0.235, 0.150,
             "Projection $W$ ($5\\times6$)\nthen cosine\nValueFit$(u,r)$",
             fill=FILL_AUDIT, edge=AUDIT, lw=1.2)
    sk = box(ax, 0.383, 0.560, 0.235, 0.150,
             "Required-skill\nmatch\n$S(u,r)$", fill=FILL_CORE)
    fe = box(ax, 0.720, 0.560, 0.235, 0.150,
             "Stage $\\times$ seniority\nlookup\n$F(u,r)$", fill=FILL_CORE)

    for b, key in ((vf, "value_fit"), (sk, "skill_readiness"), (fe, "feasibility")):
        x, y, w, h = b
        ax.text(x + w / 2, y - 0.028, rng_txt[key], ha="center", va="top",
                fontsize=6.4, color=AUDIT if key == "value_fit" else "#555555",
                zorder=4)

    for cx in (0.162, 0.500, 0.838):
        arrow(ax, (0.255, 0.830), (cx - 0.045, 0.710), color="#b0b0b0", lw=0.7)
        arrow(ax, (0.745, 0.830), (cx + 0.045, 0.710), color="#b0b0b0", lw=0.7)

    # The contract boundary, drawn in the gap between the two layers.
    ax.plot([0.02, 0.98], [0.388, 0.388], color=AUDIT, lw=1.0,
            linestyle=(0, (3, 2)), zorder=5)
    ax.text(0.5, 0.374, "each criterion's effective range is fixed above  "
            "$\\cdot$  the coefficients are set below  $\\cdot$  "
            "no interface carries the range",
            ha="center", va="top", fontsize=6.5, color=AUDIT, zorder=5)

    # Aggregation.
    box(ax, 0.243, 0.178, 0.514, 0.098,
        "RoleSuitability $= 0.60\\,$ValueFit $+\\ 0.25\\,S + 0.15\\,F$   (Eq. 1)",
        fill=FILL_AUDIT, edge=AUDIT, lw=1.2, fs=7.0)
    for x, tx in ((0.162, 0.330), (0.500, 0.500), (0.838, 0.670)):
        arrow(ax, (x, 0.560), (tx, 0.276))

    # Output.
    box(ax, 0.383, 0.025, 0.235, 0.075, "Ranked role shortlist")
    arrow(ax, (0.500, 0.178), (0.500, 0.100))

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
