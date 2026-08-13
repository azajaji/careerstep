"""Figure 1: block diagram of the audited scorer.

Left-to-right stages: elicitation, the three criterion producers,
the weighted sum, the ranked output. The observed range of each criterion is
printed under its producer, because the range is fixed there while the weight
is set one stage later, and that separation is what the article tests.

The elicitation box is labelled for what this study actually used. The
inventory was never administered to anyone: profiles are generated in code,
and the ranking benchmark supplies orientation vectors directly. The figure
must not imply human respondents.

Ranges are read from results/exp8e_component_scale.json; the layout is fixed.

    python figures/make_mechanism_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "exp8e_component_scale.json"
OUT = Path(__file__).resolve().parent / "fig_mechanism.pdf"

INK = "#1a1a1a"
LINE = "#666666"
AUDIT = "#b00000"
NOTE = "#444444"
FILL_IN = "#f0f0f0"      # inputs
FILL_MID = "#e4ecf5"     # standard producers and output
FILL_AUD = "#fbeaea"     # the two elements under test
FS = 6.6

BUS = 0.588
DASH = 0.556


def box(ax, x, y, w, h, text, fill=FILL_MID, edge="#8a8a8a", lw=0.8, fs=FS):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                           linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=4, linespacing=1.5)
    return x, y, w, h


def arrow(ax, p, q, color=LINE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color=color,
                                 shrinkA=0, shrinkB=0, zorder=2))


def elbow(ax, x0, y0, x1, y1, xm, color=LINE):
    ax.plot([x0, xm], [y0, y0], color=color, lw=0.9, zorder=2)
    ax.plot([xm, xm], [y0, y1], color=color, lw=0.9, zorder=2)
    arrow(ax, (xm, y1), (x1, y1), color=color)


def main() -> None:
    c = json.loads(RESULT.read_text(encoding="utf8"))["components"]
    rng = {k: c[k]["range"] for k in ("value_fit", "skill_readiness", "feasibility")}

    fig, ax = plt.subplots(figsize=(7.16, 2.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for x, label in ((0.105, "ELICITATION"), (0.415, "REPRESENTATION"),
                     (0.707, "AGGREGATION"), (0.922, "RANKING")):
        ax.text(x, 0.985, label, ha="center", va="top", fontsize=6.2,
                color="#666666", fontweight="bold")

    a = box(ax, 0.010, 0.620, 0.190, 0.250,
            "Student profile\norientation vector,\nknown skills,\ntraining stage",
            fill=FILL_IN)
    ax.text(0.105, 0.600, "synthetic; the 25-item inventory\nwas not administered",
            ha="center", va="top", fontsize=5.9, color=NOTE, style="italic")

    b = box(ax, 0.010, 0.150, 0.190, 0.250,
            "Catalog role\nO*NET Work Values,\nrequired skills,\nseniority",
            fill=FILL_IN)
    ax.text(0.105, 0.130, "public and author-curated",
            ha="center", va="top", fontsize=5.9, color=NOTE, style="italic")

    p1 = box(ax, 0.300, 0.700, 0.230, 0.195,
             "Projection $W$ ($5{\\times}6$),\nthen cosine\n$\\rightarrow$ ValueFit",
             fill=FILL_AUD, edge=AUDIT, lw=1.3)
    p2 = box(ax, 0.300, 0.420, 0.230, 0.195,
             "Required-skill\nmatch\n$\\rightarrow S$")
    p3 = box(ax, 0.300, 0.140, 0.230, 0.195,
             "Stage $\\times$ seniority\nlookup\n$\\rightarrow F$")

    for bx, key in ((p1, "value_fit"), (p2, "skill_readiness"),
                    (p3, "feasibility")):
        x, y, w, h = bx
        ax.text(x + w / 2, y - 0.018, f"observed range {rng[key]:.3f}",
                ha="center", va="top", fontsize=6.2, color=NOTE)

    s = box(ax, 0.615, 0.398, 0.185, 0.240,
            "RoleSuitability\n$0.60\\,$ValueFit\n$+\\ 0.25\\,S + 0.15\\,F$\n(Eq. 1)",
            fill=FILL_AUD, edge=AUDIT, lw=1.3)

    box(ax, 0.850, 0.453, 0.145, 0.130, "Ranked role\nshortlist", fill=FILL_IN)

    for src, dsts in ((a, (p1, p2)), (b, (p1, p3))):
        sx, sy, sw, sh = src
        for d in dsts:
            dx, dy, dw, dh = d
            elbow(ax, sx + sw, sy + sh / 2, dx, dy + dh / 2, xm=0.255)

    for p in (p1, p2, p3):
        elbow(ax, p[0] + p[2], p[1] + p[3] / 2, s[0], s[1] + s[3] / 2, xm=BUS)
    arrow(ax, (s[0] + s[2], s[1] + s[3] / 2), (0.850, 0.518))

    ax.plot([DASH, DASH], [0.075, 0.930], color=AUDIT, lw=0.9,
            linestyle=(0, (3, 2.5)), zorder=1)
    ax.text(0.500, 0.048,
            "each criterion's range is fixed left of the dashed line; "
            "its weight is set right of it",
            ha="center", va="top", fontsize=6.3, color=AUDIT)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
