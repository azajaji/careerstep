"""Figure: nominal weight against realized influence for the three criteria.

Reads results/exp8f_criterion_influence.json and writes fig_influence.pdf.
Every value plotted is taken from that file; nothing is hard-coded except the
panel titles.

    python figures/make_influence_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "exp8f_criterion_influence.json"
OUT = Path(__file__).resolve().parent / "fig_influence.pdf"

CRITERIA = [("value_fit", "Value-fit"),
            ("skill_readiness", "Skill-\nreadiness"),
            ("feasibility", "Feasibility")]

# One color per criterion, held constant across panels so a reader tracks a
# single bar from left to right.
COLORS = {"value_fit": "#1f4e79",
          "skill_readiness": "#7f7f7f",
          "feasibility": "#c00000"}


def main() -> None:
    d = json.loads(RESULT.read_text(encoding="utf8"))
    w = d["weights"]
    disp = d["within_profile_dispersion"]
    loco = d["leave_one_criterion_out"]

    panels = [
        ("(a) Nominal coefficient $w_j$",
         [w[k] for k, _ in CRITERIA], "", 0.72),
        ("(b) Observed spread $\\sigma_j$\nwithin a profile",
         [disp[k]["within_profile_sd_mean"] for k, _ in CRITERIA], "", 0.30),
        ("(c) Normalized weighted\ndispersion $w_j\\sigma_j$",
         [disp[k]["weighted_dispersion_share_mean"] for k, _ in CRITERIA],
         "", 0.72),
        ("(d) Profiles whose top role\nchanges without criterion $j$",
         [loco[k]["top1_change_rate"] for k, _ in CRITERIA], "%", 1.0),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05))
    ypos = range(len(CRITERIA))[::-1]

    for ax, (title, vals, unit, xmax) in zip(axes, panels):
        for y, (key, _), v in zip(ypos, CRITERIA, vals):
            ax.barh(y, v, height=0.62, color=COLORS[key], zorder=3)
            label = f"{v * 100:.1f}%" if unit == "%" else f"{v:.3f}"
            ax.text(v + xmax * 0.035, y, label, va="center", ha="left",
                    fontsize=6.6, color="#222222")
        ax.set_title(title, fontsize=7.2, pad=5)
        ax.set_xlim(0, xmax)
        ax.set_yticks(list(ypos))
        ax.set_ylim(-0.6, len(CRITERIA) - 0.4)
        ax.tick_params(axis="x", labelsize=6.2)
        ax.tick_params(axis="y", labelsize=7.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#999999")
        ax.spines["bottom"].set_color("#999999")
        ax.xaxis.grid(True, color="#e6e6e6", zorder=0)
        ax.set_axisbelow(True)

    # ypos runs high-to-low, so the labels stay in CRITERIA order.
    axes[0].set_yticklabels([lbl for _, lbl in CRITERIA])
    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.tight_layout(pad=0.5, w_pad=1.1)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    for key, lbl in CRITERIA:
        print(f"  {lbl.replace(chr(10), ' '):16s} w={w[key]:.2f}  "
              f"sd={disp[key]['within_profile_sd_mean']:.4f}  "
              f"disp={disp[key]['weighted_dispersion_share_mean']:.3f}  "
              f"top1chg={loco[key]['top1_change_rate']:.3f}")


if __name__ == "__main__":
    main()
