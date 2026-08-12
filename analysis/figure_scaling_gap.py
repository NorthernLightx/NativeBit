"""Render the scaling figure for the README: quality gap to float vs model size.

Two series, three scales: NativeBit QAT against the strongest post-hoc
baseline at the same bit width. Numbers come from the tables in README.md;
edit them here when the curve gains a point.

    python analysis/plot_scaling_gap.py --out assets/scaling_gap.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (params, QAT gap %, RTN gap %) — gap to that scale's float baseline, 3-bit.
# 48M/76M: WikiText-103 in-domain, RTX 3070, float+5K continued baseline.
# 2.2B: OpenWebText training, WikiText-103 cross-eval, TPU v6e-8. That point
# predates the continued-float control, so its baseline is slightly weaker.
POINTS = [
    (48e6, 4.3, 11.3),
    (76e6, 3.3, 9.3),
    (2.2e9, -0.03, 2.7),
]

QAT_COLOR = "#2a78d6"
RTN_COLOR = "#eb6834"
INK = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#e5e4df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/scaling_gap.png")
    args = ap.parse_args()

    sizes = [p[0] for p in POINTS]
    qat = [p[1] for p in POINTS]
    rtn = [p[2] for p in POINTS]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(sizes[0] * 0.82, 0.45, "float quality", color=MUTED, fontsize=8.5,
            va="bottom")

    for ys, color, label in ((rtn, RTN_COLOR, "Post-hoc RTN"),
                             (qat, QAT_COLOR, "NativeBit QAT")):
        ax.plot(sizes, ys, color=color, lw=2, marker="o", markersize=8,
                markerfacecolor=color, markeredgecolor="#fcfcfb",
                markeredgewidth=2, label=label, zorder=3, clip_on=False)

    # Direct labels instead of a legend box, parked on the long empty segment
    # between the 76M and 2.2B points so they clear the value labels.
    mid_x = (sizes[1] * sizes[2]) ** 0.5
    for ys, color, label in ((rtn, RTN_COLOR, "Post-hoc RTN"),
                             (qat, QAT_COLOR, "NativeBit QAT")):
        mid_y = (ys[1] + ys[2]) / 2
        ax.annotate(label, (mid_x, mid_y), textcoords="offset points",
                    xytext=(0, 12), ha="center", color=color, fontsize=10.5,
                    fontweight="600")

    def fmt(v):
        return f"{v:+.2f}%" if abs(v) < 1 else f"{v:+.1f}%"

    for x, y in zip(sizes, qat):
        ax.annotate(fmt(y), (x, y), textcoords="offset points",
                    xytext=(0, -17), ha="center", color=INK, fontsize=9)
    for x, y in zip(sizes, rtn):
        ax.annotate(fmt(y), (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", color=INK, fontsize=9)

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(["48M", "76M", "2.2B"], fontsize=10, color=INK)
    ax.set_xlim(sizes[0] * 0.72, sizes[-1] * 1.5)
    ax.set_ylim(-2, 13.5)
    ax.set_yticks([0, 4, 8, 12])
    ax.set_yticklabels(["0", "+4%", "+8%", "+12%"], fontsize=9, color=MUTED)
    ax.set_xlabel("model size (log scale)", fontsize=9.5, color=MUTED, labelpad=8)
    ax.set_ylabel("perplexity gap to float", fontsize=9.5, color=MUTED, labelpad=8)
    ax.set_title("3-bit quality gap closes with scale, and QAT stays ahead",
                 fontsize=12, color=INK, pad=14, loc="left", fontweight="600")

    ax.set_xticks([], minor=True)  # log minor ticks read as unlabeled data points
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)

    fig.text(0.5, -0.02,
             "48M/76M: WikiText-103 in-domain, RTX 3070, baseline is float trained "
             "the same extra 5K steps.\n2.2B: OpenWebText training, WikiText-103 "
             "cross-eval, TPU v6e-8, no continued-float control.",
             fontsize=7.5, color=MUTED, ha="center", va="top")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
