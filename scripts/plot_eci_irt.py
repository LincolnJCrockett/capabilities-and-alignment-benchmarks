"""Figures for the ECI x IRT analysis: the eight benchmarks only.

Factor scores are never plotted beside the benchmarks (user decision 2026-11-22, journal): each factor
score is a weighted mix of all eight benchmarks. They get their own figures for the report's separate factor
section: factor_correlations.png and factor_weights.png (how each score is composed).

Inputs : reports/eci_irt/capabilities_correlations.csv
Outputs: reports/figures/fig5_capabilities_correlations.png  (main result)
         reports/figures/fig6_sensitivity.png
         reports/figures/factor_correlations.png, factor_weights.png  (separate factor section)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "eci_irt"
FIG = ROOT / "reports" / "figures"

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

res = pd.read_csv(RES / "capabilities_correlations.csv")
ROWS = ["HarmBench", "SORRY-Bench", "AHB", "OR-Bench-Hard", "TruthfulQA", "Do-Not-Answer", "AdvBench", "Health-ORSC"]
LABEL = {"OR-Bench-Hard": "OR-Bench-Hard (answers benign prompts)", "TruthfulQA": "TruthfulQA (free text)"}
BASE = "Main: empty-at-cap and grader failures missing"
ys = np.arange(len(ROWS))[::-1]


def frame(ax):
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])


# ---------------- main result ----------------
r = res[res.variant == BASE].set_index("measure").loc[ROWS]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
frame(ax)
ax.errorbar(r.rho, ys, xerr=[r.rho - r.lo, r.hi - r.rho], fmt="o", color=BLUE, ms=7,
            mec=SURFACE, mew=1.5, elinewidth=1.6, capsize=0)
for y, (m, row) in zip(ys, r.iterrows()):
    ax.text(row.hi + 0.03, y, f"{row.rho:+.2f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(ys)
ax.set_yticklabels([f"{LABEL.get(m, m)}  (n={int(r.loc[m, 'n'])})" for m in ROWS])
ax.set_xlabel("Spearman ρ with ECI (95% bootstrap CI)")
fig.suptitle("Capabilities correlations of the IRT safety benchmarks\n"
             "(primary sample; empty-at-cap and ungraded responses missing)", x=0.01, ha="left", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig5_capabilities_correlations.png", dpi=200)
plt.close(fig)

# ---------------- sensitivity ----------------
variants = [(BASE, BLUE), ("As released (paper default)", MUTED),
            ("Main, dropping models with >10% empty-at-cap", AQUA),
            ("Main, adding non-exact matches", ORANGE)]
fig, ax = plt.subplots(figsize=(7.8, 5.6))
frame(ax)
for j, (v, c) in enumerate(variants):
    rv = res[res.variant == v].set_index("measure").loc[ROWS]
    off = (1.5 - j) * 0.17
    ax.errorbar(rv.rho, ys + off, xerr=[rv.rho - rv.lo, rv.hi - rv.rho], fmt="o", color=c, ms=5.5,
                mec=SURFACE, mew=1.2, elinewidth=1.2, capsize=0, label=v)
ax.set_yticks(ys)
ax.set_yticklabels([LABEL.get(m, m) for m in ROWS])
ax.set_xlabel("Spearman ρ with ECI (95% bootstrap CI)")
ax.set_title("Sensitivity of the capabilities correlations")
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2, frameon=False, fontsize=8.5)
fig.text(0.99, 0.005, "Each masking on its own, and factor scores: reports/eci_irt/capabilities_correlations.csv",
         ha="right", fontsize=7, color=INK2)
fig.tight_layout()
fig.savefig(FIG / "fig6_sensitivity.png", dpi=200)
plt.close(fig)
print("ok")

# =====================================================================================================
# Separate factor section (report §7): factor correlations, and how each factor score is composed.
# These are kept apart from the benchmark figures because each factor score weights all 8 benchmarks.
# =====================================================================================================
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FACT = ["Refusal strictness", "Truthfulness", "Contextual harm"]
fy = np.arange(len(FACT))[::-1]
fig, ax = plt.subplots(figsize=(7.8, 3.6))
ax.axvline(0, color=INK2, lw=1)
ax.set_xlim(-1, 1)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.grid(axis="y", visible=False)
for j, (v, c) in enumerate(variants):
    rv = res[res.variant == v].set_index("measure").loc[FACT]
    off = (1.5 - j) * 0.17
    ax.errorbar(rv.rho, fy + off, xerr=[rv.rho - rv.lo, rv.hi - rv.rho], fmt="o", color=c, ms=6 if j == 0 else 5,
                mec=SURFACE, mew=1.2, elinewidth=1.6 if j == 0 else 1.1, capsize=0,
                label=f"{v} (n = {int(rv.n.min())})")
ax.set_ylim(-0.6, len(FACT) - 0.4)
ax.set_yticks(fy)
ax.set_yticklabels(FACT)
ax.set_xlabel("Spearman ρ with ECI (95% bootstrap CI)")
fig.suptitle("Capabilities correlations of the IRT paper's factor scores", x=0.01, ha="left",
             fontweight="bold", fontsize=12)
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.25), ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "factor_correlations.png", dpi=200)
plt.close(fig)

# composition: weight of each benchmark (per SD) in each factor score; diverging, neutral midpoint at 0
W = pd.read_csv(RES / "factor_score_weights.csv", index_col=0).loc[ROWS, FACT]
cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#efeeea", BLUE])
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.imshow(W.to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
for i in range(W.shape[0]):
    for k in range(W.shape[1]):
        v = W.iat[i, k]
        ax.text(k, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                color=SURFACE if abs(v) > 0.6 else INK, fontweight="bold" if abs(v) > 0.3 else "normal")
ax.set_xticks(range(len(FACT)))
ax.set_xticklabels(FACT)
ax.xaxis.tick_top()
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels(ROWS)
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(FACT)), minor=True)
ax.set_yticks(np.arange(-0.5, len(ROWS)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="both", length=0)
fig.suptitle("How each factor score is built: weight of each benchmark\n(per standard deviation; paper's factor model)",
             x=0.01, ha="left", fontweight="bold", fontsize=11.5)
fig.text(0.01, 0.01, "Blue = adds to the score, orange = subtracts. Every factor score uses all 8 benchmarks.",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "factor_weights.png", dpi=200)
plt.close(fig)
print("factor figures ok")
