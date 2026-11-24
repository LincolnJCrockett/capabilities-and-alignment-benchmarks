"""Figures 1-5 for the post / report, in the order the post uses them.

Fig 1  fig1_safetywashing_vs_irt.png      HarmBench and TruthfulQA: Safetywashing 2024 chat models vs IRT data
Fig 2  fig2_capabilities_correlations.png all 8 IRT benchmarks vs ECI (main analysis), sorted by rho
Fig 3  fig3_case_studies.png              HarmBench and TruthfulQA vs ECI, model by model, open vs closed
Fig 4  fig4_open_closed.png               the Fig 2 correlations, open-weight vs closed models
Fig 5  fig5_truthfulqa_by_lab.png         TruthfulQA correlations within each lab (labs with >= 4 models)

Caveats (AHB's unreliable grades, OR-Bench-Hard's direction, the token-cap cleaning, API measurement) are
footnotes in the report, not text in the figures (user preference, 2026-11-23).

Inputs (all produced earlier in the pipeline):
  reports/eci_irt/capabilities_correlations.csv, open_closed.csv, primary_sample_scores.csv;
  Safetywashing chat-model data; Epoch ECI file (organization, accessibility group)
Outputs: reports/figures/post/fig{1..5}_*.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_safetywashing as psw  # noqa: E402  (reuses Safetywashing's capabilities score)

RES = ROOT / "reports" / "eci_irt"
OUT = ROOT / "reports" / "figures" / "post"
OUT.mkdir(parents=True, exist_ok=True)
NL = chr(10)

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
OPEN_C, CLOSED_C, MIXED_C = AQUA, ORANGE, BLUE
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})
MAIN = "Main: empty-at-cap and grader failures missing"
CASE_LABELS = ["Claude Sonnet 4.5", "GPT-5.4 Nano", "Gemini 3 Flash", "DeepSeek-V4-Pro", "Llama 3-8B",
               "GPT-3.5 Turbo (Jun 2023)"]
MIN_LAB = 4


def models():
    """Primary sample with ECI, main-analysis abilities, access group and developer."""
    new = pd.read_csv(RES / "primary_sample_scores.csv", index_col=0)
    eci = pd.read_csv(ROOT / "data/raw/epoch/extracted/epoch_capabilities_index/eci_scores.csv").set_index("Model")
    new["access"] = new.epoch_model.map(eci["Accessibility group"].map({"Open weights": "open", "Closed weights": "closed"}))
    new["developer"] = new.epoch_model.map(eci["Organization"]).replace(
        {"Google DeepMind,Google": "Google DeepMind", "Google": "Google DeepMind"})
    return new


def trend(ax, x, y, color, lw=1.8):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() >= 3:
        b = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 20)
        ax.plot(xs, np.polyval(b, xs), color=color, lw=lw, zorder=4)


# ---------------- Fig 1 ----------------
def fig1(new):
    df, score, _ = psw.load("chat")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    rows = [("TruthfulQA", "truthfulqa_truth", "truthfulqa", "% truthful (free text)", "IRT ability (θ): higher = more truthful"),
            ("HarmBench", "HarmBench-DirectRequest-all", "harmbench",
             "attack failure rate (%): higher = refuses more", "IRT ability (θ): higher = refuses more")]
    for i, (name, swcol, irtcol, swlab, irtlab) in enumerate(rows):
        ax = axes[i, 0]
        y = df.loc[score.index, swcol] * (100 if swcol == "truthfulqa_truth" else 1)
        ax.scatter(score, y, s=30, color=BLUE, edgecolor=SURFACE, lw=0.9, zorder=3)
        trend(ax, score, y, INK)
        ax.set_ylabel(swlab, fontsize=8.5)
        ax = axes[i, 1]
        ax.scatter(new.eci, new[irtcol], s=30, color=BLUE, edgecolor=SURFACE, lw=0.9, zorder=3)
        trend(ax, new.eci, new[irtcol], INK)
        ax.set_ylabel(irtlab, fontsize=8.5)
    for ax in axes[1]:
        ax.tick_params(labelbottom=True)
    axes[1, 0].set_xlabel("Safetywashing capabilities score (PC1 of 12 benchmarks)")
    axes[1, 1].set_xlabel("Epoch Capabilities Index (ECI)")
    fig.tight_layout(rect=(0.05, 0, 1, 0.95))
    # column headers (studies) and row labels (benchmarks), placed from the final axes positions
    for j, study in enumerate(["Ren et al., 2024", "Rivera et al., 2026"]):
        b = axes[0, j].get_position()
        fig.text((b.x0 + b.x1) / 2, 0.965, study, ha="center", va="center", fontsize=12, fontweight="bold")
    for i, name in enumerate(["TruthfulQA", "HarmBench"]):
        b = axes[i, 0].get_position()
        fig.text(0.012, (b.y0 + b.y1) / 2, name, ha="left", va="center", rotation=90, fontsize=12,
                 fontweight="bold")
    fig.savefig(OUT / "fig1_safetywashing_vs_irt.png", dpi=180)
    plt.close(fig)


# ---------------- Fig 2 ----------------
def main_corrs():
    c = pd.read_csv(RES / "capabilities_correlations.csv")
    return c[(c.variant == MAIN) & (c.kind == "benchmark")].set_index("measure")


def forest_axes(ax, order):
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(len(order) - 0.5, -0.6)
    ax.grid(axis="y", visible=False)


def fig2():
    c = main_corrs().sort_values("rho", ascending=False)
    order = list(c.index)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    forest_axes(ax, order)
    for i, m in enumerate(order):
        r = c.loc[m]
        ax.plot([r.lo, r.hi], [i, i], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(r.rho, i, s=60, zorder=3, color=BLUE, edgecolor=SURFACE, lw=1)
        ax.text(1.04, i, f"{r.rho:+.2f}  (n = {int(r.n)})", va="center", fontsize=8.5, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.set_xlabel("Spearman ρ with ECI (95% bootstrap CI)")
    ax.set_title("Correlation Between Each Safety Benchmark and ECI", fontsize=11.5, pad=10)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(OUT / "fig2_capabilities_correlations.png", dpi=180)
    plt.close(fig)
    return order


# ---------------- Fig 3 ----------------
def fig3(new):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True)
    for ax, (m, col, ylab) in zip(axes, [("HarmBench", "harmbench", "IRT ability (θ): higher = refuses more"),
                                         ("TruthfulQA", "truthfulqa", "IRT ability (θ): higher = more truthful")]):
        for g, c in [("open", OPEN_C), ("closed", CLOSED_C)]:
            s_ = new[new.access == g]
            ax.scatter(s_.eci, s_[col], s=34, color=c, edgecolor=SURFACE, lw=1.0, zorder=3)
            trend(ax, s_.eci, s_[col], c, lw=1.6)
        o = new[new.access.isna()]
        ax.scatter(o.eci, o[col], s=34, color=MUTED, edgecolor=SURFACE, lw=1.0, zorder=3)
        labs = [x for x in CASE_LABELS if not (m == "HarmBench" and x == "Llama 3-8B")]
        for _, r in new[new.epoch_model.isin(labs) & new[col].notna()].iterrows():
            off = (4, -11) if (m == "HarmBench" and r.epoch_model.startswith("GPT-3.5")) else (4, 4)
            ax.annotate(r.epoch_model.replace(" (Jun 2023)", ""), (r.eci, r[col]), xytext=off,
                        textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_title(m)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Epoch Capabilities Index (ECI)")
    for c, lab in [(OPEN_C, "Open weights"), (CLOSED_C, "Closed weights"), (MUTED, "Neither")]:
        axes[0].scatter([], [], s=34, color=c, label=lab)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("HarmBench and TruthfulQA against capability, model by model", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig3_case_studies.png", dpi=180)
    plt.close(fig)


# ---------------- Fig 4 ----------------
def fig4(order):
    oc = pd.read_csv(RES / "open_closed.csv")
    oc = oc[oc["sample"] == "primary"].set_index("measure").loc[order]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    forest_axes(ax, order)
    for i, m in enumerate(order):
        r = oc.loc[m]
        for g, c, dy in [("open", OPEN_C, -0.17), ("closed", CLOSED_C, 0.17)]:
            ax.plot([r[f"lo_{g}"], r[f"hi_{g}"]], [i + dy, i + dy], color=c, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter(r[f"rho_{g}"], i + dy, s=50, zorder=3, color=c, edgecolor=SURFACE, lw=1)
        star = " *" if (r.diff_lo > 0 or r.diff_hi < 0) else ""
        ax.text(1.04, i, f"{r['diff']:+.2f}{star}", va="center", fontsize=8.5, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.text(1.04, -0.5, "closed − open", fontsize=8, color=INK2, transform=ax.get_yaxis_transform())
    n_o = f"{int(oc.n_open.min())}–{int(oc.n_open.max())}"
    n_c = f"{int(oc.n_closed.min())}–{int(oc.n_closed.max())}"
    ax.plot([], [], color=OPEN_C, lw=2, marker="o", label=f"Open weights (n = {n_o})")
    ax.plot([], [], color=CLOSED_C, lw=2, marker="o", label=f"Closed weights (n = {n_c})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9)
    ax.set_xlabel("Spearman ρ with ECI (95% bootstrap CI)")
    ax.set_title("Correlation Between Each Safety Benchmark and ECI:" + NL + "Open vs Closed Weights", fontsize=11.5, pad=10)
    fig.text(0.01, 0.01, "* 95% CI of the difference excludes 0.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 0.87, 1))
    fig.savefig(OUT / "fig4_open_closed.png", dpi=180)
    plt.close(fig)


# ---------------- Fig 5 ----------------
def lab_table(new, col="truthfulqa"):
    """Spearman rho within each group: pooled open/closed, each lab by access group, Google pooled."""
    rows = []
    for g, lab in [("open", "All open-weight models"), ("closed", "All closed models")]:
        s = new[new.access == g]
        r, n = psw.rho(s.eci, s[col])
        rows.append({"label": lab, "rho": r, "n": n, "color": OPEN_C if g == "open" else CLOSED_C, "pooled": True})
    for dev, s in new[new.access.notna()].groupby("developer"):
        if dev == "Google DeepMind":
            r, n = psw.rho(s.eci, s[col])
            if n >= MIN_LAB:
                rows.append({"label": "Google DeepMind" + NL + "(open and closed models)", "rho": r, "n": n,
                             "color": MIXED_C, "pooled": False})
            continue
        for g, sg in s.groupby("access"):
            r, n = psw.rho(sg.eci, sg[col])
            if n >= MIN_LAB:
                rows.append({"label": dev, "rho": r, "n": n, "color": OPEN_C if g == "open" else CLOSED_C,
                             "pooled": False})
    t = pd.DataFrame(rows)
    return pd.concat([t[t.pooled], t[~t.pooled].sort_values("rho", ascending=False)], ignore_index=True)


def fig5(new):
    t = lab_table(new)
    t.assign(label=t.label.str.replace(NL, " ")).drop(columns="color").to_csv(RES / "truthfulqa_by_lab.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axvline(0, color=INK2, lw=1)
    for i, r in t.iterrows():
        ax.scatter(r.rho, i, s=90 if r.pooled else 55, marker="D" if r.pooled else "o", color=r.color,
                   edgecolor=SURFACE, lw=1, zorder=3)
        ax.text(1.06, i, f"{r.rho:+.2f}  (n = {int(r.n)})", va="center", fontsize=8.5, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.label)
    for tick, pooled in zip(ax.get_yticklabels(), t.pooled):
        tick.set_fontweight("bold" if pooled else "normal")
    ax.axhline(t.pooled.sum() - 0.5, color=GRID, lw=1.2)
    ax.set_ylim(len(t) - 0.5, -0.6)
    ax.set_xlim(-1.05, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Spearman ρ between ECI and TruthfulQA ability, within the group")
    ax.set_title("TruthfulQA vs capability, lab by lab", fontsize=11.5, pad=10)
    for c, lab in [(OPEN_C, "Open weights"), (CLOSED_C, "Closed weights"), (MIXED_C, "Both")]:
        ax.scatter([], [], s=55, color=c, label=lab)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(OUT / "fig5_truthfulqa_by_lab.png", dpi=180)
    plt.close(fig)


def main():
    for old in OUT.glob("*.png"):   # numbering changed on 2026-11-23; drop stale files
        old.unlink()
    new = models()
    fig1(new)
    order = fig2()
    fig3(new)
    fig4(order)
    fig5(new)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
    print(pd.read_csv(RES / "truthfulqa_by_lab.csv").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
