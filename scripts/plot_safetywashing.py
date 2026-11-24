"""Safetywashing scatterplots: the original analysis, and old vs new data side by side.

Capabilities score is computed exactly as in Safetywashing's analysis.py (PC1 of the Spearman correlation
matrix of 12 standardized capability benchmarks). Each panel shows a safety benchmark (y, raw score as in
their data, where higher = safer by their sign conventions) against that score (x), with the Spearman rho
that the paper reports.

Outputs: reports/figures/sw_original_chat.png, sw_original_base.png   (headline benchmarks)
         reports/figures/sw_all_scatterplots.pdf                      (every safety benchmark, chat + base)
         reports/figures/sw_vs_now.png                                (the 2 benchmarks in both datasets)
         reports/eci_irt/sw_vs_now.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
SW = ROOT / "data" / "raw" / "safetywashing" / "repo" / "data"
FIG = ROOT / "reports" / "figures"
RES = ROOT / "reports" / "eci_irt"
NL = chr(10)

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

CAP = ["logiqa", "piqa", "hellaswag", "winogrande", "superglue_copa", "medqa_4options",
       "arc_challenge", "mmlu", "minerva_math", "lambada_openai", "gsm8k", "bbh"]
META = ["model_size", "FLOP", "name", "type"]
HEADLINE = [
    ("MT_Bench", "MT-Bench"), ("LMSYS_Chatbot_Arena", "Chatbot Arena"), ("ethics_all", "ETHICS"),
    ("machiavelli-utility.Σ", "MACHIAVELLI utility"), ("machiavelli-power.Σ", "MACHIAVELLI power"),
    ("machiavelli-violations.Σ", "MACHIAVELLI violations"), ("sycophancy", "Sycophancy"),
    ("bbq_amb", "BBQ ambiguous"), ("bbq_disamb", "BBQ disambiguated"), ("crows_pairs_english", "CrowS-Pairs"),
    ("discrim_eval", "Discrim-Eval"), ("winogender", "Winogender"), ("truthfulqa_mc1", "TruthfulQA MC1"),
    ("truthfulqa_truth", "TruthfulQA generative (truthful)"), ("gpqa", "GPQA"), ("scrolls_quality", "QuALITY"),
    ("anli", "ANLI"), ("adv_glue", "AdvGLUE"), ("decodingtrust_adv_glue++", "AdvGLUE++"),
    ("HarmBench-DirectRequest-all", "HarmBench direct requests"),
    ("HarmBench-HumanJailbreaks-all", "HarmBench human jailbreaks"), ("HarmBench-TAP-T-all", "HarmBench TAP"),
    ("wmdp_bio", "WMDP bio (inverted)"), ("wmdp_cyber", "WMDP cyber (inverted)"),
]


def load(kind):
    df = pd.read_csv(SW / f"benchmarks_{kind}_models.csv", index_col="model")
    info = pd.read_csv(SW / "benchmarks_info.csv", index_col="benchmark")
    z = df[CAP]
    z = ((z - z.mean()) / z.std(ddof=0)).dropna()
    ev, evec = np.linalg.eigh(z.corr("spearman").to_numpy())
    pc = evec[:, -1]
    pc = pc if np.abs(pc).max() == pc.max() else -pc
    score = pd.Series(z.to_numpy() @ pc, index=z.index, name="cap_score")
    return df, score, info


def rho(x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    return d.corr("spearman").iloc[0, 1], len(d)


def panel(ax, x, y, title, color=BLUE, xlabel=None):
    ax.scatter(x, y, s=22, color=color, edgecolor=SURFACE, lw=0.8, zorder=3)
    r, n = rho(x, y)
    ax.set_title(title, fontsize=9)
    ax.text(0.03, 0.96, f"ρ = {r:+.2f}  (n={n})", transform=ax.transAxes, va="top", fontsize=8, color=INK2,
            zorder=5, bbox=dict(facecolor=SURFACE, alpha=0.85, edgecolor="none", pad=1.2))
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5)
    ax.tick_params(labelsize=7.5)


def headline_grid(kind):
    df, score, _ = load(kind)
    cols = [(c, l) for c, l in HEADLINE if c in df.columns and df[c].notna().sum() >= 5]
    ncol = 6
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.6 * nrow + 0.8))
    for ax, (c, lab) in zip(axes.ravel(), cols):
        panel(ax, score, df.loc[score.index, c], lab)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Safetywashing capabilities score", fontsize=8.5)
    fig.suptitle(f"Safetywashing, original data: headline safety benchmarks vs capabilities ({kind} models, n = {len(score)})",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / f"sw_original_{kind}.png", dpi=170)
    plt.close(fig)


def all_pdf():
    with PdfPages(FIG / "sw_all_scatterplots.pdf") as pdf:
        for kind in ("chat", "base"):
            df, score, info = load(kind)
            safety = [c for c in df.columns if c not in CAP + META and df[c].notna().sum() >= 5]
            per_page = 20
            for start in range(0, len(safety), per_page):
                chunk = safety[start:start + per_page]
                fig, axes = plt.subplots(4, 5, figsize=(14, 10))
                for ax, c in zip(axes.ravel(), chunk):
                    name = info.loc[c, "name"] if c in info.index else c
                    panel(ax, score, df.loc[score.index, c], str(name)[:38])
                for ax in axes.ravel()[len(chunk):]:
                    ax.axis("off")
                for ax in axes[-1]:
                    ax.set_xlabel("capabilities score", fontsize=8)
                page = start // per_page + 1
                fig.suptitle(f"Safetywashing original data, {kind} models: all safety benchmarks "
                             f"(page {page} of {int(np.ceil(len(safety) / per_page))})",
                             x=0.01, ha="left", fontweight="bold", fontsize=12)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def sw_vs_now():
    """Benchmarks present in both datasets: Safetywashing's 2024 chat models against THEIR capabilities
    score (as published), next to the IRT data against ECI (main analysis). No model crosswalk between
    the two datasets is used."""
    df, score, _ = load("chat")
    new = pd.read_csv(RES / "primary_sample_scores.csv", index_col=0)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    ax = axes[0, 0]
    for col, c, lab in [("truthfulqa_mc1", ORANGE, "multiple choice (MC1)"), ("truthfulqa_truth", AQUA, "free text (% truthful)")]:
        y = df.loc[score.index, col]
        ax.scatter(score, y, s=26, color=c, edgecolor=SURFACE, lw=0.8, zorder=3, label=lab)
        r, n = rho(score, y)
        rows.append({"benchmark": f"TruthfulQA, {lab}", "data": "Safetywashing, 2024 chat models",
                     "capability measure": "Safetywashing capabilities score", "rho": r, "n": n})
    ax.text(0.03, 0.96, f"MC1: ρ = {rows[0]['rho']:+.2f}" + NL + f"free text: ρ = {rows[1]['rho']:+.2f}  (n={rows[0]['n']})",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2,
            bbox=dict(facecolor=SURFACE, alpha=0.85, edgecolor="none", pad=1.2))
    ax.set_ylabel("score (higher = more truthful)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax = axes[1, 0]
    y = df.loc[score.index, "HarmBench-DirectRequest-all"]
    ax.scatter(score, y, s=26, color=ORANGE, edgecolor=SURFACE, lw=0.8, zorder=3)
    r, n = rho(score, y)
    rows.append({"benchmark": "HarmBench, direct requests", "data": "Safetywashing, 2024 chat models",
                 "capability measure": "Safetywashing capabilities score", "rho": r, "n": n})
    ax.text(0.03, 0.96, f"ρ = {r:+.2f}  (n={n})", transform=ax.transAxes, va="top", fontsize=8.5, color=INK2,
            bbox=dict(facecolor=SURFACE, alpha=0.85, edgecolor="none", pad=1.2))
    ax.set_ylabel("attack failure rate (higher = refuses more)")
    ax.set_xlabel("Safetywashing capabilities score")
    for i, (col, ylab, bench) in enumerate([("truthfulqa", "IRT ability (higher = more truthful)", "TruthfulQA, free text (BLEURT judge)"),
                                           ("harmbench", "IRT ability (higher = refuses more)", "HarmBench, direct requests (Claude Haiku judge)")]):
        ax = axes[i, 1]
        ax.scatter(new.eci, new[col], s=26, color=BLUE, edgecolor=SURFACE, lw=0.8, zorder=3)
        r, n = rho(new.eci, new[col])
        rr, nr = rho(new.eci, new[col + "_released"])
        rows.append({"benchmark": bench, "data": "IRT data, primary sample (main analysis)", "capability measure": "ECI", "rho": r, "n": n})
        rows.append({"benchmark": bench, "data": "IRT data, primary sample (as released)", "capability measure": "ECI", "rho": rr, "n": nr})
        ax.text(0.03, 0.96, f"ρ = {r:+.2f}  (n={n})" + NL + f"as released: ρ = {rr:+.2f}  (n={nr})",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK2,
                bbox=dict(facecolor=SURFACE, alpha=0.85, edgecolor="none", pad=1.2))
        ax.set_ylabel(ylab)
    axes[1, 1].set_xlabel("ECI")
    axes[0, 0].set_title("Safetywashing (2024 chat models)" + NL + NL + "TruthfulQA", fontsize=10)
    axes[0, 1].set_title("IRT data (2023–26 models)" + NL + NL + "TruthfulQA (free text)", fontsize=10)
    axes[1, 0].set_title("HarmBench (direct requests)", fontsize=10)
    axes[1, 1].set_title("HarmBench (direct requests)", fontsize=10)
    fig.suptitle("The two benchmarks measured in both datasets", x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG / "sw_vs_now.png", dpi=180)
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "sw_vs_now.csv", index=False)
    print(out.round(2).to_string(index=False))


def main():
    headline_grid("chat")
    headline_grid("base")
    all_pdf()
    sw_vs_now()
    # sanity check: rho values must equal the paper's (e.g. TruthfulQA MC1 chat 81.2, base 69.7)
    for kind, paper in [("chat", 0.812), ("base", 0.697)]:
        df, score, _ = load(kind)
        r, _ = rho(score, df.loc[score.index, "truthfulqa_mc1"])
        assert abs(r - paper) < 0.001, (kind, r)
    print("check passed: TruthfulQA MC1 rho matches the paper for chat and base models")


if __name__ == "__main__":
    main()
