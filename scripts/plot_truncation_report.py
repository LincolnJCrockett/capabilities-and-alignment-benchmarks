"""Figures + summary numbers for the token-cap / empty-response write-up.

Inputs : data/interim/irt_cell_flags.parquet, data/interim/model_crosswalk.csv, IRT matrices
Outputs: reports/figures/fig{1..4}_*.png, reports/truncation_numbers.csv (fig 4 values)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from irt_data import load_matrix  # noqa: E402  (applies Health-ORSC orientation fix)

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data" / "raw" / "irt" / "repo"
INTERIM = ROOT / "data" / "interim"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- style (reference palette, light mode; first 3 categorical slots validate all-pairs) ---
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

# Display names, grouped by the IRT paper's three factors
BENCH = [  # (key, label, factor, desired behavior coded as 1)
    ("harmbench", "HarmBench", "Refusal strictness", "refuse"),
    ("sorrybench", "SORRY-Bench", "Refusal strictness", "refuse"),
    ("ahb", "AHB", "Refusal strictness", "refuse"),
    ("orbench_hard", "OR-Bench-Hard", "Refusal strictness", "answer (benign)"),
    ("truthfulqa", "TruthfulQA", "Truthfulness", "truthful answer"),
    ("donotanswer", "Do-Not-Answer", "Truthfulness", "safe response"),
    ("advbench", "AdvBench", "Contextual harm", "refuse"),
    ("health_orsc", "Health-ORSC", "Contextual harm", "refuse"),
]
LABEL = {k: l for k, l, _, _ in BENCH}
ORDER = [k for k, *_ in BENCH]

manifest = json.loads((REPO / "matrices" / "manifest.json").read_text())["datasets"]
mats = {ds: load_matrix(ds) for ds, m in manifest.items()}

flags = pd.read_parquet(INTERIM / "irt_cell_flags.parquet")
flags = flags[flags.matrix_value.notna()]  # only cells that enter the released matrices

cw = pd.read_csv(INTERIM / "model_crosswalk.csv")
prim = (cw[(cw.source == "irt") & (cw.match_type == "exact") & (cw.confidence == "high")]
        .drop_duplicates("epoch_model").set_index("source_model"))

# ---------------- Fig 1: composition per benchmark ----------------
comp = (flags.groupby(["dataset", "kind"], observed=True).size().unstack(fill_value=0))
comp = comp.div(comp.sum(axis=1), axis=0) * 100
comp = comp.loc[ORDER[::-1]]
fig, ax = plt.subplots(figsize=(7.5, 4.3))
left = np.zeros(len(comp))
parts = [("empty_at_cap", "Empty: hit 1,024-token cap", ORANGE),
         ("truncated_with_text", "Truncated: partial text", BLUE),
         ("empty_other", "Empty: other (e.g. API refusal)", AQUA)]
for col, lab, c in parts:
    ax.barh([LABEL[d] for d in comp.index], comp[col], left=left, color=c, label=lab,
            height=0.6, edgecolor=SURFACE, linewidth=2)
    left += comp[col].values
for y, (d, tot) in enumerate(zip(comp.index, left)):
    ax.text(tot + 0.6, y, f"{tot:.0f}%", va="center", color=INK2, fontsize=9)
ax.set_xlabel("% of scored responses (all 192 models)")
ax.set_xlim(0, max(left) * 1.15)
ax.grid(axis="y", visible=False)
ax.set_title("Share of responses affected by the 1,024-token cap")
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig1_composition_by_benchmark.png", dpi=200)
plt.close(fig)

# ---------------- Fig 2: how empty-at-cap responses are scored ----------------
ec = flags[flags.kind == "empty_at_cap"].groupby("dataset").matrix_value.agg(["mean", "size"])
ec = ec.loc[ORDER[::-1]]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.barh([LABEL[d] for d in ec.index], ec["mean"] * 100, color=BLUE, height=0.6)
desired = {k: d for k, _, _, d in BENCH}
for y, d in enumerate(ec.index):
    v = ec.loc[d, "mean"] * 100
    ax.text(v + 1.5, y, f"{v:.0f}%   (1 = {desired[d]}; n={int(ec.loc[d, 'size']):,})",
            va="center", color=INK2, fontsize=8.5)
ax.set_xlim(0, 160)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("% of empty-at-cap responses scored 1 (the benchmark's 'desired' outcome)")
ax.grid(axis="y", visible=False)
ax.set_title("Empty responses are scored as refusals")
fig.tight_layout()
fig.savefig(FIG / "fig2_scoring_of_empty_responses.png", dpi=200)
plt.close(fig)

# ---------------- Fig 3: ECI vs % empty-at-cap (primary sample) ----------------
per_model = flags.groupby("model").kind.apply(lambda s: 100 * (s == "empty_at_cap").mean())
p = prim.join(per_model.rename("pct_cap"), how="inner")
rho3 = p[["eci", "pct_cap"]].corr("spearman").iloc[0, 1]
fig, ax = plt.subplots(figsize=(7.5, 4.9))
ax.scatter(p.eci, p.pct_cap, s=42, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
OFFSETS = {"Qwen3.5-35B-A3B": (-8, -2, "right"), "GLM-5": (6, 3, "left"),
           "Qwen3.5 397B-A17B": (6, -7, "left"), "Qwen3.6 27B": (-8, -5, "right"),
           "GLM-5.1": (6, 2, "left"), "GLM-4.7": (0, 9, "center")}
for m, r in p[p.pct_cap > 18].iterrows():
    dx, dy, ha = OFFSETS.get(r.epoch_model, (6, 0, "left"))
    ax.annotate(r.epoch_model, (r.eci, r.pct_cap), xytext=(dx, dy), textcoords="offset points",
                fontsize=8, color=INK2, va="center", ha=ha)
ax.set_xlabel("Epoch Capabilities Index (ECI)")
ax.set_ylabel("% of responses empty at token cap")
ax.set_title("More capable models hit the cap more often")
fig.tight_layout()
fig.savefig(FIG / "fig3_eci_vs_empty_at_cap.png", dpi=200)
plt.close(fig)

# ---------------- Fig 4: ECI-benchmark correlation under three codings ----------------
rng = np.random.default_rng(0)
cap = flags[flags.kind == "empty_at_cap"]
heavy = set(p.index[p.pct_cap > 10])


def spearman_ci(x, y, B=2000):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    rho = d.corr("spearman").iloc[0, 1]
    n = len(d)
    boots = [d.iloc[rng.integers(0, n, n)].corr("spearman").iloc[0, 1] for _ in range(B)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, lo, hi, n


rows = []
for ds in ORDER:
    mat = mats[ds]
    models = [m for m in p.index if m in mat.index]
    released = mat.loc[models].mean(axis=1)
    masked = mat.loc[models].copy()
    cc = cap[(cap.dataset == ds) & cap.model.isin(models)]
    for m, i in zip(cc.model, cc.item_id):
        masked.at[m, i] = np.nan
    masked = masked.mean(axis=1)
    eci = p.loc[models, "eci"]
    for variant, score, keep in [
        ("As released (paper default)", released, models),
        ("Empty-at-cap cells set to missing", masked, models),
        ("Drop models with >10% empty-at-cap", released, [m for m in models if m not in heavy]),
    ]:
        rho, lo, hi, n = spearman_ci(eci[keep], score[keep])
        rows.append({"dataset": ds, "variant": variant, "rho": rho, "lo": lo, "hi": hi, "n": n})
res = pd.DataFrame(rows)
res.to_csv(ROOT / "reports" / "truncation_numbers.csv", index=False)

variants = res.variant.unique()
colors = [BLUE, ORANGE, AQUA]
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ys = np.arange(len(ORDER))[::-1]
for j, (v, c) in enumerate(zip(variants, colors)):
    r = res[res.variant == v].set_index("dataset").loc[ORDER]
    off = (1 - j) * 0.22
    ax.errorbar(r.rho, ys + off, xerr=[r.rho - r.lo, r.hi - r.rho], fmt="o", color=c,
                ms=6, mec=SURFACE, mew=1.5, elinewidth=1.4, capsize=0, label=v)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(ys)
ax.set_yticklabels([LABEL[d] for d in ORDER])
ax.set_xlim(-1, 1)
ax.set_xlabel("Spearman ρ between ECI and benchmark score (mean of 0/1 items), 95% bootstrap CI")
ax.grid(axis="y", visible=False)
for y0, y1, name in [(4, 7, "Refusal strictness"), (2, 3, "Truthfulness"), (0, 1, "Contextual harm")]:
    ax.axhspan(y0 - 0.45, y1 + 0.45, color=GRID, alpha=0.35 if name != "Truthfulness" else 0, lw=0)
    ax.text(0.98, y1 + 0.32, name, transform=ax.get_yaxis_transform(), ha="right",
            fontsize=8, color=INK2, style="italic")
ax.set_title("Does the capability–safety correlation depend on how\nempty-at-cap responses are coded?")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig4_correlation_sensitivity.png", dpi=200)
plt.close(fig)

print(res.pivot(index="dataset", columns="variant", values="rho").loc[ORDER].round(2).to_string())
print(res.groupby("variant").n.agg(["min", "max"]))
print("heavy models:", sorted(heavy))
