"""Show-the-data figure, open vs closed weights, and developers, for the ECI x IRT analysis.

Uses the MAIN-ANALYSIS abilities (empty-at-cap and grader-failure cells missing; journal Q21) and the
factor scores from the paper's factor model (reports/eci_irt/factor_scores_main.csv).

Inputs : data/interim/model_crosswalk.csv, data/interim/irt_abilities_main.csv,
         reports/eci_irt/factor_scores_main.csv, Epoch ECI file, data/interim/irt_cell_flags.parquet
Outputs: reports/eci_irt/{open_closed,open_closed_by_lab,within_developer,developer_means}.csv
         reports/figures/{safety_vs_eci,factor_scatter,open_closed_correlations,open_closed_by_lab,developers}.png

Definitions
  open / closed : Epoch "Accessibility group" (Open weights vs Closed weights; 'Other' excluded)
  developer : Epoch "Organization" (Google variants merged); developers with >= 3 models
  within-developer rho : Pearson correlation of developer-demeaned ranks (removes between-developer
                         differences; answers "within a lab, do more capable models score higher?")
All correlations are Spearman unless noted; CIs are 95% percentile bootstrap (2,000 resamples).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
RES = ROOT / "reports" / "eci_irt"
FIG = ROOT / "reports" / "figures"
B = 2000
rng = np.random.default_rng(0)

FACT = ["Refusal strictness", "Truthfulness", "Contextual harm"]
BENCH = {"harmbench": "HarmBench", "sorrybench": "SORRY-Bench", "ahb": "AHB", "orbench_hard": "OR-Bench-Hard",
         "truthfulqa": "TruthfulQA", "donotanswer": "Do-Not-Answer", "advbench": "AdvBench", "health_orsc": "Health-ORSC"}
MEASURES = FACT + list(BENCH.values())   # tables (factors included)
PLOT = list(BENCH.values())              # figures: benchmarks only (factor scores mix all 8; journal)

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
OPEN_C, CLOSED_C = AQUA, ORANGE      # open/closed weights
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


# ---------------- data ----------------
def build():
    cw = pd.read_csv(INTERIM / "model_crosswalk.csv")
    irt = cw[(cw.source == "irt") & cw.epoch_model.notna()].copy()
    irt["primary"] = (irt.match_type == "exact") & (irt.confidence == "high")
    irt = irt.drop_duplicates("epoch_model").set_index("source_model")
    eci = pd.read_csv(ROOT / "data/raw/epoch/extracted/epoch_capabilities_index/eci_scores.csv").set_index("Model")
    th = pd.read_csv(INTERIM / "irt_abilities_main.csv", index_col=0).rename(columns=BENCH)
    fs = pd.read_csv(RES / "factor_scores_main.csv", index_col=0)
    flags = pd.read_parquet(INTERIM / "irt_cell_flags.parquet")
    flags = flags[flags.matrix_value.notna()]
    cap = flags.groupby("model").kind.apply(lambda s: 100 * (s == "empty_at_cap").mean()).rename("pct_cap")

    d = irt[["epoch_model", "eci", "primary"]].join(eci[["date", "Organization", "Accessibility group"]], on="epoch_model")
    d["date"] = pd.to_datetime(d["date"])
    d["developer"] = d.Organization.replace({"Google DeepMind,Google": "Google DeepMind", "Google": "Google DeepMind"})
    d["access"] = d["Accessibility group"].map({"Open weights": "open", "Closed weights": "closed"})
    return d.join(th).join(fs).join(cap)


# ---------------- statistics ----------------
def _rank_rows(a):
    return rankdata(a, axis=-1)


def _pearson_rows(x, y):
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    return (x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1))


def spearman_boot(x, y):
    """rho, bootstrap replicates (B,) for paired arrays; NaN pairs dropped."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    n = len(x)
    rho = _pearson_rows(rankdata(x), rankdata(y))
    idx = rng.integers(0, n, (B, n))
    with np.errstate(invalid="ignore"):
        boots = _pearson_rows(_rank_rows(x[idx]), _rank_rows(y[idx]))
    return rho, boots, n


def ci(boots):
    return np.nanpercentile(boots, [2.5, 97.5])


def within_developer(df, m, min_n=3):
    """Correlation of developer-demeaned ranks of ECI and measure; bootstrap stratified by developer."""
    sub = df[["eci", m, "developer"]].dropna()
    keep = sub.developer.value_counts()
    sub = sub[sub.developer.isin(keep[keep >= min_n].index)]

    def stat(s):
        rk = s[["eci", m]].rank()
        rk = rk - rk.groupby(s.developer.values).transform("mean")
        return np.corrcoef(rk.eci, rk[m])[0, 1]
    rho = stat(sub)
    groups = [g for _, g in sub.groupby("developer")]
    boots = []
    for _ in range(B):
        s = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for g in groups])
        boots.append(stat(s))
    return rho, ci(np.array(boots)), len(sub), sub.developer.nunique()


def compare_groups(df, col, a, b, label):
    rows = []
    for m in MEASURES:
        out = {"sample": label, "measure": m}
        reps = {}
        for g in (a, b):
            s = df[df[col] == g]
            rho, boots, n = spearman_boot(s.eci.to_numpy(float), s[m].to_numpy(float))
            lo, hi = ci(boots)
            out.update({f"rho_{g}": rho, f"lo_{g}": lo, f"hi_{g}": hi, f"n_{g}": n})
            reps[g] = boots
        diff = reps[b] - reps[a]
        out["diff"] = out[f"rho_{b}"] - out[f"rho_{a}"]
        out["diff_lo"], out["diff_hi"] = ci(diff)
        rows.append(out)
    return pd.DataFrame(rows)


# ---------------- figures ----------------
def fit_line(ax, x, y, color):
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() >= 3:
        b = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 20)
        ax.plot(xs, np.polyval(b, xs), color=color, lw=1.6, zorder=2)


def grid_axes():
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10))
    return fig, axes.ravel()


def ylabel_for(m):
    return "factor score" if m in FACT else "IRT ability (θ)"


def fig_safety_vs_eci(d):
    fig, axes = grid_axes()
    for ax, m in zip(axes, PLOT):
        lines = []
        rho, _, n = spearman_boot(d.eci.to_numpy(float), d[m].to_numpy(float))
        lines.append(f"all: ρ = {rho:+.2f} (n={n})")
        for g, c in [("open", OPEN_C), ("closed", CLOSED_C)]:
            s_ = d[d.access == g]
            ax.scatter(s_.eci, s_[m], s=26, color=c, edgecolor=SURFACE, lw=0.9, zorder=3)
            fit_line(ax, s_.eci.to_numpy(float), s_[m].to_numpy(float), c)
            r_g, _, n_g = spearman_boot(s_.eci.to_numpy(float), s_[m].to_numpy(float))
            lines.append(f"{g}: ρ = {r_g:+.2f} (n={n_g})")
        other = d[d.access.isna()]
        ax.scatter(other.eci, other[m], s=26, color=MUTED, edgecolor=SURFACE, lw=0.9, zorder=3)
        ax.set_title(m)
        ax.set_xlabel("ECI", fontsize=8.5)
        ax.text(0.02, 0.97, "\n".join(lines),
                transform=ax.transAxes, va="top", fontsize=8, color=INK2, zorder=5,
                bbox=dict(facecolor=SURFACE, alpha=0.85, edgecolor="none", pad=1.5))
        ax.set_ylabel(ylabel_for(m), fontsize=8.5)
    leg = axes[-1]
    leg.axis("off")
    leg.scatter([], [], s=40, color=OPEN_C, label="Open weights")
    leg.scatter([], [], s=40, color=CLOSED_C, label="Closed weights")
    leg.legend(loc="upper left", frameon=False, fontsize=9.5)
    leg.scatter([], [], s=40, color=MUTED, label="Neither (excluded from groups)")
    leg.legend(loc="upper left", frameon=False, fontsize=9.5)
    leg.text(0.0, 0.05, "Lines: least-squares fit within each group,\nin the group's colour. ρ: Spearman.\n"
             "Group CIs: open_closed_correlations.png.\n"
             "Higher = more of the measured behavior\n(refusing harmful requests; answering\n"
             "benign ones on OR-Bench; truthful answers).\nEmpty-at-cap and ungraded responses\n"
             "are treated as missing.",
             transform=leg.transAxes, fontsize=8.5, color=INK2)
    fig.suptitle("Safety benchmarks against capabilities (primary sample)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "safety_vs_eci.png", dpi=180)
    plt.close(fig)


def fig_factor_scatter(d):
    """Factor scores against ECI (report's separate factor section). Factor scores weight all 8 benchmarks."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharex=True)
    for ax, f in zip(axes, FACT):
        lines = []
        rho, _, n = spearman_boot(d.eci.to_numpy(float), d[f].to_numpy(float))
        lines.append(f"all: ρ = {rho:+.2f} (n={n})")
        for g, c in [("open", OPEN_C), ("closed", CLOSED_C)]:
            s_ = d[d.access == g]
            ax.scatter(s_.eci, s_[f], s=30, color=c, edgecolor=SURFACE, lw=0.9, zorder=3,
                       label="Open weights" if g == "open" else "Closed weights")
            fit_line(ax, s_.eci.to_numpy(float), s_[f].to_numpy(float), c)
            r_g, _, n_g = spearman_boot(s_.eci.to_numpy(float), s_[f].to_numpy(float))
            lines.append(f"{g}: ρ = {r_g:+.2f} (n={n_g})")
        other = d[d.access.isna()]
        ax.scatter(other.eci, other[f], s=30, color=MUTED, edgecolor=SURFACE, lw=0.9, zorder=3)
        ax.set_title(f)
        ax.set_xlabel("ECI")
        ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8, color=INK2, zorder=5,
                bbox=dict(facecolor=SURFACE, alpha=0.85, edgecolor="none", pad=1.5))
    axes[0].set_ylabel("factor score")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("IRT factor scores against capabilities (primary models with all 8 benchmarks; "
                 "lines = fit within each group)", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "factor_scatter.png", dpi=180)
    plt.close(fig)


def dot_compare(tab, a, b, ca, cb, la, lb, title, fname):
    t = tab.set_index("measure").loc[PLOT]
    ys = np.arange(len(PLOT))[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.axvline(0, color=INK2, lw=1)
    for g, c, lab, off in [(a, ca, la, 0.16), (b, cb, lb, -0.16)]:
        ax.errorbar(t[f"rho_{g}"], ys + off, xerr=[t[f"rho_{g}"] - t[f"lo_{g}"], t[f"hi_{g}"] - t[f"rho_{g}"]],
                    fmt="o", color=c, ms=6, mec=SURFACE, mew=1.2, elinewidth=1.4, capsize=0,
                    label=f"{lab} (n = {int(t[f'n_{g}'].min())}–{int(t[f'n_{g}'].max())})")
    for y, (m, r) in zip(ys, t.iterrows()):
        sig = "" if r.diff_lo <= 0 <= r.diff_hi else " *"
        ax.text(1.04, y, f"{r['diff']:+.2f}{sig}", va="center", fontsize=8.5, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.set_ylim(-0.7, len(PLOT) - 0.1)
    ax.text(1.04, len(PLOT) - 0.45, "diff.", fontsize=8.5, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(ys)
    ax.set_yticklabels(PLOT)
    ax.set_xlim(-1, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Spearman ρ with ECI (95% bootstrap CI)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8.5)
    fig.text(0.99, 0.01, "* = 95% CI of the difference excludes 0", ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def fig_developers(d, wd):
    counts = d.developer.value_counts()
    devs = counts[counts >= 3].index
    sub = d[d.developer.isin(devs)]
    order = sub.groupby("developer").eci.mean().sort_values().index
    ys = {dev: i for i, dev in enumerate(order)}
    fig, axes = plt.subplots(3, 3, figsize=(12, 10.5), sharey=True)
    axes = axes.ravel()
    w = wd.set_index("measure")
    for ax, col in zip(axes, ["eci"] + PLOT):
        for dev in order:
            s_ = sub[sub.developer == dev][col].dropna()
            if s_.empty:
                continue
            jitter = rng.uniform(-0.18, 0.18, len(s_))
            ax.scatter(s_, ys[dev] + jitter, s=22, color=BLUE, edgecolor=SURFACE, lw=0.8, zorder=3)
            ax.plot([s_.mean()] * 2, [ys[dev] - 0.32, ys[dev] + 0.32], color=INK, lw=2, zorder=4)
        if col == "eci":
            ax.set_title("ECI", fontsize=10)
            ax.set_xlabel("ECI", fontsize=8.5)
        else:
            ax.set_title(col + chr(10) + f"within-developer ρ = {w.loc[col, 'rho_within']:+.2f}", fontsize=10)
            ax.set_xlabel("IRT ability (θ)", fontsize=8.5)
        ax.grid(axis="y", visible=False)
    for ax in axes[::3]:
        ax.set_yticks(list(ys.values()))
        ax.set_yticklabels([f"{dev} ({counts[dev]})" for dev in order], fontsize=8.5)
    fig.suptitle("Models by developer (developers with ≥3 primary models; ordered by mean ECI; bar = developer mean)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "developers.png", dpi=180)
    plt.close(fig)


GAP_BENCH = ["TruthfulQA", "Do-Not-Answer", "SORRY-Bench", "OR-Bench-Hard"]  # largest open/closed gaps (|diff| >= 0.39)
MIN_LAB = 4  # a Spearman rho needs at least 4 models to mean anything


def by_lab(d):
    """Per-lab Spearman rho with ECI, within open and within closed models, for the gap benchmarks."""
    rows = []
    for acc in ("open", "closed"):
        g_acc = d[d.access == acc]
        for b in GAP_BENCH:
            s = g_acc[["eci", b]].dropna()
            rows.append({"access": acc, "lab": f"All {acc}-weight", "benchmark": b, "n": len(s),
                         "rho": s.eci.corr(s[b], method="spearman"), "mean_score": s[b].mean(),
                         "eci_min": s.eci.min(), "eci_max": s.eci.max()})
        for lab, g in g_acc.groupby("developer"):
            for b in GAP_BENCH:
                s = g[["eci", b]].dropna()
                rows.append({"access": acc, "lab": lab, "benchmark": b, "n": len(s),
                             "rho": s.eci.corr(s[b], method="spearman") if len(s) >= MIN_LAB else np.nan,
                             "mean_score": s[b].mean() if len(s) else np.nan,
                             "eci_min": s.eci.min() if len(s) else np.nan, "eci_max": s.eci.max() if len(s) else np.nan})
    return pd.DataFrame(rows)


def fig_by_lab(t):
    labs = {acc: (t[(t.access == acc) & ~t.lab.str.startswith("All") & (t.n >= MIN_LAB)]
                  .groupby("lab").n.max().sort_values(ascending=False).index.tolist()) for acc in ("open", "closed")}
    order = [("open", f"All open-weight")] + [("open", l) for l in labs["open"]] + \
            [("closed", f"All closed-weight")] + [("closed", l) for l in labs["closed"]]
    ys = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, len(GAP_BENCH), figsize=(13, 4.6), sharey=True)
    for ax, b in zip(axes, GAP_BENCH):
        tb = t[t.benchmark == b].set_index(["access", "lab"])
        ax.axvline(0, color=INK2, lw=1)
        ax.axhline(len(order) - 1 - len(labs["open"]) - 0.5, color=INK2, lw=0.8)
        for y, key in zip(ys, order):
            if key not in tb.index or pd.isna(tb.loc[key, "rho"]):
                continue
            r = tb.loc[key]
            c = OPEN_C if key[0] == "open" else CLOSED_C
            pooled = key[1].startswith("All")
            ax.scatter(r.rho, y, s=90 if pooled else 55, color=c, edgecolor=SURFACE, lw=1.5, zorder=3,
                       marker="D" if pooled else "o")
            ax.text(1.04, y, f"n={int(r.n)}", va="center", fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
        ax.set_xlim(-1.08, 1.08)
        ax.set_title(b)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Spearman ρ with ECI")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, lab in order])
    for lab in axes[0].get_yticklabels():
        if lab.get_text().startswith("All"):
            lab.set_fontweight("bold")
    fig.suptitle(f"Capabilities correlations within each lab, for the benchmarks where open and closed models differ most\n"
                 f"(primary sample; labs with ≥{MIN_LAB} models in the group; diamonds = all models in the group)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "open_closed_by_lab.png", dpi=180)
    plt.close(fig)


def main():
    d_all = build()
    d = d_all[d_all.primary]
    samples = {"primary": d, "primary + non-exact": d_all}

    # open / closed
    oc = pd.concat([compare_groups(s[s.access.notna()], "access", "open", "closed", k) for k, s in samples.items()])
    oc.to_csv(RES / "open_closed.csv", index=False)

    # developers
    wd = []
    for m in MEASURES:
        rho, (lo, hi), n, k = within_developer(d, m)
        overall, _, _ = spearman_boot(d.eci.to_numpy(float), d[m].to_numpy(float))
        wd.append({"measure": m, "rho_overall": overall, "rho_within": rho, "lo": lo, "hi": hi,
                   "n_models": n, "n_developers": k})
    wd = pd.DataFrame(wd)
    wd.to_csv(RES / "within_developer.csv", index=False)
    dm = d.groupby("developer").agg(n=("eci", "size"), mean_eci=("eci", "mean"),
                                    **{f: (f, "mean") for f in FACT}).sort_values("mean_eci", ascending=False)
    dm.to_csv(RES / "developer_means.csv")

    # figures (primary sample)
    fig_safety_vs_eci(d)
    fig_factor_scatter(d)
    dot_compare(oc[oc["sample"] == "primary"], "open", "closed", OPEN_C, CLOSED_C,
                "Open weights", "Closed weights",
                "Capabilities correlations: open-weight vs closed models (primary sample)", "open_closed_correlations.png")
    fig_developers(d, wd)
    bl = by_lab(d)
    bl.to_csv(RES / "open_closed_by_lab.csv", index=False)
    fig_by_lab(bl)

    pd.set_option("display.width", 220)
    show = lambda t, cols: print(t[cols].round(2).to_string(index=False), "\n")
    print("ACCESS GROUPS (primary):", d.access.value_counts(dropna=False).to_dict(), "\n")
    print("OPEN vs CLOSED"); show(oc, ["sample", "measure", "rho_open", "lo_open", "hi_open", "n_open", "rho_closed",
                                       "lo_closed", "hi_closed", "n_closed", "diff", "diff_lo", "diff_hi"])
    print("WITHIN DEVELOPER"); show(wd, ["measure", "rho_overall", "rho_within", "lo", "hi", "n_models", "n_developers"])
    print(dm.round(2).to_string())


if __name__ == "__main__":
    main()
