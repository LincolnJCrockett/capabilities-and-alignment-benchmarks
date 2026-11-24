"""Safetywashing-style analysis on new data: Epoch Capabilities Index (ECI) vs IRT-for-AI-Safety measures.

Safetywashing statistic: Spearman correlation across models between a capabilities score and each
safety measure ("capabilities correlation"). Here the capabilities score is ECI and the safety
measures are (a) per-benchmark 2PL abilities and (b) three factor scores, following Rivera et al. 2026.

Steps
  1. Validate our IRT refit against the paper (benchmark correlations, factor variance explained),
     on the data as released (that is what the paper analysed).
  2. Fit the paper's 3-factor minres/oblimin model on the released abilities (latent cohort) and use it to
     score every variant, so the three constructs mean the same thing throughout. Also refit the factor
     model on the main-analysis abilities to see whether the structure itself changes.
  3. Capabilities correlations for the primary sample, with bootstrap CIs.
     MAIN ANALYSIS (user decision 2026-11-22, journal Q21): empty-at-cap cells AND cells without a valid
     grader score set to missing. Sensitivity: as released (paper default); each masking alone;
     drop models with >10% empty-at-cap; add back non-exact crosswalk matches.

Outputs: reports/eci_irt/*.csv, printed summary.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from irt_data import load_matrix  # noqa: E402  (applies Health-ORSC orientation fix)
from factor_analyzer import FactorAnalyzer

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "reports" / "eci_irt"
OUT.mkdir(parents=True, exist_ok=True)
REPO = ROOT / "data" / "raw" / "irt" / "repo"
rng = np.random.default_rng(0)

BENCH = ["harmbench", "sorrybench", "ahb", "orbench_hard", "truthfulqa", "donotanswer", "advbench", "health_orsc"]
LABEL = {"harmbench": "HarmBench", "sorrybench": "SORRY-Bench", "ahb": "AHB", "orbench_hard": "OR-Bench-Hard",
         "truthfulqa": "TruthfulQA", "donotanswer": "Do-Not-Answer", "advbench": "AdvBench", "health_orsc": "Health-ORSC"}
# anchor benchmark that defines each factor's name and positive direction (paper, Results)
MAIN = "Main: empty-at-cap and grader failures missing"
ANCHORS = {"Refusal strictness": "harmbench", "Truthfulness": "truthfulqa", "Contextual harm": "advbench"}


def spearman_ci(x, y, B=2000):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(d)
    rho = d.corr("spearman").iloc[0, 1]
    idx = rng.integers(0, n, (B, n))
    xr, yr = d.x.to_numpy(), d.y.to_numpy()
    boots = [pd.Series(xr[i]).corr(pd.Series(yr[i]), method="spearman") for i in idx]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, lo, hi, n


VARIANT_FILES = ("released", "cap_masked", "grader_masked", "main")


def load_inputs():
    th = {v: pd.read_csv(INTERIM / f"irt_abilities_{v}.csv", index_col="model").reindex(columns=BENCH)
          for v in VARIANT_FILES}
    cw = pd.read_csv(INTERIM / "model_crosswalk.csv")
    irt = cw[(cw.source == "irt") & cw.epoch_model.notna()].copy()
    irt["primary"] = (irt.match_type == "exact") & (irt.confidence == "high")
    flags = pd.read_parquet(INTERIM / "irt_cell_flags.parquet")
    flags = flags[flags.matrix_value.notna()]
    pct_cap = flags.groupby("model").kind.apply(lambda s: (s == "empty_at_cap").mean() * 100)
    return th, irt, pct_cap


def cohorts(th):
    """Cross-benchmark cohort = models with all 8 benchmarks; latent cohort = also complete on every item."""
    items = pd.read_json(REPO / "full_battery" / "items.jsonl", lines=True)
    manifest = json.loads((REPO / "matrices" / "manifest.json").read_text())["datasets"]
    cross = th.dropna().index
    complete = set(cross)
    for ds, meta in manifest.items():
        m = load_matrix(ds)[items.loc[items.dataset == ds, "question_id"]]
        complete &= set(m.index[m.notna().all(axis=1)])
    return list(cross), sorted(complete)


def factor_model(theta, n=3):
    """Fit minres/oblimin FA; name + orient factors by anchor benchmark.
    Returns (scorer, loadings, scores); scorer(theta_df) scores new data with the SAME fitted model."""
    fa = FactorAnalyzer(n_factors=n, method="minres", rotation="oblimin" if n > 1 else None)
    fa.fit(theta)
    load = pd.DataFrame(fa.loadings_, index=theta.columns)
    names, signs = {j: j for j in load.columns}, {j: 1 for j in load.columns}
    if n == 3:
        used = set()
        for name, anchor in ANCHORS.items():
            j = load.loc[anchor].abs().drop(list(used)).idxmax()
            used.add(j)
            names[j] = name
            signs[j] = -1 if load.loc[anchor, j] < 0 else 1

    def scorer(t):
        s = pd.DataFrame(fa.transform(t[theta.columns]), index=t.index)
        s = s * pd.Series(signs)
        return s.rename(columns=names)[list(ANCHORS)] if n == 3 else s

    load = (load * pd.Series(signs)).rename(columns=names)
    load = load[list(ANCHORS)] if n == 3 else load
    return scorer, load, scorer(theta)


def main():
    th, irt, pct_cap = load_inputs()
    cross, latent = cohorts(th["released"])
    print(f"cohorts: cross-benchmark {len(cross)} (paper 169), latent {len(latent)} (paper 134)\n")

    # ---------- 1. validation vs paper ----------
    T = th["released"].loc[cross]
    rho = T.corr("spearman")
    checks = [("harmbench", "sorrybench", "0.89"), ("harmbench", "orbench_hard", "-0.77 to -0.67"),
              ("sorrybench", "orbench_hard", "-0.77 to -0.67"), ("advbench", "health_orsc", "0.58")]
    val = [{"pair": f"{LABEL[a]} - {LABEL[b]}", "ours": round(rho.loc[a, b], 2), "paper": p} for a, b, p in checks]
    TL = th["released"].loc[latent]
    for n, paper in [(1, "47%"), (2, "69%"), (3, "77%")]:
        fa = FactorAnalyzer(n_factors=n, method="minres", rotation="oblimin" if n > 1 else None).fit(TL)
        val.append({"pair": f"variance explained, {n} factor(s)", "ours": f"{fa.get_factor_variance()[2][-1]:.0%}", "paper": paper})
    val = pd.DataFrame(val)
    val.to_csv(OUT / "validation_vs_paper.csv", index=False)
    print(val.to_string(index=False), "\n")
    rho.rename(index=LABEL, columns=LABEL).round(2).to_csv(OUT / "benchmark_ability_correlations.csv")

    # ---------- 2. factor scores ----------
    # Constructs are defined once, on the released abilities (the paper's data). Every variant is scored
    # with that SAME factor model, on latent-cohort models that have all 8 abilities in that variant.
    factor_scores, loadings = {}, {}
    scorer, loadings["released"], _ = factor_model(th["released"].loc[latent])
    # Factor scores are linear in the 8 abilities. Recover the weights (per 1 SD of each benchmark, on the
    # released latent cohort) so the report can show exactly how each factor score is composed.
    TLr = th["released"].loc[latent]
    Z = (TLr - TLr.mean()) / TLr.std(ddof=0)
    S = scorer(TLr)
    W = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z.to_numpy()], S.to_numpy(), rcond=None)[0][1:]
    weights = pd.DataFrame(W, index=[LABEL[b] for b in TLr.columns], columns=S.columns)
    weights.round(4).to_csv(OUT / "factor_score_weights.csv")
    print("Factor-score weights per SD of each benchmark:")
    print(weights.round(2).to_string(), "\n")
    for v in VARIANT_FILES:
        lat_v = [m for m in latent if m in th[v].dropna().index]
        factor_scores[v] = scorer(th[v].loc[lat_v])
    lat_main = list(factor_scores["main"].index)
    _, loadings["main_refit"], _ = factor_model(th["main"].loc[lat_main])
    fa_refit = FactorAnalyzer(n_factors=3, method="minres", rotation="oblimin").fit(th["main"].loc[lat_main])
    for v, load in loadings.items():
        load.rename(index=LABEL).round(2).to_csv(OUT / f"factor_loadings_{v}.csv")
    for v, fs in factor_scores.items():
        fs.rename_axis("model").to_csv(OUT / f"factor_scores_{v}.csv")
    print(f"factor-score cohort: released {len(factor_scores['released'])}, main {len(lat_main)}")
    print("3-factor loadings (fitted on released abilities; used for all variants):")
    print(loadings["released"].rename(index=LABEL).round(2).to_string(), "\n")
    print(f"3-factor loadings refit on main-analysis abilities (structure check; "
          f"variance explained {fa_refit.get_factor_variance()[2][-1]:.0%}):")
    print(loadings["main_refit"].rename(index=LABEL).round(2).to_string(), "\n")

    # ---------- 3. capabilities correlations ----------
    sample = {
        "primary": irt[irt.primary].drop_duplicates("epoch_model"),
        "primary + non-exact": irt.drop_duplicates("epoch_model"),
    }
    heavy = set(pct_cap[pct_cap > 10].index)
    variants = [
        (MAIN, "main", "primary", False),
        ("As released (paper default)", "released", "primary", False),
        ("Only empty-at-cap cells missing", "cap_masked", "primary", False),
        ("Only grader failures missing", "grader_masked", "primary", False),
        ("Main, dropping models with >10% empty-at-cap", "main", "primary", True),
        ("Main, adding non-exact matches", "main", "primary + non-exact", False),
    ]
    rows = []
    for vname, thv, samp, drop_heavy in variants:
        s = sample[samp].set_index("source_model")
        if drop_heavy:
            s = s.loc[[m for m in s.index if m not in heavy]]
        for b in BENCH:
            models = [m for m in s.index if m in th[thv].index and pd.notna(th[thv].at[m, b])]
            r, lo, hi, n = spearman_ci(s.loc[models, "eci"], th[thv].loc[models, b])
            rows.append({"variant": vname, "measure": LABEL[b], "kind": "benchmark", "rho": r, "lo": lo, "hi": hi, "n": n})
        fs = factor_scores[thv]
        for f in fs.columns:
            models = [m for m in s.index if m in fs.index]
            r, lo, hi, n = spearman_ci(s.loc[models, "eci"], fs.loc[models, f])
            rows.append({"variant": vname, "measure": f, "kind": "factor", "rho": r, "lo": lo, "hi": hi, "n": n})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "capabilities_correlations.csv", index=False)
    show = res.assign(cell=res.apply(lambda x: f"{x.rho:+.2f} [{x.lo:+.2f},{x.hi:+.2f}] n={x.n}", axis=1))
    order = [LABEL[b] for b in BENCH] + list(ANCHORS)
    pd.set_option("display.width", 250)
    print(show.pivot(index="measure", columns="variant", values="cell").loc[order, [v[0] for v in variants]].to_string(), "\n")

    # which primary models lose a measure under the main analysis (selection check)
    prim = sample["primary"].set_index("source_model")
    lost = []
    for b in BENCH:
        had = [m for m in prim.index if m in th["released"].index and pd.notna(th["released"].at[m, b])]
        gone = [m for m in had if not (m in th["main"].index and pd.notna(th["main"].at[m, b]))]
        lost.append({"measure": LABEL[b], "n_released": len(had), "n_main": len(had) - len(gone),
                     "median_eci_dropped": prim.loc[gone, "eci"].median() if gone else np.nan,
                     "median_eci_kept": prim.loc[[m for m in had if m not in gone], "eci"].median(),
                     "dropped": "; ".join(gone)})
    lost = pd.DataFrame(lost)
    lost.to_csv(OUT / "main_analysis_dropped_models.csv", index=False)
    print("Primary-sample models without a main-analysis ability (fewer than half their items left):")
    print(lost.drop(columns="dropped").round(1).to_string(index=False), "\n")

    # per-model table for plotting (main analysis; released kept for comparison)
    p = prim[["epoch_model", "eci", "release_date"]]
    p = (p.join(th["main"]).join(factor_scores["main"]).join(pct_cap.rename("pct_empty_at_cap"))
         .join(th["released"].add_suffix("_released")))
    p.to_csv(OUT / "primary_sample_scores.csv")


if __name__ == "__main__":
    main()
