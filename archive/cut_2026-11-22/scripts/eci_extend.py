"""Put models that have no official ECI onto the ECI scale, so that every Safetywashing and IRT model
can be compared on the same capability axis.

Epoch's ECI model (verified against eci_bootstraps.json; see notes/journal.md Q19):
    performance[m, b] = sigmoid(slope[b] * (ECI[m] - difficulty[b]))
where performance = clip((score*scale - random_baseline) / (ceiling - random_baseline), 0, 1),
max over duplicate rows of a model group. Benchmark parameters (difficulty, slope) are released
per bootstrap sample, already on the ECI scale.

Steps
  1. assemble : rebuild Epoch's processed data from the raw benchmark files for ALL model groups
                (not only the 257 with an official ECI); check it matches processed_data_for_eci.csv.
  2. project  : for a model, estimate ECI by least squares with Epoch's benchmark parameters held fixed
                (repeated over the 500 bootstrap samples -> 90% interval). Validate on official-ECI models.
  3. link SW  : treat each Safetywashing capability benchmark as a NEW benchmark (its own difficulty and
                slope; SW ran its own harness, so scores aren't assumed comparable with Epoch's copies).
                Fit its parameters on SW models that have an official ECI (anchors), then project SW
                models without ECI. Validate by leave-one-anchor-out.
Outputs: data/interim/eci_extended.csv, reports/eci_extended/*.{csv,txt}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
EPOCH = ROOT / "data" / "raw" / "epoch" / "extracted"
ECI_DIR = EPOCH / "epoch_capabilities_index"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "reports" / "eci_extended"


# ---------------------------------------------------------------------------------------------
# 1. Assemble Epoch benchmark performance for all model groups
# ---------------------------------------------------------------------------------------------
def assemble_epoch() -> pd.DataFrame:
    """Rows: (model_version, model_group, benchmark, performance) for every in-ECI benchmark."""
    meta = pd.read_csv(EPOCH / "benchmark_metadata.csv")
    meta = meta[meta.in_eci & meta.source_file.notna()]
    groups = pd.read_csv(EPOCH / "model_metadata.csv").dropna(subset=["model_version"])
    vg = groups.drop_duplicates("model_version").set_index("model_version").model_group
    rows = []
    for r in meta.itertuples():
        d = pd.read_csv(EPOCH / r.source_file)
        s = pd.to_numeric(d[r.score_column], errors="coerce") * r.scale
        perf = ((s - r.random_baseline) / (r.score_ceiling - r.random_baseline)).clip(0, 1)
        rows.append(pd.DataFrame({"model_version": d["Model version"], "benchmark": r.benchmark,
                                  "performance": perf}))
    a = pd.concat(rows).dropna(subset=["performance"])
    a = a.groupby(["model_version", "benchmark"], as_index=False).performance.max()
    a["model_group"] = a.model_version.map(vg)
    return a


def check_assembly(a: pd.DataFrame) -> str:
    p = pd.read_csv(ECI_DIR / "processed_data_for_eci.csv")
    m = p.merge(a, on=["model_version", "benchmark"], how="left", suffixes=("", "_mine"))
    diff = (m.performance - m.performance_mine).abs()
    grp_ok = (m.Model == m.model_group).mean()
    return (f"assembly check: {m.performance_mine.notna().mean():.1%} of processed rows found; "
            f"{(diff < 1e-6).mean():.1%} exact; max |diff| {diff.max():.4f}; "
            f"Model == model_group for {grp_ok:.1%}")


# ---------------------------------------------------------------------------------------------
# 2. Epoch parameters and the MAP fit
# ---------------------------------------------------------------------------------------------
SW_CAP = ["logiqa", "piqa", "hellaswag", "winogrande", "superglue_copa", "medqa_4options",
          "arc_challenge", "mmlu", "minerva_math", "lambada_openai", "gsm8k", "bbh"]  # SW analysis.py
# Chance level per SW benchmark (number of options). SW's BBH is free-text exact match (min 0.03),
# so its chance level is 0, not Epoch's 0.25.
SW_CHANCE = {"logiqa": .25, "piqa": .5, "hellaswag": .25, "winogrande": .5, "superglue_copa": .5,
             "medqa_4options": .25, "arc_challenge": .25, "mmlu": .25, "minerva_math": 0.,
             "lambada_openai": 0., "gsm8k": 0., "bbh": 0.}

# Priors (our choice). Residual SD is the RMSE of Epoch's own fit (checked in main()).
SIGMA_E = 0.08
C_MU, C_SD = 120.0, 30.0
D_MU, D_SD = 120.0, 50.0
LOGS_MU, LOGS_SD = np.log(0.1), 1.0

# Inclusion rules for estimated (non-official) scores
MIN_BENCHMARKS = 4    # Epoch's own minimum per model
MIN_ECI = 80          # re-estimation error is large below this (validation: RMSE 5.4 for ECI <= 80)
MAX_CI_WIDTH = 20     # 90% bootstrap interval, ECI points


def load_epoch_params():
    d = json.load(open(ECI_DIR / "eci_bootstraps.json"))
    B, M = d["benchmark"], d["model"]
    D = pd.DataFrame(np.array(B["difficulty_samples"]), columns=B["names"])
    S = pd.DataFrame(np.array(B["slope_samples"]), columns=B["names"])
    C = pd.DataFrame(np.array(M["capability_samples"]), columns=M["names"])
    return D, S, C


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit(rows: pd.DataFrame, fixed_C: dict, fixed_D: dict, fixed_S: dict):
    """MAP fit of performance = sigmoid(S_b (C_m - D_b)).
    rows: columns model, bench, perf. Parameters not in fixed_* are free (with priors).
    Returns (C dict for free models, D dict, S dict for free benchmarks)."""
    free_m = sorted(set(rows.model) - set(fixed_C))
    free_b = sorted(set(rows.bench) - set(fixed_D))
    mi = {m: i for i, m in enumerate(free_m)}
    bi = {b: i for i, b in enumerate(free_b)}
    nm, nb = len(free_m), len(free_b)
    rm = rows.model.map(mi).to_numpy(dtype=float)
    rb = rows.bench.map(bi).to_numpy(dtype=float)
    cfix = rows.model.map(fixed_C).to_numpy(dtype=float)
    dfix = rows.bench.map(fixed_D).to_numpy(dtype=float)
    sfix = rows.bench.map(fixed_S).to_numpy(dtype=float)
    y = rows.perf.to_numpy()
    m_free, b_free = ~np.isnan(rm), ~np.isnan(rb)
    rm_i, rb_i = np.nan_to_num(rm).astype(int), np.nan_to_num(rb).astype(int)

    def unpack(x):
        return x[:nm], x[nm:nm + nb], np.exp(x[nm + nb:])

    def resid(x):
        c, dd, ss = unpack(x)
        C = np.where(m_free, c[rm_i] if nm else 0, cfix)
        D = np.where(b_free, dd[rb_i] if nb else 0, dfix)
        S = np.where(b_free, ss[rb_i] if nb else 0, sfix)
        r = (y - sig(S * (C - D))) / SIGMA_E
        pri = [(c - C_MU) / C_SD, (dd - D_MU) / D_SD, (x[nm + nb:] - LOGS_MU) / LOGS_SD]
        return np.concatenate([r, *pri])

    x0 = np.concatenate([np.full(nm, C_MU), np.full(nb, D_MU), np.full(nb, LOGS_MU)])
    sol = least_squares(resid, x0, method="trf")
    c, dd, ss = unpack(sol.x)
    return dict(zip(free_m, c)), dict(zip(free_b, dd)), dict(zip(free_b, ss))


# ---------------------------------------------------------------------------------------------
# 3. Inputs: Epoch rows per source model, SW capability rows, anchors
# ---------------------------------------------------------------------------------------------
def load_inputs(a: pd.DataFrame):
    cw = pd.read_csv(INTERIM / "model_crosswalk.csv")
    anchors = cw[(cw.match_type == "exact") & (cw.confidence == "high")]
    anchors = anchors.set_index(["source", "source_model"]).epoch_model

    vmap = pd.read_csv(ROOT / "data" / "manual" / "epoch_version_map.csv")
    ep_rows = []
    for r in vmap.itertuples():
        vers = r.epoch_versions.split(";")
        sub = a[a.model_version.isin(vers)]
        missing = set(vers) - set(sub.model_version)
        if missing == set(vers):
            sys.exit(f"epoch_version_map: no benchmark rows for any version of {r.source_model}: {missing}")
        if missing:
            print(f"  note: {r.source_model}: versions with no benchmark rows: {sorted(missing)}")
        sub = sub.groupby("benchmark", as_index=False).performance.max()
        ep_rows.append(pd.DataFrame({"model": f"{r.source}:{r.source_model}", "bench": sub.benchmark,
                                     "perf": sub.performance}))
    ep_rows = pd.concat(ep_rows)

    sw_rows = []
    sw_dir = ROOT / "data" / "raw" / "safetywashing" / "repo" / "data"
    for kind in ["chat", "base"]:
        d = pd.read_csv(sw_dir / f"benchmarks_{kind}_models.csv")
        for b in SW_CAP:
            perf = ((d[b] - SW_CHANCE[b]) / (1 - SW_CHANCE[b])).clip(0, 1)
            sw_rows.append(pd.DataFrame({"model": f"safetywashing_{kind}:" + d.model, "bench": f"SW{kind}:{b}",
                                         "perf": perf}))
    sw_rows = pd.concat(sw_rows).dropna()
    return anchors, ep_rows, sw_rows


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
def validate_epoch_projection(a, D, S, rng, n_rep=20):
    """Project official-ECI models from k of their own benchmarks (Epoch params fixed, medians)."""
    p = pd.read_csv(ECI_DIR / "processed_data_for_eci.csv")
    e = pd.read_csv(ECI_DIR / "eci_scores.csv").set_index("Model").eci
    Dm, Sm = D.median().to_dict(), S.median().to_dict()
    out = []
    for model, g in p.groupby("Model"):
        rows = pd.DataFrame({"model": model, "bench": g.benchmark, "perf": g.performance})
        c, _, _ = fit(rows, {}, Dm, Sm)
        out.append({"model": model, "k": "all", "n": len(g), "proj": c[model], "eci": e[model]})
        for k in [1, 2, 3, 4, 6]:
            if len(g) <= k:
                continue
            for _ in range(n_rep):
                sub = rows.sample(k, random_state=rng.integers(1 << 31))
                c, _, _ = fit(sub, {}, Dm, Sm)
                out.append({"model": model, "k": k, "n": len(g), "proj": c[model], "eci": e[model]})
    return pd.DataFrame(out)


def sw_fit_once(sw_rows, ep_rows, anchors_C, Dfix, Sfix, drop_anchor=None):
    """Joint fit: SW benchmark params free; anchored SW models' C fixed at official ECI;
    unanchored SW models free, using their SW rows plus any Epoch rows (Epoch params fixed)."""
    fixed = {m: c for m, c in anchors_C.items() if m != drop_anchor}
    sw_models = set(sw_rows.model)
    extra = ep_rows[ep_rows.model.isin(sw_models - set(fixed))]
    rows = pd.concat([sw_rows, extra])
    c, d, s = fit(rows, fixed, Dfix, Sfix)
    return c, d, s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    a = assemble_epoch()
    say(check_assembly(a))
    D, S, Cs = load_epoch_params()
    e = pd.read_csv(ECI_DIR / "eci_scores.csv").set_index("Model").eci

    # Epoch model form + residual SD
    p = pd.read_csv(ECI_DIR / "processed_data_for_eci.csv")
    pred = sig(S.median()[p.benchmark].values * (e[p.Model].values - D.median()[p.benchmark].values))
    say(f"Epoch fit with official ECI + median benchmark params: r = {np.corrcoef(pred, p.performance)[0, 1]:.3f}, "
        f"RMSE = {np.sqrt(np.mean((pred - p.performance) ** 2)):.3f} (SIGMA_E = {SIGMA_E})")

    # --- Validation 1: projection from Epoch benchmarks only
    v = validate_epoch_projection(a, D, S, rng)
    v["err"] = v.proj - v.eci
    v.to_csv(OUT / "validation_epoch_projection.csv", index=False)
    say("\nValidation 1: project official-ECI models from k of their own Epoch benchmarks")
    tab = v.groupby("k", sort=False).err.agg(n="size", bias="mean", rmse=lambda x: np.sqrt(np.mean(x ** 2)),
                                             mae=lambda x: np.mean(np.abs(x)))
    say(tab.round(2).to_string())

    anchors, ep_rows, sw_rows = load_inputs(a)
    # anchors as model keys
    anc = {f"{s}:{m}": em for (s, m), em in anchors.items()}
    sw_anc = {k: e[em] for k, em in anc.items() if k.startswith("safetywashing") and k in set(sw_rows.model)}
    say(f"\nSW anchors (exact+high): {len(sw_anc)} "
        f"({sum(k.startswith('safetywashing_chat') for k in sw_anc)} chat, "
        f"{sum(k.startswith('safetywashing_base') for k in sw_anc)} base)")

    # --- Validation 2: leave one SW anchor out
    Dm, Sm = D.median().to_dict(), S.median().to_dict()
    loo = []
    for m in sw_anc:
        c, _, _ = sw_fit_once(sw_rows, ep_rows, sw_anc, Dm, Sm, drop_anchor=m)
        loo.append({"model": m, "eci": sw_anc[m], "proj": c[m]})
    loo = pd.DataFrame(loo)
    loo["err"] = loo.proj - loo.eci
    loo.to_csv(OUT / "validation_sw_loo.csv", index=False)
    say("\nValidation 2: leave-one-anchor-out, SW benchmarks -> ECI")
    say(loo.round(1).to_string(index=False))
    for kind in ["chat", "base"]:
        x = loo[loo.model.str.startswith(f"safetywashing_{kind}")]
        say(f"  {kind}: n={len(x)} bias={x.err.mean():.2f} RMSE={np.sqrt((x.err ** 2).mean()):.2f} "
            f"r={np.corrcoef(x.proj, x.eci)[0, 1]:.3f}")

    # --- Point estimates + bootstrap intervals
    # SW joint fit, per bootstrap sample: Epoch params + anchor ECI from that sample, SW benchmarks resampled
    sw_bench = {k: sorted(b for b in sw_rows.bench.unique() if b.startswith(f"SW{k}:")) for k in ["chat", "base"]}
    c_point, d_sw, s_sw = sw_fit_once(sw_rows, ep_rows, sw_anc, Dm, Sm)
    pd.DataFrame({"difficulty": d_sw, "slope": s_sw}).to_csv(OUT / "sw_benchmark_params.csv")
    boot_sw = []
    n_boot = 200
    for i in range(n_boot):
        anc_i = {k: Cs.loc[i, anc[k]] for k in sw_anc}
        pick = np.concatenate([rng.choice(v_, len(v_)) for v_ in sw_bench.values()])
        rows_i = pd.concat([sw_rows[sw_rows.bench == b].assign(bench=f"{b}#{j}") for j, b in enumerate(pick)])
        c, _, _ = sw_fit_once(rows_i, ep_rows, anc_i, D.loc[i].to_dict(), S.loc[i].to_dict())
        boot_sw.append(c)
    boot_sw = pd.DataFrame(boot_sw)

    # Epoch-only projection for IRT models in the version map
    irt_models = sorted(m for m in ep_rows.model.unique() if m.startswith("irt:"))
    boot_ep = []
    for i in range(n_boot):
        rows_i = ep_rows[ep_rows.model.isin(irt_models)]
        c, _, _ = fit(rows_i, {}, D.loc[i].to_dict(), S.loc[i].to_dict())
        boot_ep.append(c)
    boot_ep = pd.DataFrame(boot_ep)
    c_ep, _, _ = fit(ep_rows[ep_rows.model.isin(irt_models)], {}, Dm, Sm)

    # --- Assemble output table: every SW and IRT model
    cw = pd.read_csv(INTERIM / "model_crosswalk.csv")
    rows = []
    rmse_k = tab.rmse.to_dict()
    for r in cw.itertuples():
        key = f"{r.source}:{r.source_model}"
        n_ep = int((ep_rows.model == key).sum())
        n_sw = int((sw_rows.model == key).sum())
        rec = {"source": r.source, "source_model": r.source_model, "match_type": r.match_type,
               "confidence": r.confidence, "eci_official": r.eci if key in anc else np.nan,
               "n_epoch_benchmarks": n_ep, "n_sw_benchmarks": n_sw}
        if key in anc:
            rec.update(method="official", eci_ext=r.eci, eci_ext_lo=r.eci_ci_low, eci_ext_hi=r.eci_ci_high)
        elif key in c_point:
            b = boot_sw[key]
            rec.update(method="sw_link", eci_ext=c_point[key], eci_ext_lo=b.quantile(.05), eci_ext_hi=b.quantile(.95))
        elif key in c_ep:
            b = boot_ep[key]
            rec.update(method="epoch_projection", eci_ext=c_ep[key], eci_ext_lo=b.quantile(.05),
                       eci_ext_hi=b.quantile(.95),
                       expected_rmse=rmse_k[max(k for k in rmse_k if k != "all" and k <= n_ep)])
        else:
            rec.update(method="none")
        rows.append(rec)
    out = pd.DataFrame(rows)
    # LOO projection for SW anchors (validation column)
    out = out.merge(loo.assign(source=loo.model.str.split(":").str[0],
                               source_model=loo.model.str.split(":", n=1).str[1])[["source", "source_model", "proj"]]
                    .rename(columns={"proj": "eci_loo_projection"}), on=["source", "source_model"], how="left")
    # --- Inclusion rules for estimated scores (user decision 2026-11-22; see reports/methodology_capability_scores.md)
    def exclusion_reason(r):
        if r.method == "official":
            return ""
        if r.method == "none":
            return "no benchmark data"
        n_bench = r.n_sw_benchmarks if r.method == "sw_link" else r.n_epoch_benchmarks
        if n_bench < MIN_BENCHMARKS:
            return f"fewer than {MIN_BENCHMARKS} benchmarks"
        if "qwen" in r.source_model.lower():
            return "Qwen (chat/base estimates disagree; no Qwen anchor)"
        if r.method == "sw_link" and r.source_model.lower().startswith("falcon") and r.source == "safetywashing_chat":
            return "Falcon chat (worst leave-one-out error, Falcon-40B-Instruct +9)"
        if r.eci_ext < MIN_ECI:
            return f"estimate below {MIN_ECI} (re-estimation unreliable for weak models)"
        if r.eci_ext_hi - r.eci_ext_lo > MAX_CI_WIDTH:
            return f"90% interval wider than {MAX_CI_WIDTH} points"
        return ""
    out["exclusion_reason"] = out.apply(exclusion_reason, axis=1)
    out["included"] = out.exclusion_reason == ""
    out["eci_final"] = out.eci_ext.where(out.included)
    out.to_csv(INTERIM / "eci_extended.csv", index=False)

    say("\nCoverage (method x source):")
    say(pd.crosstab(out.source, out.method).to_string())
    say("\nProjected models:")
    show = out[out.method.isin(["sw_link", "epoch_projection"])]
    say(show[["source", "source_model", "method", "n_epoch_benchmarks", "n_sw_benchmarks", "eci_ext",
              "eci_ext_lo", "eci_ext_hi", "expected_rmse", "included", "exclusion_reason"]]
        .round(1).to_string(index=False))
    say("\nFinal coverage (official or included estimate):")
    say(out.assign(status=np.where(out.method == "official", "official",
                                   np.where(out.included, "estimated", "no score")))
        .pipe(lambda d: pd.crosstab(d.source, d.status)).to_string())
    (OUT / "eci_extended_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
