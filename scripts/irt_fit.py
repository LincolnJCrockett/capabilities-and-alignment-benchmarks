"""Fit per-benchmark 2PL IRT models to the IRT-for-AI-Safety matrices, following Rivera et al. (2026).

Paper spec (Materials & Methods): separate unidimensional 2PL per benchmark, full model cohort,
marginal maximum likelihood, log-normal prior on discrimination, normal prior on the intercept,
ability = posterior mode under N(0,1). Hyperparameters and software are not reported; ours are below
and recorded in notes/journal.md.

Variants:
  released : matrices as released (paper default)
  cap_masked    : cells that were empty because the model hit the 1,024-token cap set to missing
  grader_masked : cells whose grader returned no valid score set to missing (data/interim/irt_grader_invalid.csv)
  main          : both of the above (MAIN ANALYSIS, user decision 2026-11-22; journal Q21)
In the masked variants a model's ability on a benchmark is set to missing when fewer than
MIN_COVERAGE of its items remain (an ability from a handful of self-selected items is unreliable).

Items: the authors' canonical battery (full_battery/items.jsonl, 5,062 items).
Outputs: data/interim/irt_abilities_{variant}.csv (model x benchmark theta),
         data/interim/irt_items_{variant}.csv (item a, d)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from irt_data import load_matrix  # noqa: E402  (applies Health-ORSC orientation fix)
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data" / "raw" / "irt" / "repo"
INTERIM = ROOT / "data" / "interim"

# Priors (our choice; paper gives families only)
LOGA_MU, LOGA_SD = 0.0, 0.5   # log a ~ N(0, 0.5^2)  -> a log-normal, median 1
D_SD = 2.0                    # d ~ N(0, 2^2) on the slope-intercept logit a*theta + d
N_QUAD = 41

nodes, weights = hermegauss(N_QUAD)
weights = weights / weights.sum()  # N(0,1) quadrature


def fit_2pl(X: np.ndarray):
    """X: models x items with 1/0/NaN. Returns a, d (items)."""
    M = (~np.isnan(X)).astype(float)
    Xf = np.nan_to_num(X)
    MX = M * Xf
    n_items = X.shape[1]
    logw = np.log(weights)

    def negpost(params):
        alpha, d = params[:n_items], params[n_items:]
        a = np.exp(alpha)
        Z = np.outer(nodes, a) + d                      # Q x I
        logp, log1mp = -np.logaddexp(0, -Z), -np.logaddexp(0, Z)
        ll_jq = MX @ logp.T + (M - MX) @ log1mp.T        # J x Q
        A = ll_jq + logw
        mx = A.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(A - mx).sum(axis=1))
        R = np.exp(A - lse[:, None])                     # posterior over nodes, J x Q
        P = np.exp(logp)
        G = R.T @ MX - P * (R.T @ M)                     # dLL/dZ, Q x I
        g_a = (nodes[:, None] * G).sum(0)
        g_d = G.sum(0)
        lp = -0.5 * ((alpha - LOGA_MU) / LOGA_SD) ** 2 - 0.5 * (d / D_SD) ** 2
        f = -(lse.sum() + lp.sum())
        g_alpha = -(g_a * a - (alpha - LOGA_MU) / LOGA_SD**2)
        g_dd = -(g_d - d / D_SD**2)
        return f, np.concatenate([g_alpha, g_dd])

    # init d from item means
    pbar = np.clip(np.nanmean(X, axis=0), 0.02, 0.98)
    x0 = np.concatenate([np.zeros(n_items), np.log(pbar / (1 - pbar))])
    res = minimize(negpost, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    if not res.success:
        print("   warning:", res.message)
    return np.exp(res.x[:n_items]), res.x[n_items:]


def map_theta(X: np.ndarray, a: np.ndarray, d: np.ndarray, iters: int = 200) -> np.ndarray:
    """Posterior-mode ability under N(0,1); damped Newton (steps clipped to +-0.5), vectorized.
    The objective is strictly concave, so the damped iteration converges to the unique mode."""
    M = ~np.isnan(X)
    Xf = np.nan_to_num(X)
    th = np.zeros(X.shape[0])
    for _ in range(iters):
        Z = np.clip(th[:, None] * a + d, -35, 35)
        P = 1 / (1 + np.exp(-Z))
        g = (M * a * (Xf - P)).sum(1) - th
        h = -(M * a**2 * P * (1 - P)).sum(1) - 1
        th = th - np.clip(g / h, -0.5, 0.5)
    assert np.abs(g).max() < 1e-4, f"theta not converged: max |grad| = {np.abs(g).max():.2e}"
    return th


MIN_COVERAGE = 0.5
VARIANTS = {"released": (), "cap_masked": ("cap",), "grader_masked": ("grader",), "main": ("cap", "grader")}


def main():
    manifest = json.loads((REPO / "matrices" / "manifest.json").read_text())["datasets"]
    items = pd.read_json(REPO / "full_battery" / "items.jsonl", lines=True)
    flags = pd.read_parquet(INTERIM / "irt_cell_flags.parquet")
    masks = {"cap": flags.loc[flags.kind == "empty_at_cap", ["dataset", "model", "item_id"]],
             "grader": pd.read_csv(INTERIM / "irt_grader_invalid.csv")[["dataset", "model", "item_id"]]}

    for variant, mask_kinds in VARIANTS.items():
        thetas, item_rows = {}, []
        for ds, meta in manifest.items():
            mat = load_matrix(ds)
            mat = mat[items.loc[items.dataset == ds, "question_id"]].copy()
            n_before = mat.notna().sum(axis=1)
            for k in mask_kinds:
                cc = masks[k][(masks[k].dataset == ds) & masks[k].model.isin(mat.index) & masks[k].item_id.isin(mat.columns)]
                for m, i in zip(cc.model, cc.item_id):
                    mat.at[m, i] = np.nan
            keep = mat.notna().sum(axis=1) >= MIN_COVERAGE * n_before
            dropped = sorted(mat.index[~keep])
            mat = mat.loc[keep]
            X = mat.to_numpy(dtype=float)
            a, d = fit_2pl(X)
            th = map_theta(X, a, d)
            thetas[ds] = pd.Series(th, index=mat.index)
            item_rows.append(pd.DataFrame({"dataset": ds, "item_id": mat.columns, "a": a, "d": d}))
            print(f"{variant:13s} {ds:13s} models={X.shape[0]:3d} items={X.shape[1]:4d} "
                  f"a median={np.median(a):.2f} theta sd={th.std():.2f}"
                  + (f"  dropped (<{MIN_COVERAGE:.0%} items left): {dropped}" if dropped else ""))
        pd.DataFrame(thetas).rename_axis("model").to_csv(INTERIM / f"irt_abilities_{variant}.csv")
        pd.concat(item_rows).to_csv(INTERIM / f"irt_items_{variant}.csv", index=False)


if __name__ == "__main__":
    main()
