"""Audit IRT raw responses for truncation / empty output, and how those cells are coded in the matrices.

Output: data/interim/irt_response_audit.csv (one row per model x dataset) and a printed summary.
"""
import json
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from irt_data import load_matrix  # noqa: E402  (applies Health-ORSC orientation fix)

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data" / "raw" / "irt" / "repo"
OUT = ROOT / "data" / "interim"

manifest = json.loads((REPO / "matrices" / "manifest.json").read_text())["datasets"]

rows, cells = [], []
for ds, meta in manifest.items():
    mat = load_matrix(ds)
    for f in sorted((REPO / "responses" / ds).glob("*.jsonl")):
        n = trunc = empty = empty_cap = err = 0
        model = None
        last = {}  # keep only the final attempt per item (files include retries)
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            prev = last.get(r["item_id"])
            if prev is None or r.get("attempt", 1) >= prev.get("attempt", 1):
                last[r["item_id"]] = r
        for r in last.values():
            model = r["model"]
            n += 1
            # mutually exclusive: empty (no text, incl. hit the cap while reasoning) vs truncated-with-text
            is_empty = r.get("status") == "ok" and not (r.get("response") or "").strip()
            is_trunc = r.get("finish_reason") == "length" and not is_empty
            is_err = r.get("status") != "ok"
            trunc += is_trunc
            empty += is_empty
            empty_cap += is_empty and r.get("finish_reason") == "length"
            err += is_err
            if (is_trunc or is_empty) and model in mat.index and r["item_id"] in mat.columns:
                cells.append({"dataset": ds, "kind": "truncated_with_text" if is_trunc else "empty",
                              "value": mat.at[model, r["item_id"]]})
        rows.append({"dataset": ds, "model": model, "n": n, "truncated": trunc,
                     "empty": empty, "empty_at_cap": empty_cap, "error": err, "in_matrix": model in mat.index})

df = pd.DataFrame(rows)
df.to_csv(OUT / "irt_response_audit.csv", index=False)

tot = df.groupby("dataset")[["n", "truncated", "empty", "error"]].sum()
tot["pct_trunc"] = (100 * tot.truncated / tot.n).round(2)
tot["pct_empty"] = (100 * tot["empty"] / tot.n).round(2)
print(tot, "\n")

df["pct_bad"] = 100 * (df.truncated + df["empty"]) / df.n
worst = df.groupby("model")[["n", "truncated", "empty"]].sum()
worst["pct_bad"] = (100 * (worst.truncated + worst["empty"]) / worst.n).round(1)
print("models with >5% truncated or empty responses:")
print(worst[worst.pct_bad > 5].sort_values("pct_bad", ascending=False).to_string(), "\n")

c = pd.DataFrame(cells)
if len(c):
    c["value"] = c["value"].map(lambda v: "NaN" if pd.isna(v) else v)
    print("matrix coding of truncated/empty cells (1 = safe/desired):")
    print(c.groupby(["kind", "dataset", "value"]).size().unstack(fill_value=0))
