"""Write a cell-level flag table for IRT responses: which (dataset, model, item) were empty or truncated.

Output: data/interim/irt_cell_flags.parquet with columns
  dataset, model, item_id, kind in {empty_at_cap, empty_other, truncated_with_text, ok}, matrix_value
Only the final attempt per item is kept (response files include retries).
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
recs = []
for ds, meta in manifest.items():
    mat = load_matrix(ds)
    for f in sorted((REPO / "responses" / ds).glob("*.jsonl")):
        last = {}
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            prev = last.get(r["item_id"])
            if prev is None or r.get("attempt", 1) >= prev.get("attempt", 1):
                last[r["item_id"]] = r
        for r in last.values():
            if r.get("status") != "ok":
                continue
            empty = not (r.get("response") or "").strip()
            cap = r.get("finish_reason") == "length"
            kind = ("empty_at_cap" if empty and cap else "empty_other" if empty
                    else "truncated_with_text" if cap else "ok")
            m, i = r["model"], r["item_id"]
            val = mat.at[m, i] if (m in mat.index and i in mat.columns) else float("nan")
            recs.append((ds, m, i, kind, val))

df = pd.DataFrame(recs, columns=["dataset", "model", "item_id", "kind", "matrix_value"])
df["kind"] = df["kind"].astype("category")
df.to_parquet(OUT / "irt_cell_flags.parquet", index=False)
print(df.groupby(["dataset", "kind"], observed=True).size().unstack(fill_value=0))
