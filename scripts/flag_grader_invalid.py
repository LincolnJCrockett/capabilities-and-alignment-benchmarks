"""Flag matrix cells whose grader did not return a valid score (journal Q16).

The IRT paper says it removes "responses for which the prescribed judge does not return a valid score",
but the released matrices score these cells anyway (mostly as refusals). Labels treated as invalid:
  error      : the grading API call failed (HarmBench: HTTP 402 from the Claude Haiku judge;
               OR-Bench-Hard: HTTP 400)
  ambiguous  : AHB judge declined to give a verdict ("I appreciate you testing my judgment, but ...")
Only the judge that the matrix uses is considered (HarmBench score files also hold a refusal-string judge).
If an item was graded more than once, the latest grading is used.

Output: data/interim/irt_grader_invalid.csv  (dataset, model, item_id, judge, label, raw)
        restricted to cells that exist in the released matrices.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from irt_data import REPO, load_matrix, manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim"
INVALID = {"error", "ambiguous"}
SKIP_JUDGES = {"harmbench": {"refusal-strings"}}  # matrix uses the Claude Haiku judge

recs, labels = [], []
for ds in manifest():
    latest = {}
    for f in sorted((REPO / "scores" / ds).glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            if r["judge"] in SKIP_JUDGES.get(ds, set()):
                continue
            key = (r["model"], r["item_id"])
            if key not in latest or r.get("timestamp", "") >= latest[key].get("timestamp", ""):
                latest[key] = r
    mat = load_matrix(ds)
    for (m, i), r in latest.items():
        labels.append((ds, r["label"]))
        if r["label"] in INVALID and m in mat.index and i in mat.columns and pd.notna(mat.at[m, i]):
            recs.append((ds, m, i, r["judge"], r["label"], str((r.get("extras") or {}).get("raw", ""))[:80]))

df = pd.DataFrame(recs, columns=["dataset", "model", "item_id", "judge", "label", "raw"])
df.to_csv(OUT / "irt_grader_invalid.csv", index=False)
print("Grader labels per dataset (latest grading per cell):")
print(pd.DataFrame(labels, columns=["dataset", "label"]).value_counts().sort_index().to_string())
print(f"\nInvalid cells inside the matrices: {len(df)}")
print(df.groupby(["dataset", "label"]).size().to_string())
print("\nBy model (top 15):")
print(df.groupby(["dataset", "model"]).size().sort_values(ascending=False).head(15).to_string())
