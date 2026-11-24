"""Export a browsable Excel workbook showing how the 1,024-token cap affects IRT responses.

Output: reports/token_cap_examples.xlsx with sheets
  README            what each sheet and column means
  by_model          per model: % empty at cap / % truncated, overall and per benchmark, with ECI
  empty_at_cap      sample of responses where the model used all 1,024 tokens and returned no text
  truncated         sample of responses cut off mid-answer (partial text was judged)
  same_item         the same prompts answered by a heavily affected model vs models that answered
Samples are drawn from the primary sample (exact, high-confidence ECI matches), fixed seed.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from irt_data import load_matrix, manifest  # noqa: E402  (applies Health-ORSC orientation fix)

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data" / "raw" / "irt" / "repo"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "reports" / "token_cap_examples.xlsx"
SEED, PER_BENCH, MAXCHARS = 0, 25, 3000

MEANING = {  # what a score of 1 means in the (orientation-fixed) matrix
    "harmbench": "1 = refused", "sorrybench": "1 = refused", "ahb": "1 = refused",
    "advbench": "1 = refused", "health_orsc": "1 = refused", "donotanswer": "1 = safe response",
    "orbench_hard": "1 = answered the benign prompt", "truthfulqa": "1 = truthful",
}
KIND_LABEL = {"empty_at_cap": "Empty: hit 1,024-token cap", "empty_other": "Empty: other",
              "truncated_with_text": "Truncated: partial text", "ok": "Complete"}


def read_records(ds, model):
    """Final attempt per item for one model/benchmark, as a DataFrame."""
    f = REPO / "responses" / ds / (model.replace("/", "_") + ".jsonl")
    last = {}
    for line in f.open(encoding="utf-8"):
        r = json.loads(line)
        prev = last.get(r["item_id"])
        if prev is None or r.get("attempt", 1) >= prev.get("attempt", 1):
            last[r["item_id"]] = r
    rows = []
    for r in last.values():
        meta = r.get("dataset_metadata") or {}
        rows.append({
            "item_id": r["item_id"], "prompt": r.get("prompt"),
            "response": (r.get("response") or "")[:MAXCHARS],
            "response_chars": len(r.get("response") or ""),
            "finish_reason": r.get("finish_reason"),
            "output_tokens": (r.get("usage") or {}).get("output_tokens"),
            "truthfulqa_best_answer": meta.get("best_answer") if ds == "truthfulqa" else None,
        })
    return pd.DataFrame(rows)


def main():
    flags = pd.read_parquet(INTERIM / "irt_cell_flags.parquet")
    flags = flags[flags.matrix_value.notna()].copy()
    cw = pd.read_csv(INTERIM / "model_crosswalk.csv")
    prim = (cw[(cw.source == "irt") & (cw.match_type == "exact") & (cw.confidence == "high")]
            .drop_duplicates("epoch_model").set_index("source_model"))
    eci = cw[cw.source == "irt"].set_index("source_model")[["epoch_model", "eci"]]

    # ---- by_model ----
    flags["empty_cap"] = flags.kind == "empty_at_cap"
    flags["trunc"] = flags.kind == "truncated_with_text"
    overall = flags.groupby("model")[["empty_cap", "trunc"]].mean().mul(100).round(1)
    overall.columns = ["% empty at cap (all benchmarks)", "% truncated (all benchmarks)"]
    per = flags.pivot_table(index="model", columns="dataset", values="empty_cap", aggfunc="mean").mul(100).round(1)
    per.columns = [f"% empty at cap: {c}" for c in per.columns]
    by_model = eci.join(overall, how="right").join(per)
    by_model.insert(2, "in primary sample", by_model.index.isin(prim.index))
    by_model = by_model.sort_values("% empty at cap (all benchmarks)", ascending=False).reset_index()

    # ---- samples ----
    cache = {}

    def records(ds, model):
        if (ds, model) not in cache:
            cache[(ds, model)] = read_records(ds, model).set_index("item_id")
        return cache[(ds, model)]

    def sample(kind):
        sub = flags[(flags.kind == kind) & flags.model.isin(prim.index)]
        parts = []
        for ds, g in sub.groupby("dataset"):
            parts.append(g.sample(min(PER_BENCH, len(g)), random_state=SEED))
        rows = []
        for r in pd.concat(parts).itertuples():
            rec = records(r.dataset, r.model).loc[r.item_id]
            rows.append({
                "benchmark": r.dataset, "model": r.model, "ECI": prim.at[r.model, "eci"],
                "item_id": r.item_id, "prompt": rec.prompt, "response (as returned)": rec.response,
                "response length (chars)": rec.response_chars, "finish_reason": rec.finish_reason,
                "output tokens used": rec.output_tokens, "score in matrix": r.matrix_value,
                "what 1 means": MEANING[r.dataset], "TruthfulQA best answer": rec.truthfulqa_best_answer,
            })
        return pd.DataFrame(rows)

    empty_cap = sample("empty_at_cap")
    truncated = sample("truncated_with_text")

    # ---- same_item: heavy model's empty answer vs how two unaffected models answered ----
    heavy_models = ["openai/gpt-5-nano", "z-ai/glm-5", "qwen/qwen3.5-35b-a3b"]
    compare_models = ["anthropic/claude-sonnet-4.5", "meta-llama/llama-3.3-70b-instruct"]
    same = []
    for ds in ["harmbench", "orbench_hard", "truthfulqa"]:
        mat = load_matrix(ds)
        for hm in heavy_models:
            cells = flags[(flags.dataset == ds) & (flags.model == hm) & (flags.kind == "empty_at_cap")]
            for item in cells.sample(min(3, len(cells)), random_state=SEED).item_id:
                for m in [hm] + compare_models:
                    if item not in records(ds, m).index:
                        continue
                    rec = records(ds, m).loc[item]
                    k = flags[(flags.dataset == ds) & (flags.model == m) & (flags.item_id == item)].kind
                    same.append({
                        "benchmark": ds, "item_id": item, "prompt": rec.prompt, "model": m,
                        "status": KIND_LABEL.get(k.iloc[0] if len(k) else "ok"),
                        "response (as returned)": rec.response[:1500],
                        "output tokens used": rec.output_tokens,
                        "score in matrix": mat.at[m, item] if m in mat.index else None,
                        "what 1 means": MEANING[ds],
                    })
    same = pd.DataFrame(same)

    readme = pd.DataFrame({"Sheet / column": [
        "by_model", "empty_at_cap", "truncated", "same_item", "",
        "score in matrix", "output tokens used", "finish_reason", "response (as returned)", "Note"],
        "Meaning": [
        "Every model: share of its scored responses that were empty at the cap / truncated, overall and per benchmark. Sorted worst first.",
        f"Random sample ({PER_BENCH} per benchmark) of responses from primary-sample models that used all 1,024 tokens and returned no visible text. Look at 'score in matrix': the empty answer is scored as a refusal.",
        f"Random sample ({PER_BENCH} per benchmark) of responses cut off mid-answer; the judge scored the partial text.",
        "The same prompt: a heavily affected model (empty at cap) next to two models that answered normally.",
        "",
        "The score that enters the IRT analysis (Health-ORSC orientation fixed). See 'what 1 means' for each benchmark.",
        "1024 = the cap was reached. For empty-at-cap rows, all 1,024 tokens went to hidden reasoning; that reasoning text was not saved.",
        "'length' = stopped by the token cap; 'stop' = model finished on its own.",
        f"Text returned to the user (first {MAXCHARS} characters). Blank = nothing returned.",
        "Full raw data: data/raw/irt/repo/responses/<benchmark>/<model>.jsonl. Built by scripts/export_cap_examples.py.",
    ]})

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        for name, df in [("README", readme), ("by_model", by_model), ("empty_at_cap", empty_cap),
                         ("truncated", truncated), ("same_item", same)]:
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            for col in ws.columns:  # sensible widths; wrap long text
                width = min(60, max(10, max(len(str(c.value or "")) for c in col[:50])))
                ws.column_dimensions[col[0].column_letter].width = width
    print(f"wrote {OUT} | empty_at_cap {len(empty_cap)} rows, truncated {len(truncated)}, same_item {len(same)}")


if __name__ == "__main__":
    main()
