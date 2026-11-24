"""Generate fuzzy-match candidates between safety-source model names and Epoch ECI models.

Output: data/interim/crosswalk_candidates.csv (top-3 Epoch candidates per source model).
This is a helper for building the curated crosswalk; it is not the final mapping.
"""
import re
import glob
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim"
OUT.mkdir(parents=True, exist_ok=True)

DROP_TOKENS = {"instruct", "it", "chat", "hf", "v1", "001", "latest"}


def norm(s: str) -> str:
    s = s.lower()
    s = s.split("/")[-1]                       # drop OpenRouter org prefix
    s = re.sub(r"_(xhigh|high|medium|low|max|minimal|none|unknown|\d+k)$", "", s)
    s = re.sub(r"[\s_\-:.()]+", " ", s)
    toks = [t for t in s.split() if t not in DROP_TOKENS]
    return " ".join(toks)


def epoch_aliases() -> pd.DataFrame:
    eci = pd.read_csv(RAW / "epoch/extracted/epoch_capabilities_index/eci_scores.csv")
    proc = pd.read_csv(RAW / "epoch/extracted/epoch_capabilities_index/processed_data_for_eci.csv")
    rows = [(m, m) for m in eci["Model"]]
    rows += [(m, d) for m, d in zip(eci["Model"], eci["Display name"])]
    rows += list(proc[["Model", "model_version"]].drop_duplicates().itertuples(index=False, name=None))
    df = pd.DataFrame(rows, columns=["epoch_model", "alias"]).drop_duplicates()
    df["alias_norm"] = df["alias"].astype(str).map(norm)
    return df


def irt_models() -> list[str]:
    models = set()
    for f in glob.glob(str(RAW / "irt/repo/matrices/*.parquet")):
        models |= set(pd.read_parquet(f).index)
    return sorted(models)


def sw_models() -> list[tuple[str, str]]:
    out = []
    for kind in ("chat", "base"):
        df = pd.read_csv(RAW / f"safetywashing/repo/data/benchmarks_{kind}_models.csv")
        out += [(f"safetywashing_{kind}", m) for m in df["model"]]
    return out


def main():
    al = epoch_aliases()
    choices = al["alias_norm"].tolist()
    sources = [("irt", m) for m in irt_models()] + sw_models()
    rows = []
    for src, m in sources:
        hits = process.extract(norm(m), choices, scorer=fuzz.token_sort_ratio, limit=15)
        seen, cands = set(), []
        for _, score, idx in hits:
            em = al.iloc[idx]["epoch_model"]
            if em not in seen:
                seen.add(em)
                cands.append((em, al.iloc[idx]["alias"], round(score, 1)))
            if len(cands) == 3:
                break
        row = {"source": src, "source_model": m}
        for i, (em, alias, sc) in enumerate(cands, 1):
            row[f"cand{i}"], row[f"cand{i}_alias"], row[f"cand{i}_score"] = em, alias, sc
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "crosswalk_candidates.csv", index=False)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
