"""Shared loader for the IRT-for-AI-Safety matrices, with orientation fixes.

ORIENTATION FIX (2026-11-19, see notes/journal.md):
  The released matrix_health_orsc.parquet is NOT flipped even though manifest.json says flip=True:
  its values equal the raw refusal-string judge scores (complied=1, refused=0) for 100% of cells,
  whereas every other flip=True benchmark is inverted relative to its raw scores. The paper reports
  AdvBench-Health-ORSC rho = +0.58; the unflipped matrix gives -0.57. We therefore apply the flip
  so that 1 = refused, matching the manifest, the paper, and the other harm benchmarks.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data" / "raw" / "irt" / "repo"
FLIP_FIX = {"health_orsc"}


def manifest() -> dict:
    return json.loads((REPO / "matrices" / "manifest.json").read_text())["datasets"]


def load_matrix(ds: str) -> pd.DataFrame:
    m = pd.read_parquet(REPO / "matrices" / manifest()[ds]["path"])
    return 1 - m if ds in FLIP_FIX else m


def canonical_items() -> pd.DataFrame:
    return pd.read_json(REPO / "full_battery" / "items.jsonl", lines=True)
