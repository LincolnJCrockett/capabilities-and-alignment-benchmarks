# Capabilities vs. safety benchmarks

How strongly do AI safety measures track general capabilities? This project relates the safety measures
of *Item Response Theory for AI Safety* (Rivera et al. 2026), eight benchmarks and three factors, to the
Epoch Capabilities Index (ECI), and compares the results with *Safetywashing* (Ren et al. 2024).

## Where things are

| Path | Contents |
|---|---|
| `reports/capabilities_and_irt_safety.md` | **Main write-up**, companion to the post: Safetywashing comparison, IRT safety measures vs ECI, open vs closed, robustness, factors |
| `reports/archive/` | Earlier draft write-up (superseded) |
| `archive/cut_2026-11-22/` | Work cut from the analysis (ECI estimation, before/after split); not used by the pipeline |
| `reports/token_cap_findings.md` | Write-up on the 1,024-token-cap artifact in the IRT data (Figs 1–4) |
| `reports/token_cap_examples.xlsx` | **Browse the token-cap problem**: per-model rates, sample empty and truncated responses, same-prompt comparisons |
| `reports/figures/` | All figures (PNG) |
| `reports/eci_irt/` | Result tables behind the main write-up (CSV) |
| `notes/journal.md` | Research journal: questions, answers, decisions, open items |
| `scripts/` | All code (see pipeline below) |
| `data/raw/` | Downloaded source data (not modified) |
| `data/interim/` | Intermediate files built by the scripts (crosswalk, cell flags, IRT abilities). **Regenerated on every run, so don't hand-edit these.** |
| `data/manual/` | Hand-entered inputs that no script ever writes to. `crosswalk_review.csv` holds your review of the flagged crosswalk rows; `build_crosswalk.py` merges it into `model_crosswalk.csv` as the `review_output` column. |

## Getting the raw data

From the project root (Git Bash). The IRT repo is about 7 GB.

```bash
mkdir -p data/raw/epoch/extracted data/raw/safetywashing data/raw/irt
curl -sSL -o data/raw/epoch/benchmark_data.zip https://epoch.ai/data/benchmark_data.zip
unzip -oq data/raw/epoch/benchmark_data.zip -d data/raw/epoch/extracted
git clone --depth 1 https://github.com/centerforaisafety/safetywashing data/raw/safetywashing/repo
git clone --depth 1 https://github.com/jaguilar04/metasafetybench-sample data/raw/irt/repo
```

Note: Epoch's ZIP is updated continually. Our results used the version downloaded on 2026-11-19, so a
fresh download may differ slightly (new models, revised ECI values).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

(`scikit-learn` is pinned below 1.6 because `factor_analyzer` breaks with newer versions.)

## Reproducing everything

```bash
.venv/Scripts/python -I scripts/run_all.py
```

This rebuilds every file in `data/interim/`, `reports/eci_irt/` and `reports/figures/` from `data/raw/`,
in about 10 minutes. To resume from one step: `scripts/run_all.py --from irt_fit`.

## Pipeline

| Step | Script | Produces |
|---|---|---|
| 1 | `crosswalk_candidates.py` | fuzzy name-match candidates (helper) |
| 2 | `build_crosswalk.py` | `data/interim/model_crosswalk.csv`: hand-curated model-name mapping to ECI |
| 3 | `replication/safetywashing_original_copy.py` | Safetywashing's own script (paths changed, plots removed) → `data/interim/safetywashing_original_output.txt` |
| 4 | `flag_irt_cells.py` | `data/interim/irt_cell_flags.parquet`: empty / truncated flag for every response |
| 4b | `flag_grader_invalid.py` | `data/interim/irt_grader_invalid.csv`: cells whose grader returned no valid score |
| 5 | `audit_irt_responses.py` | per-model truncation audit |
| 6 | `plot_truncation_report.py` | Figs 1–4, `reports/truncation_numbers.csv` |
| 6b | `export_cap_examples.py` | `reports/token_cap_examples.xlsx` |
| 7 | `irt_fit.py` | per-benchmark 2PL abilities: `released`, `cap_masked`, `grader_masked`, and `main` (both masked; the main analysis) |
| 8 | `analysis_eci_irt.py` | validation vs the IRT paper, factor analysis, capabilities correlations → `reports/eci_irt/` |
| 9 | `plot_eci_irt.py` | capabilities-correlation, sensitivity and factor-scatter figures |
| 10 | `analysis_extensions.py` | show-the-data figure, open vs closed, developers (tables + figures) |
| 11 | `plot_safetywashing.py` | Safetywashing's original scatterplots (+ full PDF) and the two benchmarks shared with the IRT data |
| 12 | `plot_post_figures.py` | Figures 1–5 for the post (`reports/figures/post/`) |

**Main analysis:** responses left empty because the model hit the 1,024-token cap, and responses the grader
never scored, are treated as missing (`notes/journal.md`, Q21).

`irt_data.py` is a shared loader used by steps 4–8. It applies one data correction: the released
Health-ORSC matrix is scored in the opposite direction to the paper and to its own manifest, so we
flip it (details in `notes/journal.md`, Q12).
