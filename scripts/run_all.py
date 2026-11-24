"""Rebuild every derived file and figure from the raw data, in dependency order.

Usage (from the project root):
    .venv\\Scripts\\python -I scripts\\run_all.py            # everything
    .venv\\Scripts\\python -I scripts\\run_all.py --from irt_fit   # resume from a step

Raw data must already be in data/raw/ (see README.md, "Getting the raw data").
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    # (name, script, stdout file or None, what it produces)
    ("crosswalk_candidates", "scripts/crosswalk_candidates.py", None,
     "data/interim/crosswalk_candidates.csv"),
    ("build_crosswalk", "scripts/build_crosswalk.py", None,
     "data/interim/model_crosswalk.csv"),
    ("safetywashing_original", "scripts/replication/safetywashing_original_copy.py",
     "data/interim/safetywashing_original_output.txt",
     "Safetywashing replication output"),
    ("flag_irt_cells", "scripts/flag_irt_cells.py", None,
     "data/interim/irt_cell_flags.parquet (reads ~4 GB of responses)"),
    ("flag_grader_invalid", "scripts/flag_grader_invalid.py", "data/interim/irt_grader_invalid_log.txt",
     "data/interim/irt_grader_invalid.csv (cells with no valid grader score)"),
    ("audit_irt_responses", "scripts/audit_irt_responses.py", "data/interim/irt_response_audit.txt",
     "data/interim/irt_response_audit.csv"),
    ("plot_truncation_report", "scripts/plot_truncation_report.py", None,
     "reports/figures/fig1-4, reports/truncation_numbers.csv"),
    ("export_cap_examples", "scripts/export_cap_examples.py", None,
     "reports/token_cap_examples.xlsx (browsable examples)"),
    ("irt_fit", "scripts/irt_fit.py", None,
     "data/interim/irt_abilities_{released,cap_masked,grader_masked,main}.csv"),
    ("analysis_eci_irt", "scripts/analysis_eci_irt.py", "reports/eci_irt/analysis_log.txt",
     "reports/eci_irt/*.csv"),
    ("plot_eci_irt", "scripts/plot_eci_irt.py", None,
     "reports/figures/fig5-6 (benchmarks), factor_correlations + factor_weights (factor section)"),
    ("analysis_extensions", "scripts/analysis_extensions.py", "reports/eci_irt/extensions_log.txt",
     "show-the-data figure (8 panels), open/closed, developer analyses"),
    ("plot_safetywashing", "scripts/plot_safetywashing.py", "reports/eci_irt/sw_plots_log.txt",
     "Safetywashing original scatterplots + the 2 shared benchmarks"),
    ("plot_post_figures", "scripts/plot_post_figures.py", None,
     "reports/figures/post/fig1-5 (the post's figures; report §3-§5)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=[s[0] for s in STEPS])
    args = ap.parse_args()
    names = [s[0] for s in STEPS]
    steps = STEPS[names.index(args.start):] if args.start else STEPS
    (ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)

    for name, script, log, produces in steps:
        t0 = time.time()
        print(f"[{name}] -> {produces}", flush=True)
        cmd = [sys.executable, "-I", "-X", "utf8", "-W", "ignore", str(ROOT / script)]
        out = open(ROOT / log, "w", encoding="utf-8") if log else subprocess.DEVNULL
        env = {**__import__("os").environ, "MPLBACKEND": "Agg"}
        r = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.PIPE, text=True, env=env)
        if log:
            out.close()
        if r.returncode != 0:
            print(r.stderr)
            sys.exit(f"[{name}] FAILED (exit {r.returncode})")
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
    print("All steps completed.")


if __name__ == "__main__":
    main()
