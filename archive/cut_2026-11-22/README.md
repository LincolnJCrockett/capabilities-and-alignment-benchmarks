# Work cut from the analysis on 2026-11-22 (user decision)

Kept for reference only; nothing in the pipeline reads these files.
- `scripts/eci_extend.py`, `data_manual/epoch_version_map.csv`, `reports/eci_extended/`,
  `reports/methodology_capability_scores.md`, `data_interim/eci_extended.csv`: ECI estimation for models
  without an official score (journal Q19–Q20). Cut: too many assumptions.
- `reports/figures/pre_post_correlations.png`, `safety_vs_time.png`, `reports/eci_irt/pre_post.csv`,
  `time_trends.csv`: before/after-Safetywashing split and time trends. Cut: "has it changed" is no longer
  a core question.
- `reports/eci_irt/eci_vs_safetywashing_score.csv`: ECI vs Safetywashing capabilities score (needed the
  Safetywashing-to-ECI crosswalk).
- `irt_abilities_masked.csv`, `factor_scores_all_models.csv`, `factor_loadings_masked_refit.csv`:
  outputs of the earlier pipeline, replaced by the `main` / `cap_masked` / `grader_masked` variants.
The earlier full draft write-up is at `reports/archive/replication_and_eci_irt_draft_2026-11-22.md`.
