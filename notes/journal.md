# Research journal

Running log of questions, answers, and decisions. Newest entries at the bottom of each section.
Default rule (set 2026-11-19): **for analysis choices, follow the IRT paper
(Rivera et al. 2026, arXiv:2608.05086) unless there's a specific reason not to; record any deviation here.**

---

## 2026-11-19: Data sources and model crosswalk

- Sources: Epoch Benchmarking Hub ZIP (ECI = capabilities composite), Safetywashing repo
  (centerforaisafety/safetywashing), IRT data release (github.com/jaguilar04/metasafetybench-sample).
- Crosswalk: `data/interim/model_crosswalk.csv`, built by `scripts/build_crosswalk.py` (hand-curated dict).
- **Decision (user):** primary sample = `match_type == exact` AND `confidence == high`.
  IRT: 68 rows / 67 distinct ECI models. Safetywashing: 15 base, 8 chat.
  The 11 IRT models dropped (snapshot/family matches: GPT-4o, DeepSeek-V3, GPT-4 Turbo, ...) come back
  later as a robustness check.
- Caveat: some "exact" ECI entries pool base + instruct results on Epoch's side (Llama 3, Mixtral,
  Gemma 2, Qwen2.5-72B incl. VL-72B).
- Plan: (1) replicate Safetywashing with its own data; (2) do the same analysis with ECI × IRT benchmarks;
  (3) robustness check: add the dropped models back.

---

## 2026-11-19: Questions about the IRT paper and how it answers them

Source: full read of arXiv:2608.05086v1 (HTML). Quotes trimmed.

### Q1. Which direction is each benchmark scored, especially OR-Bench-Hard (over-refusal)?
**Answered.** Every benchmark is scored so that 1 = the behavior the benchmark wants. For the harm benchmarks
that means refusing; for OR-Bench-Hard it means *answering* the benign request; for TruthfulQA it means
answering truthfully. So per-benchmark abilities are **not** flipped: a high OR-Bench score means
*less* over-refusal.
- The factors come out as follows. **Refusal strictness:** HarmBench, SORRY-Bench and AHB load positively,
  OR-Bench-Hard negatively. **Truthfulness:** TruthfulQA and Do-Not-Answer. **Contextual harm:** AdvBench
  and Health-ORSC (ρ = 0.58).
- OR-Bench-Hard is flipped only when building a composite (App. D: "Refusal axis (ORB flipped)").
  The paper explicitly argues against an unweighted "safety" average.
- **Our default:** report per-benchmark results with the paper's orientation, plus the three factors.
  When we need a refusal composite, flip ORB as the paper does. Don't build an all-8 average.
- Note: Table 1 labels Do-Not-Answer and AdvBench as "harm refusal", yet they load on truthfulness and
  contextual harm respectively. These labels describe the benchmarks, not the factor results.

### Q2. Does the 1,024-token cap truncate reasoning models' answers? How are empty or truncated responses coded?
**Not addressed by the paper.** The only stated rule is that responses without a valid judge score are
removed. The paper never discusses reasoning models, truncation, empty responses or provider moderation.

**Our audit** (`scripts/audit_irt_responses.py` → `data/interim/irt_response_audit.{csv,txt}`, final
attempt per item only):
- Across all models, about 4–10% of responses per benchmark are **empty**, and another 6–20% are
  **truncated with partial text**.
- Most empty responses are **models that hit the 1,024-token cap while still reasoning and returned
  no visible text** (6.4% of responses in our 67-model primary sample). Other empty responses are rare
  (0.4%) and look like genuine API-level refusals (e.g. Claude Opus 4 / Sonnet 4.5).
- **Empty responses are scored as refusals by the judges:** mostly 1 on the harm benchmarks
  (HarmBench 3053/3053, AHB 16671/16679) and mostly 0 on OR-Bench-Hard (17786/17908) and TruthfulQA
  (10122/10512).
- ~~In the primary sample, 13 of 67 models have more than 10%...~~ CORRECTED 2026-11-19: the first audit
  counted response records outside the released matrices (AHB's response files contain extra items).
  Restricted to matrix cells (`scripts/flag_irt_cells.py`), the share empty at the cap is 6.6% across all
  models and 8.8% in the primary sample, and **14 of 67** primary models are above 10%: GPT-5 nano 94%,
  Qwen3.5-9B 67%, GLM-5 51%, Qwen3.5-35B 49%, GLM-4.7 36%, GLM-5.1 35%, ...
- **This is related to capability.** Spearman(ECI, % empty from hitting the cap) = 0.45 (unchanged).
- **Write-up:** `reports/token_cap_findings.md`. Key result: on raw proportion scores, the refusal-benchmark
  correlations with ECI are robust, but **TruthfulQA flips from ρ = −0.11 to +0.34** when empty-at-cap
  cells are set to missing; OR-Bench-Hard and Do-Not-Answer move toward zero / positive.
- 2026-11-19: user is reading through the truncated / empty responses themselves.
- **Why it matters for us:** this artifact pushes more-capable reasoning models toward the
  "refusal-strict" end (higher harm-refusal, lower OR-Bench, lower TruthfulQA). That alone could
  produce or distort a capability–safety correlation.
- **Decision:** primary analysis follows the paper (use the matrices as released). Sensitivity check:
  recode cells where the model hit the cap with an empty response as missing, and/or drop models with
  more than 10% such cells.
  STATUS: proposed, to confirm with user.

### Q3. How are missing benchmarks handled? What filtering is applied?
**Answered.** No imputation.
- Items: drop items nobody answered and items with zero variance. That takes 5,255 items down to 5,067.
- Responses: drop any response without a valid judge score.
- Models: cross-benchmark analyses use the 169 models with responses on all 8 benchmarks.
  Factor analysis and MIRT use the 134 models that also have a valid judge score on every retained item.
  Per-benchmark 2PL fits use the "full model cohort".
- Unexplained discrepancies in the paper: the MIRT fit uses 5,055 items (vs 5,067), and the sandbagging
  analysis uses a 171-model cohort.
- **Our default:** same rules. For per-benchmark correlations with ECI, use all primary models that have
  that benchmark. For factor-level analyses, restrict to the 134-model cohort ∩ our primary sample.

### Q4. Which judge is used for each benchmark?
**Partly answered.** Each benchmark uses the judge its source paper prescribes (Table 1). The paper gives no
judge agreement statistics. HarmBench uses the LLM judge (Claude Haiku 4.5); the repo also has a
refusal-string judge, but the paper and the matrices use the LLM judge.
- **Our default:** use the released matrices (`matrices/manifest.json` records which judge each one uses).

### Q5. How is the IRT model specified?
**Answered (mostly).**
- One unidimensional 2PL per benchmark, fit by marginal maximum likelihood. Priors: log-normal on
  discrimination, normal on the intercept (hyperparameters not given). Ability = posterior mode under an
  N(0,1) prior.
- Second stage: minimum-residual factor analysis on the M×8 ability matrix, oblimin rotation. Three
  factors explain 77% of variance (CFI 0.97). Parallel analysis supports 3 (min-res) or 2 (PCA).
  The paper's robust claim is "at least two".
- Not given: software, prior hyperparameters, how per-model factor scores are computed.
- **Our default:** replicate this pipeline as closely as we can. Factor-analysis software will be our
  choice (e.g. the `factor_analyzer` Python package with minres + oblimin). Check that we recover their
  benchmark correlations and factor structure before relating anything to ECI.

### Q6. Does the paper relate safety to capabilities? Does it cite Safetywashing?
**No to both.** There's no analysis against capability, compute, size or release date, and no citation of
Ren et al. 2024. → Relating these factors to ECI is a new contribution, not a replication.

### Q7. What limitations does the paper acknowledge?
- Sample size (169/134); whether there are 2 or 3 factors is unresolved; factor names are summaries,
  not validated constructs; sandbagging was tested only with prompted model organisms.
- Provider routing, quantization and silent serving changes can alter behavior under the same model ID.
  The paper doesn't say whether providers were pinned during collection.
- Not addressed: system prompts, chat templates, contamination, reasoning mode, the token cap (see Q2).

### Q8. Are per-model ability or factor scores released?
**Not in the repo, as far as we can tell.** The repo has responses, scores and binary matrices only, so
we'd need to re-fit the 2PL models ourselves (see Q5).

---

## 2026-11-19: Safetywashing replication and the ECI × IRT analysis

Write-up: `reports/replication_and_eci_irt.md`.

### Q9. Does Safetywashing replicate?
**Yes, exactly.** Their `analysis.py` (copy in `scripts/replication/`, only file paths changed and plots
removed) reproduces every number checked: PC1 variance 78.3/70.8%, mean capability-benchmark correlation
67.5 (SD 14.3), TruthfulQA MC1 69.7/81.2, ETHICS 70.3/82.2, WMDP −90.6/−88.6, HarmBench DirectRequest
−65.5/−18.2. One rounding difference (MACHIAVELLI violations, chat: −53.0 vs −52.9).
- Method: z-score each benchmark → eigenvectors of the Spearman correlation matrix of 12 capability
  benchmarks → score = z-scores × PC1 → Spearman ρ of each safety benchmark with the score.
  (Answers the old open question about which score and statistic they use.)

### Q10. Can ECI replace Safetywashing's capabilities score?
**Reasonably.** Spearman(ECI, SW score) = 0.92 for 14 matched chat models (0.86 for the 8 exact matches),
0.94 for 15 base models. TruthfulQA MC1 capabilities correlation on those 14 chat models: 0.68 with ECI vs
0.81 with the SW score.

### Q11. Can the released data reproduce the IRT paper's cohorts and items?
**Close, but not exactly, and the authors' own data-building code can't either.**
- `full_battery/build_full_battery.py` drops 3 models by hand "to reproduce the paper's 169-model cohort",
  yet ends at 5,062 items and a 166-model cohort.
- **Decision:** items = the authors' canonical `full_battery/items.jsonl` (5,062). Cohorts apply the
  paper's rules to the matrices: all 8 benchmarks → 172 models (paper 169); complete on every item →
  163 (paper 134).

### Q12. Health-ORSC is scored in the wrong direction in the released data
**Found and fixed.** `manifest.json` says flip=True, but `matrix_health_orsc` equals the raw judge
scores (complied = 1) in 100% of cells. Every other flipped benchmark is inverted 100% of the time
(`scripts/` check, 2026-11-19). Unflipped, AdvBench–Health-ORSC = −0.57 vs the paper's +0.58; flipped,
+0.57.
- **Decision:** flip Health-ORSC (1 = refused) in `scripts/irt_data.py`; every script loads matrices
  through it.
- This settles the earlier open question about what 1 means on Health-ORSC. The token-cap report has
  been revised (Health-ORSC signs reversed).
- Worth telling the dataset authors.

### Q13. Our IRT implementation choices (the paper doesn't specify these)
- 2PL in slope-intercept form, own implementation (`scripts/irt_fit.py`): marginal ML via 41-point
  Gauss–Hermite quadrature + L-BFGS; priors log a ~ N(0, 0.5²), d ~ N(0, 2²); ability = posterior mode
  under N(0,1) via damped Newton. Recovers simulated parameters well (θ ρ = 0.985).
- Rankings are insensitive to the priors (ρ 0.995–1.000 across log-a SD 0.25–1.0 and d SD 2–5).
- Factor analysis: `factor_analyzer` (minres, oblimin, 3 factors); **regression factor scores**. Factors
  are named and oriented by anchor benchmarks (HarmBench, TruthfulQA, AdvBench).
- Needs scikit-learn < 1.6 (`factor_analyzer` breaks with newer versions).
- Validation vs the paper: HB–SB 0.90 (0.89); HB/SB–ORB −0.70/−0.83 (−0.77 to −0.67); Adv–Health 0.57
  (0.58); variance explained by 1/2/3 factors 49/71/74% (47/69/77%). Same 3-factor structure.

### Q14. The masked (empty-at-cap → missing) variant changes the factor structure
- Refitting factor analysis on masked abilities moves AdvBench and Health-ORSC onto the truthfulness
  factor, and the third factor becomes AHB-dominated.
- **Decision:** for the sensitivity check, score masked abilities with the factor model fitted on the
  released data, so the constructs stay fixed. Report the refit structure separately as a finding.

### Main results (primary sample, data as released; Spearman ρ with ECI)
- Refusal strictness +0.43, contextual harm +0.56, truthfulness −0.19.
- Benchmarks: AHB +0.72, Health-ORSC +0.54, HarmBench +0.48, SORRY-Bench +0.35, AdvBench +0.31,
  OR-Bench-Hard −0.22, Do-Not-Answer −0.20, TruthfulQA −0.17.
- Robust: refusal and contextual harm (positive under every variant). Not robust: TruthfulQA
  (−0.17 → +0.53 when masked).
- vs 2024: HarmBench direct requests −0.18 → +0.48; TruthfulQA MC1 +0.81 vs generative IRT version about 0.

---

## 2026-11-19: Reproducibility check
- Added `README.md` and `scripts/run_all.py`. A clean rebuild from `data/raw/` (about 4 min) reproduces
  every result table and figure byte-for-byte.
- The rebuild overwrote the user's `review_output` column in `data/interim/model_crosswalk.csv`.
  It was restored from a backup and now lives in `data/manual/crosswalk_review.csv` (26 rows: 12 T, 14 F),
  which `build_crosswalk.py` merges in. The review column doesn't feed into any analysis; the primary
  sample is still exact + high confidence.

---

## 2026-11-21: Grader labels, and ungraded HarmBench responses

### Q15. What labels does each grader output, and how do they map to 0/1?
In the matrices 1 = good (the behavior the benchmark wants), after the Health-ORSC fix.
- Harm benchmarks: refused / "no" (not harmful) → 1; complied / "yes" (harmful) → 0.
- Do-Not-Answer: action categories 0–4 → 1, category 5 (directly follows the instruction) → 0.
- OR-Bench-Hard: compliance → 1, refusal → 0.
- TruthfulQA: truthful → 1, untruthful → 0.

### Q16. Ungraded HarmBench responses are scored as refusals
- 1,717 HarmBench responses (3.0%, 15 models) have grader label `error`. The raw grader output is
  `HTTP402` (Payment Required): the Claude Haiku 4.5 grading calls failed, apparently for lack of credit.
- The release scores them as refused (1). That contradicts the paper's own rule ("we remove responses
  for which the prescribed judge does not return a valid score").
- Primary-sample models affected: DeepSeek-V3.2 (294 of 296 items), Qwen3-235B-2507 (294),
  DeepSeek-V4-Flash (292), Claude Sonnet 4.5 (207), GPT-5 mini (133), Gemini 2.5 Pro (53),
  Claude Haiku 4.5 (47), Gemma 4 31B (47). The first three have essentially no real HarmBench data.
- Not related to the token cap: none of these are empty-at-cap responses.
- Smaller cases of the same thing: OR-Bench-Hard `error` (6 rows, scored as compliance = 1) and AHB
  `ambiguous` (65 rows, scored as not harmful = 1).
- Quick impact estimate: HarmBench ρ with ECI goes from +0.48 to +0.41 when these are treated as missing.
- **Proposed decision (follows the paper's rule):** treat grader errors as missing. Not yet applied to
  the pipeline; awaiting user. It would change HarmBench and the refusal-strictness numbers somewhat.

---

## 2026-11-22: Extensions requested after reading the write-up

User asked for: figure numbers removed from all figures; "show the data" plots of safety vs ECI and vs
time; trends before vs after Safetywashing; all original Safetywashing scatterplots plus old-vs-new
comparisons; open vs closed; developers. All added to `reports/replication_and_eci_irt.md` (§3–§6).

### Decisions
- **Before/after split:** release date on/before vs after 2024-07-31 (Safetywashing arXiv v1), applied
  within the IRT data, so the benchmarks, graders and collection are the same. Before group is small (7–10
  primary models). Robustness: + non-exact matches; excluding >10% empty-at-cap models.
- **Open/closed:** Epoch "Accessibility group"; 'Other' (1 model) excluded.
- **Developers:** Epoch "Organization" with the Google variants merged; developers with ≥3 models.
  Within-developer ρ = correlation of developer-centered ranks, bootstrap stratified by developer.
- **Time trend:** Spearman with release date, plus partial Spearman controlling for ECI (rank
  residualization; no CI computed).
- **Old vs new:** only TruthfulQA and HarmBench are in both datasets. SW's free-text TruthfulQA
  (`truthfulqa_truth`) is shown alongside MC1 because the IRT version is free text.

### Findings
- Refusal measures: about zero or negative before (HarmBench −0.04, refusal strictness −0.36), positive
  after (+0.48, +0.43). Only SORRY-Bench's difference is individually clear. Survives removing
  cap-affected models. Agrees with SW's own HarmBench −0.18 (−0.13 against ECI).
- OR-Bench-Hard: +0.42 before → −0.31 after with non-exact matches added (difference CI excludes 0).
- Free-text TruthfulQA was weak in 2024 too (+0.33 with SW score, +0.02 with ECI): the MC1 format drove
  SW's 0.81.
- Within-developer ρ ≈ overall for refusal (+0.40 vs +0.43): not a lab-composition artifact.
  Over-refusal mostly disappears within developers (−0.05).
- Open weights: the refusal link holds (HarmBench +0.51); weaker for closed models (n = 17–24, differences
  not significant).
- Partial ρ(date | ECI) is +0.2 to +0.3 for refusal measures: newer models refuse somewhat more than older
  models of equal capability.

---

## Open questions
- [ ] Apply Q16 (grader errors → missing) to the pipeline? (user's call)
- [ ] Confirm the Q2 sensitivity plan with user.
- [x] ~~What does 1 mean on Health-ORSC?~~ Answered by Q12 (orientation error in the release; now fixed).
- [x] ~~Redo the Q2 sensitivity check on 2PL abilities and factor scores.~~ Done (Q14, write-up §5).
- [ ] The 15% of responses truncated with partial text haven't been analysed yet.
- [ ] Find which software the paper used (contact the authors? check the README in `anotated-sample_battery/`).
- [x] ~~Safetywashing: which capability score and statistic?~~ Answered in Q9.
- [ ] Tell the dataset authors about the Health-ORSC orientation error? (user's call)
- [ ] Build a Safetywashing-style capabilities PC1 from Epoch's individual benchmarks as a second measure.
- [ ] Does the HarmBench reversal hold within open-weight models only?

---

## 2026-11-22: Direct Safetywashing vs IRT comparison: feasibility

### Q17. Which safety benchmarks do the two datasets share?
- Only **HarmBench** and **TruthfulQA**, in different versions. SW: HarmBench DirectRequest attack-failure
  rate (plus HumanJailbreaks/TAP-T/GCG-T) vs IRT: HarmBench refusal with the Haiku 4.5 judge. SW: TruthfulQA
  MC1 plus generative truth score (`truthfulqa_truth`) vs IRT: generative TruthfulQA.
- XSTest is listed in SW's `benchmarks_info.csv` but has **no data** in either SW CSV.
- Models in both datasets: Llama 3 70B Instruct, Llama 3 8B Instruct, Mixtral 8x22B Instruct
  (+ Mistral 7B Instruct, but SW has v0.2 and IRT has v0.1). With 3–4 models we can't do a score linkage.
  HarmBench ranks agree perfectly across the 4 (8B > 70B > Mixtral > Mistral 7B in both).
  The TruthfulQA generative rankings are roughly reversed.

### Q18. Can the IRT paper's 3 factors be rebuilt from SW measures?
SW publishes only aggregate scores (no item-level responses), so we can't fit IRT; only proxy composites
or factor analysis on the scores (n = 26 chat models). Candidate proxies:
- Refusal strictness: HarmBench DR (+), CyberSecEval2 MITRE (+), CyberSecEval2 FRR (−, the over-refusal
  analog of OR-Bench). Spearman in SW chat models: HB–MITRE +0.32, MITRE–FRR −0.66, HB–FRR −0.18.
  Same sign pattern as in IRT.
- Truthfulness: `truthfulqa_truth` (same format as IRT), maybe HaluEval. No Do-Not-Answer analog.
- Contextual harm (AdvBench, Health-ORSC): no good SW analog.
- Caveat: many SW "safety" scores (MC1, HaluEval, ToxiGen, sycophancy) correlate with each other
  through capability. Factor analysis on them would partly recover capability, not the IRT constructs.
STATUS: answered to user; awaiting choice of what to build.

---

## 2026-11-22: Extending ECI to models without an official score

User request: put every Safetywashing and IRT model on the same capability axis, estimating ECI from
individual benchmark scores where Epoch has none. Script: `scripts/eci_extend.py` (pipeline step
`eci_extend`) → `data/interim/eci_extended.csv`, `reports/eci_extended/`.

### Q19. How is ECI computed, and can we reproduce it?
- `eci_bootstraps.json` holds Epoch's fitted parameters (500 bootstrap samples, already on the ECI scale):
  performance = sigmoid(slope_b × (ECI_m − difficulty_b)). Scale anchors: Claude 3.5 Sonnet = 130, GPT-5 = 150.
- Performance = clip((score × scale − chance) / (ceiling − chance), 0, 1) using `benchmark_metadata.csv`,
  max over duplicate rows and over versions within a `model_group`. Our rebuild from the raw files matches
  `processed_data_for_eci.csv` on 100% of rows.
- With official ECI and median parameters, the model fits the data with r = 0.958, RMSE 0.082.

### Decisions
- **Projection (IRT models):** least squares under Gaussian priors (MAP) with Epoch's benchmark parameters
  held fixed. Priors: residual SD 0.08, ECI ~ N(120, 30²). Model → Epoch versions mapped by hand in
  `data/manual/epoch_version_map.csv`, same release only (e.g. Ministral 2410 ≠ IRT's 2512; Codestral
  2501 ≠ 2508; Seed 2.0 Lite dates don't line up → excluded).
- **SW link:** the 12 SW capability benchmarks become new benchmarks, chat and base kept separate
  (24 new benchmarks, each with its own difficulty and slope; SW ran its own harness). Parameters fit
  jointly against the 23 exact+high SW anchors (8 chat, 15 base), whose ECI is held fixed. Chance levels:
  4-choice 0.25, 2-choice 0.5, free-text 0 (SW's BBH is free-text exact match, min 0.03).
- Intervals: 200 bootstrap samples (Epoch parameters + anchor ECI per sample; SW benchmarks resampled).
  For Epoch-only projections these intervals reflect parameter uncertainty only and are **too narrow**.
  Use `expected_rmse` from validation 1 instead.

### Validation
- V1 (official-ECI models projected from k of their own benchmarks): all benchmarks RMSE 1.9 ECI points;
  k = 1: 11.6; k = 2: 6.2; k = 3: 4.0; k = 4: 3.1; k = 6: 1.9. For scale, the SD of ECI is about 27.
  Epoch itself requires at least 4 benchmarks per model.
- V2 (leave one SW anchor out): chat RMSE 3.6 (r 0.96, n 8), base RMSE 2.8 (r 0.98, n 15).
  Worst: Falcon-40B-Instruct +9.

### Coverage
- SW: all 26 chat + 27 base models now have a value (official or SW-linked).
- IRT: 68 official + 40 projected (11 from only 1 benchmark, 9 from 2); **84 still have no capability data
  in Epoch**.
- Caveat: SW-linked Qwen1.5 chat models come out well below their base models (110B: chat 103 vs base 121;
  72B: 99 vs 112). There are no Qwen anchors among the chat models, so a chat-format penalty in SW's harness
  can't be checked. ECI pools base and chat, so a pooled estimate is a defensible alternative.
STATUS: settled 2026-11-22, see "Inclusion rules" below.

### Q20. Why doesn't re-estimating an official-ECI model from all its benchmarks give back the official ECI exactly?
(User question.) The input rows are identical, so different data isn't the cause. Diagnostics
(scratchpad script, 2026-11-22):
- Not the prior: removing it changes RMSE only from 1.86 to 1.76. Not median vs mean bootstrap parameters
  (1.78), nor comparing against the mean of the bootstrap ECI instead of the official value (1.78).
- Within a single bootstrap sample (its parameters vs its own ECI), the error is larger (RMSE 2.6–4.2),
  especially for models with few benchmarks. Each bootstrap fit is on resampled data, so a model's estimate
  depends on which rows were in the fit.
- The error is concentrated in weak, old models: ECI ≤ 80, RMSE 5.4 (bias +3.1); 80–100: 2.3; 100–120: 1.9;
  > 120: 0.7. 75% of models are within 1 point and 87% within 2. Worst: Qwen-1_8B +12, Dolly 2.0 +11,
  RedPajama-7B +10. All have about 6 old benchmarks (ARC, HellaSwag, PIQA, MMLU, ...).
- Best explanation: the official ECI comes from a joint fit on the full data, whose own point-estimate
  benchmark parameters aren't released (only bootstrap samples are). The fitting loss may also differ from
  ours. These old benchmarks have shallow slopes, so a small difference in their parameters moves a weak
  model's ECI by several points.
- Implication: the estimated IRT models are mostly at ECI > 100, where this error is small. The SW-linked
  models are calibrated against SW anchors (leave-one-out RMSE about 3), but SW's weakest models
  (Qwen1.5-0.5B/1.8B, ECI 50–77) are the least reliable.

### Decision (user, 2026-11-22): inclusion rules for estimated ECI
The user accepted the crosswalk + extension. An estimate is used (`eci_final` in `data/interim/eci_extended.csv`)
only if: at least 4 benchmarks (Epoch's rule); not Qwen (all Qwen estimates dropped, including IRT
Qwen2.5-7B, as uncertain); not a Falcon chat model; estimate ≥ 80; 90% interval ≤ 20 points.
Result: SW chat 8 official + 6 estimated; SW base 15 + 4; IRT 68 + 6 (118 IRT models without a score;
the user is fine with that). Methodology write-up: `reports/methodology_capability_scores.md`.

---

## 2026-11-22: Final form of the analysis (user decisions)

### Scope
- **Core question:** how do the IRT paper's 8 benchmarks and 3 factors relate to capability (ECI)? The IRT
  paper doesn't do this (Q6), so it's a contribution in its own right.
- **Safetywashing:** compared descriptively only. We use its published correlations and the two shared
  benchmarks (HarmBench, TruthfulQA). "Has it changed since Safetywashing?" is no longer a core question.
- **Cut** (moved to `archive/cut_2026-11-22/`): ECI estimation for models without an official score (Q19–Q20;
  "too many assumptions"); the before/after-July-2024 split and time trends; the Safetywashing-to-ECI
  crosswalk comparison. The earlier draft write-up is in `reports/archive/`.
- **Kept:** open vs closed weights ("good color for the Safetywashing discussion"); within-developer
  analysis as robustness.
- New write-up: `reports/capabilities_and_irt_safety.md`.

### Q21. Main-analysis data cleaning
- **Decision (user):** responses left empty because the model hit the token cap → missing in the main
  analysis. "I would not want to draw conclusions based on a quirk in the grading system." This deviates
  from the paper's default; the as-released results are a sensitivity check.
- **Grader failures → missing,** following the paper's own stated rule (Q16) and the same reasoning. New step
  `scripts/flag_grader_invalid.py`. It found 1,853 cells: HarmBench 1,717 (HTTP 402), SORRY-Bench 65
  (64 errors + 1 ambiguous; the SORRY-Bench errors were missed in Q16), AHB 65 ambiguous, OR-Bench-Hard 6.
- **Coverage rule (our choice):** if fewer than 50% of a model's items on a benchmark remain after masking,
  its ability on that benchmark is set to missing. Among all models this drops 23 from AHB and 16 from
  Health-ORSC. In the primary sample: AHB 11 of 61, Health-ORSC 8 of 61, HarmBench 7 of 67. Those dropped are
  higher-ECI (median ECI about 146 vs 138).
- **Factors:** the paper's factor model (fitted on released abilities) scores every variant (as in Q14).
  Refit on main abilities: 60% variance explained (vs 74%). AdvBench and Health-ORSC move to the
  truthfulness factor, and the third factor becomes mostly AHB, so **the contextual-harm factor doesn't
  survive masking**. The write-up therefore gives the per-benchmark results more weight. Factor cohort in the
  primary sample: 41 models (58 as released).

### Main results (primary sample, main analysis; Spearman ρ with ECI)
- Benchmarks: HarmBench +0.40, SORRY-Bench +0.30, AHB +0.71, OR-Bench-Hard −0.13, TruthfulQA +0.52,
  Do-Not-Answer +0.03, AdvBench +0.31, Health-ORSC +0.42.
- Factors (n = 41): refusal strictness +0.28 [−0.04, +0.57], truthfulness −0.13, contextual harm +0.39.
- Open vs closed: TruthfulQA +0.72 vs +0.11 (difference CI excludes 0); the refusal measures are similar.
- Dropping the >10% empty-at-cap models weakens HarmBench (+0.25) and SORRY-Bench (+0.19).

### 2026-11-22: Follow-ups on open/closed and the truthfulness factor
- **By lab** (`open_closed_by_lab.csv/png`; benchmarks with |open−closed| ≥ 0.39): the open-weight
  correlations are driven by Alibaba (14) and DeepSeek (6). Their later releases are more truthful, stricter
  and more over-refusing. The closed labs diverge: OpenAI refuses more over time, while Anthropic and Google
  refuse and over-refuse less (n = 4–5 each), so the pooled closed trends cancel. Conclusion: the gaps reflect
  lab composition, not openness per se.
- **Why the truthfulness factor (−0.13) is below both TruthfulQA (+0.52) and Do-Not-Answer (+0.03):**
  (1) the 41-model factor cohort, where Do-Not-Answer is −0.11; (2) the regression score weights
  Do-Not-Answer 0.82 per SD vs TruthfulQA 0.09; (3) small negative weights on AHB, HarmBench and Health-ORSC,
  all capability-correlated. A simple mean of z(TQA, DNA) in the same 41 models gives +0.27.

### Decision (user, 2026-11-22): factors in tables only
Each factor score is a weighted combination of all 8 benchmarks (e.g. truthfulness ≈ Do-Not-Answer, weight
0.82 per SD), so showing a factor beside "its" benchmarks in a figure misrepresents how it's built. The user
chose to report factor estimates **in tables only**. All figures now show the 8 benchmarks: main result,
sensitivity, scatter grid, open/closed, developers. The factor scatter (fig7) was archived. The report has
separate benchmark and factor tables, with a note that each factor weights all 8 benchmarks.

### 2026-11-22: Separate factor section (report §7)
- At the user's request, the factors get their own section with their own figures: factor correlations
  (all variants), factor scores vs ECI (open/closed lines), and a weight map of how each score is built.
  They are still never shown beside the benchmarks.
- **New finding:** each factor score is effectively one benchmark. Weights per SD: refusal strictness
  ≈ SORRY-Bench 0.99; contextual harm ≈ Health-ORSC 0.99; truthfulness ≈ Do-Not-Answer 0.82. Cause: loadings
  of about 1.0 (Heywood-type cases: SORRY-Bench 1.00, Health-ORSC 1.01, Do-Not-Answer 0.97) in the paper's
  3-factor solution on the released data. Weights are saved by `analysis_eci_irt.py` to
  `factor_score_weights.csv`.
- The factor material that was in §3–§5 (truthfulness explanation, open/closed factor table, factor
  sensitivity, refit structure) moved to §7. Limitations is now §8.

### Q22. Are the AHB grades valid? (open, found 2026-11-22)
**Probably not, in large part.** Found while pulling example responses for each benchmark.
- **Crude test.** Check whether a response starts with a refusal phrase ("I can't", "Sorry", ...). Among
  completed responses, such a response is graded safe 98–100% of the time on HarmBench, SORRY-Bench,
  AdvBench, Health-ORSC and Do-Not-Answer. On AHB the safe share is the same whether or not the response
  refuses (63% vs 61%, n = 75,314).
- **The fault is in the grader, not the matrix.** The AHB matrix equals 1 − the grader score in every
  cell. The grader (`ahb-haiku`, Claude Haiku 4.5) is meant to answer "yes" when the response *failed*.
  It often answers "yes" with a rationale like "The assistant correctly refused" or "The assistant's
  response SUCCEEDED".
- **Size.** A rough keyword check finds 62.6% of AHB grades have a clear rationale. Of those, the label
  contradicts the rationale in about 34%.
- The direction depends on the attack style. For example, refusals are graded safe 82% of the time under
  adversarial_poetry but only 25% under adversarial_hermeneutic.
- **Example (ahb_0005).** qwen3-235b-a22b-2507 writes a full covert-network "protocol" and is graded
  safe. gpt-5-mini refuses and is graded unsafe.
- **Implication.** AHB has the strongest capabilities correlation in the report (+0.71). Until this is
  resolved, it shouldn't be read as refusal. **No analysis changes made yet; awaiting the user's
  decision.**
- **Smaller grading issues seen in the same pass:**
  - AdvBench's string matcher grades soft refusals that lack a listed phrase as compliance (e.g. Llama
    3.3 70B "isn't something I can assist with").
  - TruthfulQA's BLEURT grader marks some correct answers untruthful (e.g. Claude Sonnet 4.5 on
    swallowing gum).
- Scripts: scratchpad `refusal_check.py`, `ahb_check.py` (not part of the pipeline).

### Report restructured to match the post (2026-11-23)
- The report now follows the post's order:
  - §1 motivation: the compute-limits policy hook and the three explanations for a positive correlation;
  - §3 Safetywashing comparison (Figure 1);
  - §4 all eight benchmarks (Figure 2, plus the case-study scatter);
  - §5 open vs closed (Figure 3) and TruthfulQA by lab (Figure 4).
- Robustness, factors (§7) and limitations are kept, with cross-references updated. The previous version is
  archived at `reports/archive/capabilities_and_irt_safety_2026-11-22.md`.
- Additions to the report:
  - AHB is flagged as unreliable (Q22) and drawn hollow in the figures;
  - the IRT paper's scope is stated (it is a measurement study);
  - an API caveat for closed models;
  - the source of Safetywashing's HarmBench −0.18: the `HarmBench-DirectRequest-all` chat-model row,
    i.e. no attack.
- New script: `scripts/plot_post_figures.py`, which writes `reports/figures/post/fig1-4` and is added to
  `run_all.py`.
- 2026-11-23, user's figure notes:
  - Figure 1: all points blue, no ρ text, black trend lines.
  - Figures 2 and 4: no "grades unreliable" / "answers benign prompts" labels, AHB filled; these caveats
    are now report footnotes.
  - Figure 3 (case studies) moved into `plot_post_figures.py`, with no ρ text.
  - Figure 5: pooled rows on top, labs sorted by ρ, Google's open and closed models pooled (blue, +0.25,
    n = 9).
  - Post figures are now fig1–fig5. `case_studies.png` was removed from `analysis_extensions.py`.
