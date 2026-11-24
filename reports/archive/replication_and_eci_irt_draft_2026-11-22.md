# Safetywashing, replicated and redone on newer models

*Draft, updated 2026-11-22. Code: `scripts/replication/safetywashing_original_copy.py` (their script,
unmodified apart from file paths and the plotting section), `scripts/irt_fit.py`,
`scripts/analysis_eci_irt.py`, `scripts/plot_eci_irt.py`, `scripts/analysis_extensions.py`,
`scripts/plot_safetywashing.py`. Numbers: `reports/eci_irt/`. Decisions and open questions:
`notes/journal.md`. Rebuild everything with `scripts/run_all.py`.*

## Summary

1. **The Safetywashing analysis replicates exactly** from its public data and code. Every summary
   statistic and capabilities correlation we checked matches the paper. All their scatterplots are
   reproduced below and in `figures/sw_all_scatterplots.pdf`.
2. **The Epoch Capabilities Index (ECI) is a reasonable stand-in for Safetywashing's capabilities
   score.** On the models in both datasets, the two rank models almost identically (Spearman ρ = 0.86–0.94).
3. **Our refit of the IRT paper's pipeline recovers its published structure,** after fixing a
   scoring-direction error in the released Health-ORSC data.
4. **New result (2023–26 models, IRT safety suite):**
   - *Refusal-type measures* track capabilities moderately: refusal strictness ρ = +0.43, contextual harm
     +0.56, individual refusal benchmarks +0.31 to +0.72. **More capable models refuse harmful
     requests more.**
   - *Over-refusal* (OR-Bench-Hard) is weakly negative (−0.22): more capable models refuse benign
     prompts slightly *more*.
   - *Truthfulness* is uncorrelated with capability (−0.19), but this depends on how responses cut off by
     the 1,024-token cap are treated.
   - No measure reaches the 0.8+ range Safetywashing found for TruthfulQA MC1, ETHICS or WMDP.
5. **Has it changed since Safetywashing? For refusal, yes.**
   - Among models released before the paper (July 2024), capability did **not** predict refusal. That
     holds in Safetywashing's own data (HarmBench −0.18) and in the IRT data's pre-2024 models (HarmBench
     −0.04, refusal strictness −0.36).
   - Among models released since, it does (HarmBench +0.48, refusal strictness +0.43).
   - The pre-period sample is small (7–10 models), so individual differences are uncertain. But the two
     independent datasets agree, and the post-period correlations survive removing the token-cap-affected
     models.
6. **TruthfulQA's high capabilities correlation was a feature of the multiple-choice format, not of the
   era.** Free-text TruthfulQA tracked capability only weakly in 2024 (+0.33, and +0.02 against ECI), and
   still doesn't now (−0.17).
7. **The capability–refusal link isn't explained by which lab built the model.** Within a developer, more
   capable models also refuse more (within-developer ρ +0.40 for refusal strictness, +0.50 for
   contextual harm). It is somewhat stronger among open-weight models than closed ones, though the
   differences are within the noise.

---

## 1. Replicating Safetywashing

Their method (from `analysis.py`):
- Standardize every benchmark.
- Capabilities score = the data projected onto the first principal component of the Spearman correlation
  matrix of 12 capability benchmarks.
- "Capabilities correlation" of a safety benchmark = its Spearman ρ with that score, computed separately
  for 27 base models and 26 chat models.

We ran their script, unchanged apart from file paths and removing the plots. Selected values:

| Quantity | Paper | Our run |
|---|---|---|
| PC1 variance explained, base / chat | 78.3% / 70.8% | 78.3% / 70.8% |
| Mean (SD) correlation between capability benchmarks, chat | 67.5 (14.3) | 67.5 (14.3) |
| TruthfulQA MC1, base / chat | 69.7 / 81.2 | 69.7 / 81.2 |
| ETHICS, base / chat | 70.3 / 82.2 | 70.3 / 82.2 |
| WMDP (all), base / chat | −90.6 / −88.6 | −90.6 / −88.6 |
| HarmBench DirectRequest (all), base / chat | −65.5 / −18.2 | −65.5 / −18.2 |
| Sycophancy, base / chat | −65.6 / −66.8 | −65.6 / −66.8 |
| BBQ ambiguous, base / chat | 30.8 / −37.3 | 30.8 / −37.3 |

The only difference anywhere is one rounding (MACHIAVELLI violations, chat: −52.9 in the paper,
−53.0 in our run). Full output: `data/interim/safetywashing_original_output.txt`.

### The original scatterplots

![Safetywashing headline benchmarks, chat models](figures/sw_original_chat.png)

*Safetywashing's headline safety benchmarks against their capabilities score, for their 26 chat models.
The ρ in each panel is the capabilities correlation the paper reports. Base models:
`figures/sw_original_base.png`. Every safety benchmark in their data (chat and base, 18 pages):
`figures/sw_all_scatterplots.pdf`.*

This is the evidence behind Safetywashing's conclusion. Benchmarks like ETHICS, TruthfulQA MC1, BBQ
disambiguated, GPQA, QuALITY and ANLI rise almost in lockstep with capability. WMDP, sycophancy and the
jailbreak benchmarks fall with it.

**Can ECI replace their capabilities score?** On the models in both datasets, Spearman(ECI,
Safetywashing score) = 0.92 for the 14 matched chat models (0.86 on the 8 exact matches) and 0.94 for
the 15 base models. With ECI in place of their score, TruthfulQA MC1's capabilities correlation on those
14 chat models is 0.68, against 0.81 with their own score (n = 14, so the difference is within noise).

## 2. Redoing it with ECI and the IRT safety suite

Choices follow the IRT paper (Rivera et al. 2026) wherever it specifies one:

| Step | Choice | Source |
|---|---|---|
| Safety data | 8 benchmarks, 192 models, temperature 0, 1,024-token cap | IRT data release |
| Items | Authors' canonical battery, 5,062 items (paper: 5,067) | `full_battery/items.jsonl` |
| Scoring direction | 1 = the behavior the benchmark wants; **Health-ORSC flipped to match the paper** (see below) | paper + our fix |
| Per-benchmark ability | 2PL per benchmark, marginal ML, log-normal prior on discrimination, normal prior on intercept, posterior-mode ability under N(0,1) | paper (prior widths ours) |
| Factors | Minimum-residual factor analysis, oblimin rotation, 3 factors, on models complete on every item | paper (factor-score method ours: regression) |
| Capabilities score | Epoch Capabilities Index | replaces Safetywashing's PC1 |
| Statistic | Spearman ρ, plus 95% bootstrap CIs | Safetywashing (+ CIs) |
| Models | Primary sample: 67 models with an exact, high-confidence ECI match | crosswalk decision |

**Does our refit reproduce the paper?**

| Check | Paper | Ours |
|---|---|---|
| HarmBench vs SORRY-Bench ability | 0.89 | 0.90 |
| HarmBench / SORRY-Bench vs OR-Bench-Hard | −0.77 to −0.67 | −0.70 / −0.83 |
| AdvBench vs Health-ORSC | +0.58 | +0.57 |
| Variance explained by 1 / 2 / 3 factors | 47 / 69 / 77% | 49 / 71 / 74% |
| Factor structure | HB, SB, AHB vs ORB; TQA + DNA; AdvBench + Health-ORSC | same |
| Cohorts (all 8 benchmarks / complete on every item) | 169 / 134 | 172 / 163 |

The cohorts differ slightly. The authors' own data-building code doesn't reproduce the paper's cohort
either: it drops three models by hand and lands at 5,062 items.

**A scoring-direction error in the released data.** `manifest.json` says Health-ORSC is flipped so that
1 = refused. In fact the released matrix equals the raw judge scores (complied = 1) in 100% of cells,
whereas every other flipped benchmark is inverted 100% of the time. Unflipped, AdvBench–Health-ORSC
comes out at −0.57 instead of the paper's +0.58. We apply the flip in one place
(`scripts/irt_data.py`) so every analysis uses it.

## 3. Main result

![Capabilities correlations of the IRT safety measures](figures/fig5_capabilities_correlations.png)

| Measure | ρ with ECI [95% CI] | n |
|---|---|---|
| **Refusal strictness** (factor) | **+0.43 [+0.21, +0.61]** | 58 |
| **Truthfulness** (factor) | **−0.19 [−0.45, +0.09]** | 58 |
| **Contextual harm** (factor) | **+0.56 [+0.36, +0.72]** | 58 |
| HarmBench | +0.48 [+0.29, +0.64] | 67 |
| SORRY-Bench | +0.35 [+0.12, +0.55] | 63 |
| AHB | +0.72 [+0.59, +0.82] | 61 |
| OR-Bench-Hard (1 = answers benign prompt) | −0.22 [−0.47, +0.04] | 63 |
| TruthfulQA | −0.17 [−0.40, +0.08] | 64 |
| Do-Not-Answer | −0.20 [−0.44, +0.06] | 64 |
| AdvBench | +0.31 [+0.10, +0.50] | 64 |
| Health-ORSC | +0.54 [+0.32, +0.71] | 61 |

How to read this the way Safetywashing would:
- A high capabilities correlation means a benchmark's progress may simply come along with general
  capability gains, so a model could look safer on it just by being more capable.
- Safetywashing treats |ρ| < 0.4 as low. In 2024 it found TruthfulQA MC1, ETHICS, GPQA and others at
  0.7–0.9. Here only AHB (+0.72) is in that range.
- These refusal results are **not obviously safetywashing**. Refusing harmful requests is a deliberately
  trained behavior, not a by-product of capability. And because over-refusal rises too (OR-Bench-Hard
  −0.22), part of it is general refusal strictness rather than better judgment.

### Showing the data: safety against capabilities

![Safety measures against capabilities](figures/safety_vs_eci.png)

*Every model in the primary sample. Gray: released on or before 31 July 2024 (before Safetywashing);
blue: released after. Lines are least-squares fits within each period.*

Things the correlations alone don't show:
- **The positive refusal correlations come from the post-2024 models.** The blue points slope up in
  every refusal panel. The gray pre-2024 points slope down for refusal strictness, SORRY-Bench and
  AdvBench, are flat for AHB, and slope up only slightly for HarmBench and Health-ORSC.
- **HarmBench has a ceiling.** A cluster of models sits at the maximum (refused everything). Some of these
  are the models whose HarmBench responses were never actually graded (see Caveats).
- **TruthfulQA and Do-Not-Answer are flat**, with a few low outliers. Those outliers are mostly models hit
  hard by the token cap.

### Factor scores and the token cap

![Factor scores against capabilities](figures/fig7_factor_scatter.png)

*Hollow orange: models with more than 10% of responses empty because they hit the 1,024-token cap.*

The heavily cap-affected models (mostly high-ECI reasoning models) make up much of the high end of
refusal strictness and contextual harm, and the low end of truthfulness.

## 4. Has the Safetywashing picture changed?

### The benchmarks both datasets share

![Safetywashing data vs new data](figures/sw_vs_now.png)

*Left: Safetywashing's 2024 chat models against their capabilities score. Middle: the same models against
ECI (the 14 that match an ECI model). Right: the new IRT data against ECI.*

| Benchmark | 2024 models vs SW score | 2024 models vs ECI | New models vs ECI |
|---|---|---|---|
| TruthfulQA, multiple choice | +0.81 (n=26) | +0.68 (n=14) | not in new data |
| TruthfulQA, free text | +0.33 (n=26) | +0.02 (n=14) | −0.17 (n=64) |
| HarmBench, direct requests | −0.18 (n=26) | −0.13 (n=14) | **+0.48** (n=67) |

- **HarmBench has reversed.** In 2024, more capable models were slightly *easier* to get harmful outputs
  from. Now they refuse more. The middle column shows this isn't an artifact of switching capability
  measures: the 2024 models are still slightly negative against ECI.
- **TruthfulQA hasn't changed; the format explains Safetywashing's finding.** The free-text version, which
  is what the IRT data uses, was only weakly related to capability even in 2024.
- Only these two benchmarks appear in both datasets. ETHICS, MACHIAVELLI, WMDP, BBQ and the rest have no
  newer large-scale data we could use, so we can't say whether their Safetywashing conclusions still hold.

### Before vs after Safetywashing, within the IRT data

Same benchmarks, same graders, same collection, split by release date (before vs after 31 July 2024).

![Capabilities correlations before vs after Safetywashing](figures/pre_post_correlations.png)

| Measure | Before (n) | After (n) | Difference [95% CI] |
|---|---|---|---|
| Refusal strictness | −0.36 (7) | +0.43 (51) | +0.79 [−0.35, +1.31] |
| Contextual harm | +0.18 (7) | +0.54 (51) | +0.36 [−0.50, +1.38] |
| HarmBench | −0.04 (10) | +0.48 (57) | +0.52 [−0.12, +1.10] |
| SORRY-Bench | −0.39 (10) | +0.38 (53) | **+0.77 [+0.03, +1.19]** |
| AHB | +0.15 (9) | +0.65 (52) | +0.50 [−0.22, +1.32] |
| AdvBench | −0.19 (10) | +0.36 (54) | +0.55 [−0.23, +1.30] |
| OR-Bench-Hard | +0.16 (10) | −0.26 (53) | −0.42 [−1.16, +0.32] |
| TruthfulQA | −0.03 (10) | −0.14 (54) | −0.11 [−0.92, +0.71] |

- **Every refusal measure goes from about zero or negative before, to positive after.** Individually only
  SORRY-Bench's difference is clearly non-zero, because the before group is so small. Taken together, and
  alongside Safetywashing's own HarmBench result, the pattern is consistent.
- **It isn't the token cap.** Removing the 14 cap-affected models (all released after) leaves the
  after-period correlations essentially unchanged: refusal strictness +0.44, HarmBench +0.42, AHB +0.69.
- **Adding back the 11 non-exact matches** (before group n = 10–14) gives the same picture. With the larger
  sample, the over-refusal difference also becomes clear: OR-Bench-Hard goes from +0.42 before to −0.31
  after, difference −0.74 [−1.23, −0.08]. In other words, capable models used to over-refuse *less* and
  now over-refuse *more*.

Full table, including both robustness versions: `reports/eci_irt/pre_post.csv`.

### Over time

![Safety measures over time](figures/safety_vs_time.png)

*Shaded region: released before Safetywashing. Green: open weights; orange: closed weights.*

| Measure | ρ with release date [95% CI] | Holding ECI fixed |
|---|---|---|
| Refusal strictness | +0.47 [+0.25, +0.65] | +0.21 |
| Contextual harm | +0.60 [+0.42, +0.75] | +0.30 |
| AHB | +0.74 [+0.60, +0.83] | +0.31 |
| HarmBench | +0.47 [+0.28, +0.63] | +0.11 |
| OR-Bench-Hard | −0.28 [−0.51, −0.03] | −0.19 |
| TruthfulQA | −0.22 [−0.45, +0.03] | −0.16 |

- Newer models refuse more and over-refuse more. Release date and ECI are strongly related, so most of the
  date trend is the capability trend.
- But a modest part remains at the same capability ("holding ECI fixed": partial Spearman correlation,
  ranks of date and measure with ECI's rank regressed out). Newer models refuse somewhat more than older
  models of equal capability, consistent with safety training having intensified over time.

## 5. Open-weight vs closed models

![Open-weight vs closed models](figures/open_closed_correlations.png)

| Measure | Open weights (n) | Closed (n) | Difference [95% CI] |
|---|---|---|---|
| Refusal strictness | +0.50 (40) | +0.20 (17) | −0.30 [−0.94, +0.20] |
| Contextual harm | +0.65 (40) | +0.35 (17) | −0.30 [−0.88, +0.11] |
| HarmBench | +0.51 (42) | +0.38 (24) | −0.13 [−0.61, +0.29] |
| SORRY-Bench | +0.48 (42) | −0.02 (20) | −0.49 [−0.99, +0.06] |
| AHB | +0.73 (41) | +0.60 (19) | −0.13 [−0.65, +0.19] |
| OR-Bench-Hard | −0.43 (42) | +0.08 (20) | +0.51 [−0.03, +1.03] |
| Health-ORSC | +0.66 (41) | +0.26 (19) | −0.40 [−0.97, +0.10] |

- **The capability–refusal link holds among open-weight models alone** (HarmBench +0.51, refusal
  strictness +0.50). So the 2024→now reversal isn't just "closed models joined the sample".
- The link tends to be weaker among closed models, and for open models over-refusal also rises with
  capability (−0.43). With only 17–24 closed models, none of these differences is clearly beyond noise.

## 6. Developers

![Models by developer](figures/developers.png)

*Developers with at least 3 models with factor scores, ordered by mean ECI. Each dot is a model; the bar
is the developer mean. Anthropic and Moonshot are missing because fewer than 3 of their models have factor
scores: factor scores need a valid grade on every item, and several of their models have grading
failures.*

| Measure | Overall ρ | Within-developer ρ [95% CI] | Models / developers |
|---|---|---|---|
| Refusal strictness | +0.43 | +0.40 [+0.16, +0.63] | 52 / 7 |
| Contextual harm | +0.56 | +0.50 [+0.28, +0.69] | 52 / 7 |
| Truthfulness | −0.19 | −0.17 [−0.50, +0.18] | 52 / 7 |
| HarmBench | +0.48 | +0.51 [+0.33, +0.67] | 65 / 9 |
| AHB | +0.72 | +0.67 [+0.49, +0.80] | 59 / 9 |
| OR-Bench-Hard | −0.22 | −0.05 [−0.34, +0.20] | 61 / 9 |

(Within-developer ρ: ranks of ECI and the measure, each centered on the developer's mean, then correlated.
It asks: within one lab, do its more capable models score higher?)

- **Developers differ a lot in level.** Mistral's models are the least refusal-strict; Z.ai's and OpenAI's
  the most. Among the developers shown, Z.ai's models score lowest on truthfulness and Meta's highest.
- **But the capability–refusal relationship is nearly as strong within developers as across them**
  (+0.40 vs +0.43 for refusal strictness). It isn't an artifact of capable models coming from
  safety-focused labs: each lab's own more capable models refuse more.
- The over-refusal correlation mostly disappears within developers (−0.22 → −0.05). It seems to reflect
  differences *between* labs rather than a capability effect.

Developer means: `reports/eci_irt/developer_means.csv`.

## 7. Sensitivity

![Sensitivity of the capabilities correlations](figures/fig6_sensitivity.png)

| Measure | As released | Empty-at-cap → missing | Drop models >10% empty | + non-exact matches |
|---|---|---|---|---|
| Refusal strictness | +0.43 | +0.40 | +0.33 | +0.50 |
| Truthfulness | −0.19 | +0.08 | −0.07 | −0.15 |
| Contextual harm | +0.56 | +0.47 | +0.50 | +0.61 |
| TruthfulQA | −0.17 | **+0.53** | +0.11 | −0.16 |
| OR-Bench-Hard | −0.22 | −0.14 | −0.10 | −0.29 |

(CIs in `reports/eci_irt/capabilities_correlations.csv`. In the masked variant, factor scores come from
the factor model fitted on the as-released data, so the factors keep their meaning.)

- **Refusal strictness and contextual harm are robust** under every treatment.
- **TruthfulQA is not robust.** It goes from −0.17 to +0.53 when empty-at-cap responses are treated as
  missing.
- **The factor structure itself depends on the cap.** Refitting the factor analysis on the masked
  abilities moves AdvBench and Health-ORSC onto the truthfulness factor
  (`reports/eci_irt/factor_loadings_masked_refit.csv`).

## Caveats

- **Ungraded HarmBench responses (not yet corrected).** 1,717 HarmBench responses were never graded (the
  grading service returned "payment required"), yet the release scores them as refusals. This affects
  three primary-sample models almost entirely (DeepSeek-V3.2, Qwen3-235B-2507, DeepSeek-V4-Flash) and five
  more partly. Treating them as missing lowers HarmBench's ρ with ECI from +0.48 to +0.41. Awaiting a
  decision (journal Q16).
- **Small pre-Safetywashing sample.** 7–10 primary-sample models were released before 31 July 2024, so
  "before" correlations have CIs spanning most of [−1, 1].
- **Sample size overall.** n = 58–67, CIs about ±0.2.
- **Capability range.** Matched models span ECI 103–152. Today's frontier (ECI about 170) isn't in the
  IRT data at all.
- **Our own choices.** Prior widths, regression factor scores and the slightly different cohorts are ours
  (journal Q13). Ability rankings are insensitive to the priors (ρ 0.995–1.000 across a fourfold range).
- **Different capability measures.** ECI and Safetywashing's score are built from different benchmarks.
  The middle column of the old-vs-new comparison puts the 2024 models on ECI to check this.
- **Multiple comparisons.** Many measures × many splits. Read these as patterns, not as individual
  significance tests.

## Next steps

- Decide on the ungraded HarmBench responses (journal Q16) and rerun.
- Re-collect responses from the most cap-affected models with a larger token budget. That's the only
  clean way to settle the truthfulness result and the factor structure.
- Find newer data for the other Safetywashing benchmarks (ETHICS, WMDP, MACHIAVELLI, sycophancy) to test
  whether their conclusions still hold.
