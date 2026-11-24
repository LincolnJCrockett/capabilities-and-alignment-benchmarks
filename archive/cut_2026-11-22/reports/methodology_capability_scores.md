# Methodology: capability scores for models without an official ECI

**Purpose.** We relate safety benchmarks to capability using Epoch AI's Epoch Capabilities Index (ECI).
Some models in the Safetywashing data (Ren et al. 2024) and the IRT for AI Safety data (Rivera et al. 2026)
have no official ECI. For a subset of them we estimate a score on the ECI scale from their individual
benchmark results, so that both datasets share a single capability axis.

- Code: `scripts/eci_extend.py` (pipeline step `eci_extend` in `scripts/run_all.py`)
- Hand-checked inputs: `data/manual/epoch_version_map.csv`
- Output: `data/interim/eci_extended.csv`. Use the column `eci_final` (official ECI, or an estimate that
  passed the inclusion rules; otherwise empty).
- Validation tables and run log: `reports/eci_extended/`
- Decision log: `notes/journal.md`, Q19–Q20

---

## 1. Epoch's ECI model

Epoch fits a logistic model across benchmarks. For model *m* and benchmark *b*:

$$\text{performance}_{mb} = \sigma\big(s_b\,(\text{ECI}_m - d_b)\big), \qquad \sigma(x) = \frac{1}{1+e^{-x}}$$

Here *d_b* is the benchmark's difficulty and *s_b* its slope, both on the ECI scale. The scale is pinned by
Claude 3.5 Sonnet = 130 and GPT-5 = 150.

Each raw score is converted to a performance value before fitting:

$$\text{performance} = \text{clip}\!\left(\frac{\text{score}\times\text{scale} - \text{chance}}{\text{ceiling} - \text{chance}},\ 0,\ 1\right)$$

- Scale, chance level and ceiling come from Epoch's `benchmark_metadata.csv`.
- Duplicate results take the **maximum**, both within a model version and across versions in the same
  model group.

Epoch releases its benchmark parameters (*d_b*, *s_b*) and model ECIs as 500 bootstrap samples
(`eci_bootstraps.json`).

**Our checks**
- Rebuilding Epoch's processed data from the raw benchmark files with these rules reproduces
  `processed_data_for_eci.csv` on 100% of rows, exactly.
- With the official ECIs and median bootstrap parameters, the model fits Epoch's data with r = 0.958
  and RMSE 0.082 (on the 0–1 performance scale).

## 2. Two ways to estimate a missing score

All estimates use the same fitting rule: least squares with Gaussian priors (a MAP estimate).

| Quantity | Assumption |
|---|---|
| Residual SD | 0.08 (Epoch's own fit RMSE) |
| ECI prior | N(120, 30²) |
| New benchmark difficulty prior | N(120, 50²) |
| New benchmark log-slope prior | N(log 0.1, 1) |

The priors are weak. Removing the ECI prior changes the validation RMSE only from 1.86 to 1.76.

### 2a. Projection from Epoch benchmark data (IRT models)

Epoch's raw files contain results for some models that aren't part of ECI.

1. **Match each IRT model to Epoch model versions by hand** (`data/manual/epoch_version_map.csv`). Only the
   same release counts as a match. For example, Epoch's Ministral 3B/8B are the 2410 releases, while the IRT
   models are 2512, so they are not matched.
2. **Estimate the model's ECI from those benchmark results**, holding Epoch's benchmark parameters fixed.
   Only the model's own ECI is fitted.

### 2b. Linking through Safetywashing's capability benchmarks (Safetywashing models)

Safetywashing reports 12 capability benchmarks for every model: LogiQA, PIQA, HellaSwag, Winogrande, COPA,
MedQA, ARC-Challenge, MMLU, MATH, LAMBADA, GSM8K and BBH (`cap_names` in their `analysis.py`).

**Each is treated as a new benchmark** with its own difficulty and slope. Safetywashing ran its own
evaluation setup, so its MMLU score is not assumed to equal Epoch's MMLU score for the same model. Chat and
base models get separate parameters (24 new benchmarks in total).

**Chance levels:**
- 0.25 for four-option tasks
- 0.5 for two-option tasks (PIQA, Winogrande, COPA)
- 0 for free-text tasks (MATH, LAMBADA, GSM8K, and BBH, which Safetywashing scores by exact match, with
  scores as low as 0.03)

**Anchors.** The 23 Safetywashing models with an exact, high-confidence crosswalk match to an official ECI
(8 chat, 15 base) have their ECI held fixed. In one joint fit we estimate:

- the parameters of the 24 Safetywashing benchmarks, and
- the ECI of every other Safetywashing model, using its Safetywashing scores plus any Epoch benchmark
  results it has, with Epoch's parameters held fixed.

### 2c. Uncertainty intervals

We repeat the fit on 200 bootstrap samples. Each sample uses:

- Epoch's benchmark parameters and anchor ECIs from that sample, and
- the 12 Safetywashing benchmarks resampled with replacement.

We report 5th–95th percentile intervals.

For the IRT projections, these intervals reflect only uncertainty in the parameters, so they are too narrow.
For those models the expected error from validation 1 (below) is the better guide: about 3 ECI points for a
model with 4 benchmarks.

## 3. Validation

**Validation 1: projection.** We take every model that has an official ECI and re-estimate it from *k* of
its own Epoch benchmarks (20 random subsets per model).

| Benchmarks used | All | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|
| RMSE (ECI points) | 1.9 | 11.6 | 6.2 | 4.0 | 3.1 | 1.9 |

For scale, ECI varies across models with an SD of about 27. Even with all benchmarks, the re-estimate is not
exact. The gap is concentrated in weak, old models:

| Official ECI | ≤ 80 | 80–100 | 100–120 | > 120 |
|---|---|---|---|---|
| RMSE (ECI points) | 5.4 | 2.3 | 1.9 | 0.7 |

The input data are identical, so the gap is not caused by the data. Epoch's published scores come from a
joint fit whose own benchmark parameters aren't released (only bootstrap samples are). The old benchmarks
also have shallow slopes, so small parameter differences move weak models by several points. See journal
Q20.

**Validation 2: Safetywashing link.** We drop each anchor in turn and estimate its ECI from the remaining
anchors.

| Group | n | RMSE | Bias | r |
|---|---|---|---|---|
| Chat | 8 | 3.6 | +1.5 | 0.96 |
| Base | 15 | 2.8 | +0.8 | 0.98 |

The worst case is Falcon-40B-Instruct, at +9.

## 4. Inclusion rules

An estimate is used (`included = True`, so it appears in `eci_final`) only if **all** of the following hold.

1. **At least 4 benchmarks**, Epoch's own minimum per model. Safetywashing models always have 12.
2. **Not a Qwen model.** Safetywashing's Qwen1.5 chat models come out far below their base versions
   (110B: 103 vs 121; 72B: 99 vs 112), and no Qwen model is an anchor, so we can't check which is right.
   For consistency, the one IRT Qwen estimate (Qwen2.5-7B-Instruct) is dropped too.
3. **Not a Falcon chat model.** Falcon-40B-Instruct has the worst leave-one-out error (+9).
4. **Estimate of at least 80.** Below this, validation 1 shows large, biased re-estimation errors.
5. **90% interval no wider than 20 ECI points.**

The reason for each excluded model is in the `exclusion_reason` column.

## 5. Result

**Models with a capability score**

| Dataset | Official ECI | Estimated (included) | No score |
|---|---|---|---|
| Safetywashing chat | 8 | 6 | 12 |
| Safetywashing base | 15 | 4 | 8 |
| IRT for AI Safety | 68 | 6 | 118 |

**Estimated scores that are used**

| Dataset | Model | Method | Benchmarks | ECI (90% interval) |
|---|---|---|---|---|
| SW chat | Llama-2-7b-chat-hf | SW link | 12 SW | 91.4 (81.9–99.6) |
| SW chat | Mistral-7B-Instruct-v0.2 | SW link | 12 SW + 3 Epoch | 99.7 (93.7–106.4) |
| SW chat | Yi-34B-Chat | SW link | 12 SW + 5 Epoch | 107.5 (100.9–113.7) |
| SW chat | dbrx-instruct | SW link | 12 SW + 2 Epoch | 114.5 (110.9–118.1) |
| SW chat | deepseek-llm-67b-chat | SW link | 12 SW + 3 Epoch | 108.8 (101.9–114.7) |
| SW chat | deepseek-llm-7b-chat | SW link | 12 SW | 97.8 (90.0–104.6) |
| SW base | dbrx-base | SW link | 12 SW + 2 Epoch | 114.2 (110.3–118.3) |
| SW base | deepseek-llm-67b-base | SW link | 12 SW + 3 Epoch | 110.6 (106.4–114.9) |
| SW base | Yi-34B | SW link | 12 SW + 5 Epoch | 112.4 (108.5–116.6) |
| SW base | deepseek-llm-7b-base | SW link | 12 SW | 86.2 (76.1–95.1) |
| IRT | bytedance-seed/seed-1.6 | Projection | 4 Epoch | 141.2 (expected error ±3.1) |
| IRT | google/gemma-3-12b-it | Projection | 4 Epoch | 121.2 (±3.1) |
| IRT | google/gemma-3-4b-it | Projection | 4 Epoch | 113.2 (±3.1) |
| IRT | minimax/minimax-m2.5 | Projection | 6 Epoch | 146.9 (±1.9) |
| IRT | minimax/minimax-m2.7 | Projection | 4 Epoch | 146.1 (±3.1) |
| IRT | mistralai/mistral-small-3.1-24b-instruct | Projection | 4 Epoch | 125.4 (±3.1) |

## 6. Limitations

- **The estimates are not official ECI scores.** They come from Epoch's published model and parameters, not
  from Epoch's own fit. Label them as estimates in any figure, and keep a sensitivity check that uses
  official ECI only.
- **Epoch pools base and chat versions in one ECI.** The Safetywashing anchors inherit that pooled value
  even though Safetywashing evaluates chat and base models separately.
- **Pooling of reasoning modes.** Seed-1.6 pools Epoch's reasoning-mode variants by taking the max, as
  Epoch does within a model group. We don't know which mode the IRT data used.
- **Coverage.** 118 IRT models still have no capability score: Epoch has fewer than 4 benchmarks for them,
  or none at all. These are mostly smaller or less widely evaluated models. ECI covers most of the major
  frontier models in the IRT data.
