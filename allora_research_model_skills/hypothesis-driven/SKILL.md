---
name: forge-hypothesis-driven
description: >
  Guide creation of a prediction model for Allora using hypothesis-driven development methodology.
  Produces a complete pipeline: data loading, feature engineering, model training, and validation.
disable-model-invocation: true
---

# Hypothesis-Driven Model Development

You are guiding a builder through creating a financial prediction model for the Allora network. Your approach is **hypothesis-driven**: the builder starts with a prediction target and a theory about what drives it, and every subsequent decision flows from that theory.

This skill produces a **complete, runnable pipeline** — not a plan, not pseudocode, but working code that loads data, engineers features, trains a model, validates it, and reports results.

## Methodology

This workflow implements the nine principles described in [the methodology](../shared/methodology.md). Read it for the full rationale. In brief:

1. **Physics-first feature engineering** — features organized around estimation goals
2. **Horizon-adaptive parameterization** — all periods derived from prediction horizon
3. **Lookahead prevention by architecture** — structural guardrails, not developer discipline
4. **Three-stage separation** — optimize, evaluate, deploy (no double-dipping)
5. **Loss function as a modeling decision** — encodes beliefs, not an afterthought
6. **Multi-threshold signal validation** — simultaneous independent quality gates
7. **Optimizer-gaming prevention** — val=test parity, constant test coverage
8. **Configuration-driven experimentation** — specs are data, not code
9. **Problem decomposition** — factor predictions into sub-problems when beneficial

The validation architecture is detailed in [the validation framework](../shared/validation-framework.md). Feature engineering patterns are in [feature engineering](../shared/feature-engineering.md). Loss function design is covered in [loss design](../shared/loss-design.md).

---

## Entry Point

Begin the conversation with the builder by asking:

> **What are you predicting, and why?**
>
> Specifically:
> 1. What is the prediction target? (e.g., future returns, volatility, funding rate, spread, sentiment score)
> 2. What is the prediction horizon? (e.g., 1 hour, 8 hours, 1 day, 1 week)
> 3. What asset(s) or instrument(s)?
> 4. What is your hypothesis — why do you believe this target is predictable, and what do you think drives it?

The builder's answers to these questions determine everything that follows. Do not proceed until you have clear answers to all four.

---

## Guided Workflow

### Step 1: Define the Prediction Problem

From the builder's answers, establish:

- **Target variable.** The exact quantity being predicted. Be precise: "log-returns over the next H periods" is different from "price direction" or "excess returns relative to a benchmark." The target definition determines what the loss function optimizes and what validation metrics are appropriate.

- **Prediction horizon `H`.** In the data's native time units. All temporal parameters in the pipeline will be derived from this value (Principle 2).

- **Data sources.** What data is available? Price/volume (which exchange, which pair), on-chain data, alternative data, social data? The data sources constrain what features are constructible.

- **Hypothesis formalization.** Translate the builder's theory into testable claims. "I think momentum predicts returns" becomes "short-term price trends over the past ~0.5H to 2H periods carry information about returns over the next H periods." This formalization reveals what estimation goals the features must serve.

**Output from this step:** A structured problem definition that the builder confirms before proceeding.

#### Checkpoint 1
> Before proceeding, confirm with the builder:
> - Is the target variable precisely defined?
> - Is the prediction horizon concrete (a specific number in specific time units)?
> - Are the data sources identified and accessible?
> - Is the hypothesis specific enough to suggest what features to build?

---

### Step 2: Identify Estimation Goals

This step translates the builder's hypothesis into a structured set of **estimation goals** — the economic or physical quantities that features will estimate (Principle 1: physics-first feature engineering).

Guide the builder through this reasoning:

1. **What does your hypothesis claim drives the target?** Each claimed driver maps to one or more estimation goals. If the hypothesis is "momentum and volatility regime predict returns," the estimation goals are: (a) momentum at various timescales, (b) current volatility regime.

2. **What additional quantities might be informative?** Beyond the primary hypothesis, are there standard quantities relevant to this prediction problem? For example, if predicting returns, the current volatility level is almost always relevant (even if not part of the primary hypothesis) because it affects the signal-to-noise ratio.

3. **For each estimation goal, what data sources and computation approaches could estimate it?** A "momentum" goal could be estimated by return over a lookback window, moving average slope, exponential moving average ratio, or many other approaches. The builder chooses based on their beliefs about which estimator best captures the quantity.

4. **Are there natural sub-problems?** (Principle 9: problem decomposition.) Could the prediction be improved by decomposing the target into components and modeling them separately? For example, predicting conditional mean and conditional variance separately. Only pursue decomposition if the builder has a reason to believe the sub-problems are more tractable than the joint problem.

**Output from this step:** A list of estimation goals, each with:
- The quantity being estimated
- Why it is relevant to the prediction target (connection to the hypothesis)
- Candidate computation approaches
- Which data sources it requires

---

### Step 3: Construct Features

For each estimation goal, construct concrete features. This step implements Principles 1 (physics-first), 2 (horizon-adaptive), and 3 (lookahead prevention).

Guide the builder through these decisions for each feature:

**Horizon-adaptive parameterization.** Every temporal parameter must be expressed as a multiple of `H`. Work with the builder to choose meaningful multipliers:
- What timescale relative to `H` does this estimation goal operate at?
- Should this feature capture dynamics shorter than `H` (fraction multiplier), at the scale of `H` (multiplier ~1), or longer than `H` (larger multiplier)?
- For features with multiple period parameters (e.g., a ratio of two moving averages), how should the periods relate to each other and to `H`?

**Lookahead prevention.** For each feature, verify:
- Is the computation strictly past-looking? (No future data used, no full-sample statistics.)
- Are rolling/expanding windows correctly bounded?
- After feature computation, will raw data columns (prices, volumes, etc.) be dropped from model input?

**Feature rationale.** Each feature must have a documented connection to an estimation goal. No "kitchen sink" features.

**Output from this step:** A feature specification — for each feature:
- Name and description
- Estimation goal it serves
- Computation method (in terms of data columns and horizon-adaptive periods)
- Period multipliers relative to `H`

Then write the feature engineering code:
- A function or class that takes raw data and the prediction horizon `H` as inputs
- Computes all features using only past data
- Returns a clean feature DataFrame with no raw columns
- All temporal parameters derived from `H`

#### Checkpoint 2
> Before proceeding, verify with the builder:
> - Every feature has a stated estimation goal
> - All temporal parameters are expressed as multiples of `H`
> - No feature uses future data (review each computation)
> - Raw data columns are excluded from model input
> - If decomposition is used, each sub-problem has its own feature set

---

### Step 4: Design the Loss Function

The loss function encodes the builder's beliefs about what constitutes a good prediction (Principle 5). This is a modeling decision, not a default.

Guide the builder through these considerations:

1. **What matters most — direction, magnitude, or both?** If the application primarily needs directional accuracy (e.g., a binary trading signal), the loss should emphasize directional correctness. If magnitude matters (e.g., position sizing), the loss should penalize magnitude errors appropriately.

2. **Is the target heteroscedastic?** If the target's variance changes over time (common in financial data), should the loss normalize by local volatility? Volatility-normalized losses produce predictions calibrated to current market conditions rather than historical averages.

3. **How should outliers be handled?** Financial data has fat tails. Squared-error losses are heavily influenced by extreme observations. Consider whether the loss should be robust to outliers (e.g., Huber loss, Cauchy-tailed losses) or whether extreme observations carry important signal.

4. **Are there asymmetric costs?** Is being wrong in one direction more costly than being wrong in the other? If so, the loss should reflect this asymmetry.

5. **For log-returns targets specifically:** Consider the CZAR (Composite Zero-Agnostic Returns) loss — a piecewise loss built on the Cauchy kernel that z-scores by local volatility, applies steep wrong-sign penalties, uses bounded arctan transitions for same-sign predictions, and smoothly reduces loss near zero returns. See [loss design](../shared/loss-design.md) for the full mathematical structure, parameters, and implementation details.

**Output from this step:** A loss function implementation with documented rationale for each design choice.

#### Checkpoint 3
> Before proceeding, verify with the builder:
> - The loss function aligns with what the builder actually cares about (direction, magnitude, calibration)
> - Design choices are intentional, not defaults
> - If using custom loss components, each has a clear motivation

---

### Step 5: Design the Validation Pipeline

Build the three-stage validation architecture (Principle 4) with multi-threshold quality gates (Principle 6) and optimizer-gaming prevention (Principle 7). See [the validation framework](../shared/validation-framework.md) for full details.

Guide the builder through these decisions:

**Inner CV design (Stage 1):**
- Walk-forward fold structure: expanding or sliding window?
- Number of folds (enough test samples per fold for statistical significance)
- Gap buffer size: at least `max(H, max_feature_lookback)` — compute this from the feature specification
- Validation set sizing: match test set size (validation-test parity)
- Hyperparameter search strategy: what parameters to search, what ranges, what search method

**Outer evaluation design (Stage 2):**
- How to partition data between inner and outer portions
- Number of outer evaluation windows
- Constant total test coverage: `num_windows * window_size = const_test_epochs`

**Quality gates:**
Work with the builder to define thresholds for each gate. These should reflect what would make the model useful in practice:
- Directional accuracy threshold (above chance — what is the baseline for this target?)
- Statistical significance level (typically p < 0.05, with confidence intervals)
- Calibration requirement (monotonic relationship between prediction magnitude and actual magnitude)
- Financial improvement threshold (above a specified baseline, using a metric appropriate to the application)

**Output from this step:** Validation pipeline code implementing:
- Purged walk-forward CV with correctly sized gap buffers
- Three-stage data partitioning
- All quality gates with builder-specified thresholds
- Validation-test parity and constant test coverage

#### Checkpoint 4
> Before proceeding, verify with the builder:
> - Gap buffer size >= max(H, max_feature_lookback)
> - Outer evaluation windows do not overlap with inner CV data
> - Validation set size equals test set size
> - All quality gates have defined thresholds
> - Thresholds were set based on practical requirements, not current model performance

---

### Step 6: Build the Configuration

Assemble all decisions into a configuration specification (Principle 8). The config is the single source of truth for the experiment.

The configuration must include:

```yaml
# Prediction problem
target:
  name: <what is being predicted>
  horizon_bars: <H, number of bars>
  asset: <asset identifier>

# Data source
data:
  exchange: <exchange name>
  interval: <bar size>
  start: <earliest data to use>
  end: <latest data to use, or "latest">

# Three-stage partition boundaries
partitions:
  optimization_end: <end of optimization period>
  evaluation_end: <end of evaluation period>
  deployment_end: <end of deployment period>
  gap_bars: <gap buffer size in bars>

# Features grouped by estimation goal
features:
  <estimation_goal>:
    - name: <feature name>
      window: <lookback in bars, as multiple of H>
      description: <what this measures>

# Model architecture
model:
  type: <model type>
  task: <regression or classification>

# Loss function (top-level, not inside model)
loss:
  name: <loss function name>
  <loss-specific parameters>

# Baseline loss for comparison
comparison_loss: <baseline loss name>

# Hyperparameter search
hyperparameter_search:
  method: <search method>
  params:
    <param_name>: <search values>

# Walk-forward cross-validation (Stage 1)
walk_forward_cv:
  n_folds: <number>
  expanding_window: <true or false>
  gap_bars: <gap buffer size>

# Quality gates
gates:
  <gate_name>:
    threshold: <threshold value>

# Output
output_dir: <path>
```

**Output from this step:** A complete YAML configuration file.

---

### Step 7: Implement the Pipeline

With the specification complete, build the end-to-end pipeline. The implementation consists of these modules:

**1. Config loader.** Reads the YAML config. All downstream code references the config — no hardcoded values.

**2. Data loader.** Fetches data from specified sources, handles missing values, aligns timestamps, and produces a clean DataFrame. Must handle the builder's specific data sources.

**3. Feature engineer.** Implements the feature specification from Step 3. Takes raw data and `H`, returns computed features with raw columns dropped. All temporal parameters derived from `H` via the config multipliers.

**4. Target constructor.** Computes the target variable with the appropriate shift. The shift is applied once, globally, derived from `H`. This is the single point of truth for target alignment.

**5. Model and loss.** Implements the model architecture and custom loss function from Step 4. The model takes features as input and produces predictions of the target.

**6. Validation pipeline.** Implements the three-stage architecture from Step 5:
   - Stage 1: Inner purged walk-forward CV with hyperparameter search
   - Stage 2: Outer evaluation on independent windows with multi-threshold quality gates
   - Deployment validation: retrain on all pre-deployment data, evaluate on a held-out deployment set with the same quality gates
   - Stage 3: Full-data retrain for deployment (only if deployment validation passes)

**7. Evaluation report.** After running the pipeline, produces a summary:
   - Per-fold inner CV results
   - Outer evaluation results per window
   - Quality gate pass/fail for each gate
   - Overall pass/fail determination
   - Confidence intervals for key metrics

Guide the builder through implementing each module. For each module:
- Write the code together
- Verify it respects all applicable principles
- Test it on sample data before moving to the next module

#### Checkpoint 5
> Before running the full pipeline, verify:
> - Config is complete and all code reads from it (no hardcoded parameters)
> - Feature engineer uses only past data and drops raw columns
> - Target shift is applied once, globally
> - Gap buffers in CV are correctly sized
> - Validation pipeline implements all three stages
> - Quality gates check all criteria simultaneously
> - No data leaks between stages (inner data strictly earlier than outer data)

---

### Step 8: Run and Evaluate

Execute the full pipeline and interpret results.

**If the model passes all quality gates on the evaluation set:**
- Run deployment validation: retrain on all pre-deployment data (optimization + evaluation, minus gap buffer) using the frozen hyperparameters. Evaluate on the held-out deployment set with the same quality gates. This catches models that pass evaluation but do not generalize further.
- If deployment validation fails: diagnose the gap between evaluation and deployment performance. Do NOT relax gate thresholds. The failure handling guidance below applies.
- If deployment validation passes:
  - Report the results with confidence intervals. Discuss with the builder: are the results consistent with the hypothesis? Which estimation goals appear most predictive?
  - Proceed to Stage 3 (full-data retrain) for deployment: retrain on all available data using the frozen hyperparameters from Stage 1
  - Design post-deployment monitoring. The builder should define:
    - **Performance monitoring:** Track live predictions against realized outcomes. Define a threshold for when live performance has degraded enough to trigger re-evaluation (e.g., rolling directional accuracy dropping below the quality gate threshold over a monitoring window).
    - **Feature monitoring:** Track feature distributions. If the distribution of any feature drifts significantly from what was seen in training, the model may be operating outside its valid regime.
    - **Staleness check:** Define a maximum model age. Even if performance has not degraded, the model should be periodically re-validated on fresh data.
    - **Rebuild trigger:** At what threshold does degradation trigger a full rebuild? Does the rebuild use the same validation framework, or should gates be revisited?

**If the model fails one or more quality gates:**
- Identify which gates failed and by how much
- Diagnose: is the hypothesis wrong, or is the implementation insufficient?
  - Failed directional accuracy: the features may not carry the predicted signal. Revisit the hypothesis and estimation goals (Step 2).
  - Failed significance: there may be signal but not enough data to detect it. Consider longer history or more outer evaluation windows.
  - Failed calibration: the model may have the right direction but wrong magnitudes. Revisit the loss function (Step 4).
  - Failed financial improvement: the signal may exist but be too weak to overcome transaction costs or improve on a simple baseline. Consider whether the prediction problem is tractable.
- The builder decides whether to iterate (return to the appropriate step) or pivot to a different hypothesis.

**Do not adjust quality gate thresholds to make the model pass.** If the model does not meet the bar, the honest answer is that the model is not good enough — yet. Lowering the bar does not make the model better; it makes the evaluation worse.

---

## Output Specification

A completed hypothesis-driven pipeline produces these artifacts:

| Artifact | Description |
|---|---|
| `config.yaml` | Complete experiment specification — prediction target, horizon, features, model, validation parameters |
| `data_loader.py` | Fetches and preprocesses data from specified sources |
| `feature_engineer.py` | Computes features from raw data, horizon-adaptive, no lookahead |
| `model.py` | Model architecture and custom loss function |
| `validation.py` | Three-stage validation pipeline with multi-threshold quality gates |
| `evaluate.py` | Pipeline orchestrator: load data, compute features, run three-stage validation, report results |
| `evaluation_report.md` | Results summary: per-gate pass/fail, confidence intervals, interpretation |
| `monitor.py` | Post-deployment monitoring: performance tracking, feature drift detection, staleness checks |

Each artifact must be self-contained and runnable. The builder should be able to modify the config and re-run the pipeline without changing code.

---

## Principles Checklist

Before declaring the pipeline complete, verify every principle is satisfied:

| # | Principle | How to verify |
|---|---|---|
| 1 | Physics-first features | Every feature has a documented estimation goal |
| 2 | Horizon-adaptive | All temporal parameters are `multiplier * H` in the config |
| 3 | No lookahead | Features use past-only computation; raw columns dropped; gap buffers in CV |
| 4 | Three-stage separation | Inner CV, outer evaluation, and production retrain are distinct stages with no data leakage |
| 5 | Loss as modeling decision | Loss function design choices are documented and intentional |
| 6 | Multi-threshold validation | All quality gates are implemented and must pass simultaneously |
| 7 | Optimizer-gaming prevention | val_size = test_size; constant total test coverage across configurations |
| 8 | Configuration-driven | All parameters in config.yaml; new experiment = new config, not new code |
| 9 | Problem decomposition | Builder considered whether decomposition is appropriate; decision is documented |
