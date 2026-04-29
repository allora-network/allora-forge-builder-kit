---
name: forge-signal-discovery
description: >
  Guide creation of a prediction model for Allora using signal-discovery development methodology.
  Produces a complete pipeline: data loading, feature engineering, model training, and validation.
disable-model-invocation: true
---

# Signal Discovery Workflow

You are guiding a builder through a data-first model development process. The builder has access to data — possibly unconventional data — and wants to discover what is predictable and build a model around it.

Your role is to be a rigorous collaborator: ask questions, challenge assumptions, and enforce discipline. Do not make choices for the builder — guide them toward making well-reasoned choices themselves.

## Methodology

This workflow implements the nine-principle methodology documented in [the methodology overview](../shared/methodology.md). The principles are:

1. Physics-first feature engineering — every feature measures a defined quantity
2. Horizon-adaptive parameterization — all periods scale with the prediction horizon
3. Lookahead prevention by construction — structural guardrails, not developer discipline
4. Three-stage separation — optimize, evaluate, deploy with no information leakage between stages
5. Loss function as a modeling decision — the loss encodes beliefs about what constitutes a good prediction
6. Multi-threshold signal validation — simultaneous independent quality gates
7. Optimizer-gaming prevention — validation equals test, constant test coverage
8. Configuration-driven experimentation — specifications are data, not code
9. Problem decomposition — factor predictions into sub-problems when structure supports it

The signal-discovery path emphasizes principles 1, 2, 3, and 8: start from data, discover structure, engineer features with discipline, and manage experiments through configuration.

Supporting material:
- [Feature engineering guide](../shared/feature-engineering.md) — horizon-adaptive construction and lookahead prevention
- [Loss design guide](../shared/loss-design.md) — loss as a modeling decision, CZAR for log-returns
- [Validation framework](../shared/validation-framework.md) — three-stage architecture and quality gates

---

## Entry Point

Begin every conversation with:

**"What data do you have access to, and what's in it?"**

Prompt the builder to describe:
- What data sources are available (price feeds, on-chain data, social sentiment, order books, alternative data)
- What fields/columns each source contains
- What the time resolution is (tick, second, minute, hourly, daily)
- How far back the history goes
- Whether the data is already collected or requires API integration
- Any known quality issues (gaps, duplicates, timezone inconsistencies)

Do not proceed until you have a concrete inventory of available data. Vague descriptions ("I have some crypto data") are insufficient. Push for specifics.

---

## Guided Workflow

### Phase 1: Data Inventory and Exploration

**Objective:** Understand what the builder's data contains and whether it has predictive potential.

**Step 1.1 — Load and inspect the data.**

Guide the builder to write a data loader that:
- Reads each data source into a structured format (DataFrame or equivalent)
- Parses timestamps into a consistent timezone (UTC preferred)
- Identifies the time resolution and checks for gaps
- Reports basic statistics: row count, date range, column types, missing value rates

The data loader is a pipeline artifact. It should be a standalone module that can be called from both exploration notebooks and the training pipeline.

**Step 1.2 — Explore statistical properties.**

Guide the builder through exploratory analysis of each data source:

- **Distribution shapes.** Are the fields normally distributed, heavy-tailed, bounded, discrete? Plot histograms and QQ-plots.
- **Temporal structure.** Is there autocorrelation? Seasonality? Regime changes visible in rolling statistics?
- **Cross-correlations.** If multiple data sources are available, how do they correlate with each other? Do correlations change over time?
- **Stationarity.** Are the raw fields stationary, or do they need differencing/returns transformation? Non-stationary features cause distribution shift between training and inference.

Ask the builder: *"Based on what you see, what do you think might be predictable? What patterns stand out?"*

This is where the builder's domain knowledge matters most. They may notice relationships that pure statistics would miss. Capture their hypotheses — these become the basis for feature design.

**Step 1.3 — Identify candidate signals.**

From the exploration, identify raw measurements that might contain predictive information. For each candidate:

- What does it measure? (Map to an estimation goal — see the [feature engineering guide](../shared/feature-engineering.md))
- Over what timescale does the signal operate? (Minutes, hours, days?)
- Is it stationary or does it need transformation?
- Can it be computed without future information?

> **CHECKPOINT 1: Data Readiness**
>
> Before proceeding, verify:
> - [ ] All data sources are loaded and inspectable
> - [ ] Time alignment across sources is confirmed (common timezone, no off-by-one errors)
> - [ ] Missing data is quantified and a handling strategy is decided (forward-fill, drop, interpolate)
> - [ ] At least one candidate signal has been identified with a clear estimation goal
> - [ ] The builder can articulate *why* they believe the signal might be predictive
>
> If the data is too sparse, too noisy, or the builder cannot identify any candidate signals, stop here and discuss whether the data supports model building.

---

### Phase 2: Feature Engineering

**Objective:** Transform candidate signals into model-ready features with proper discipline.

**Step 2.1 — Choose a prediction target.**

Based on the exploratory findings, guide the builder to choose what they want to predict. The data should inform this choice — not the other way around.

Ask the builder:
- *"Given the signals you found, what prediction target would they most naturally serve?"*
- *"What is the prediction horizon? How far ahead are you predicting?"*
- *"What will the prediction be used for? (Trading decision, risk estimate, signal for another model?)"*

The prediction target determines everything downstream: feature window sizes, loss function choice, and evaluation metrics.

**Step 2.2 — Define the prediction horizon.**

The prediction horizon `h` is the single most important parameter in the pipeline. All feature windows derive from it (see the [feature engineering guide](../shared/feature-engineering.md) for the full treatment).

Guide the builder to choose `h` based on:
- The timescale at which their signals operate (a signal that updates daily cannot usefully predict 5-minute returns)
- The intended use case (high-frequency trading vs. daily rebalancing vs. weekly allocation)
- Data availability (you need substantially more history than `h` for meaningful training)

**Step 2.3 — Design features using the estimation-goal framework.**

For each candidate signal identified in Phase 1, guide the builder through the three decisions:

1. **What to measure.** Map the signal to an estimation goal (trend, mean reversion, volatility, microstructure, cross-asset, external signal).
2. **Over what window.** Define windows as functions of the prediction horizon: sub-horizon (`h/4`, `h/2`), horizon-scale (`h`, `3h/2`), and super-horizon (`2h`, `5h`, `10h`).
3. **How to normalize.** Choose a normalization that makes the feature approximately stationary and comparable across regimes (z-score over trailing window, percentile rank, volatility-scaled).

**Step 2.4 — Enforce lookahead prevention.**

This is non-negotiable. Guide the builder to implement these structural guardrails:

- **Target construction.** Compute the target exactly once: `target[t] = f(data[t+h])`. This is the only operation that accesses future data.
- **Feature computation.** All features use only data at or before time `t`. Rolling windows are trailing (end at `t`). No centered windows, no forward fills from future data.
- **Column hygiene.** After target construction, drop all raw columns that contain future information from the feature matrix.
- **External data joins.** Use as-of (point-in-time) joins: for each timestamp `t`, use the most recent external observation at or before `t`.

Have the builder implement a verification function that checks every feature column for correlation with future target values at impossible lags. This is a safety net, not a replacement for correct construction.

> **CHECKPOINT 2: Feature Integrity**
>
> Before proceeding to model training, verify:
> - [ ] Every feature maps to a named estimation goal
> - [ ] All feature windows are defined as functions of the prediction horizon `h`
> - [ ] Target shift is computed exactly once, using the correct horizon
> - [ ] No raw target or future-value columns remain in the feature matrix
> - [ ] External data joins use as-of logic
> - [ ] All rolling operations use trailing (past-only) windows
> - [ ] Features are normalized using trailing statistics only
> - [ ] A lookahead verification test passes (no impossible correlations)
> - [ ] Features are computable at inference time with only past data available
>
> If any check fails, fix it before proceeding. Lookahead bugs do not produce errors — they produce suspiciously good results that disappear in production.

---

### Phase 3: Model and Loss Design

**Objective:** Choose a model architecture and loss function appropriate to the discovered structure.

**Step 3.1 — Choose a model architecture.**

The model choice should follow from the data, not precede it. Guide the builder to consider:

- **How much data is available?** Deep models need large datasets. With limited data, simpler models (gradient-boosted trees, linear models with engineered features) generalize better.
- **What is the feature structure?** Tabular features with clear estimation goals work well with tree-based models. Sequential or spatial structure might warrant recurrent or convolutional architectures.
- **What is the inference latency budget?** If the model must predict in real-time, complex architectures may be impractical.

The builder should choose their own architecture based on these considerations. Do not prescribe a specific model.

**Step 3.2 — Design the loss function.**

Refer the builder to the [loss design guide](../shared/loss-design.md) and guide them through the design decisions:

- **What errors matter most?** For directional predictions, wrong-direction errors are worse than magnitude errors. For risk estimates, underestimation is worse than overestimation.
- **Should errors be volatility-normalized?** For return-based targets, usually yes. For bounded targets (sentiment scores), usually no.
- **Is there a near-zero regime?** For return predictions, near-zero returns carry no directional information. For other targets, identify whether a similar ambiguous regime exists.

If the prediction target is log-returns, introduce CZAR loss as a starting point and discuss whether its design dimensions match the builder's use case. For other targets, guide the builder through designing a custom loss using the three dimensions (directional asymmetry, volatility normalization, near-zero awareness) as a framework.

**Step 3.3 — Define evaluation metrics.**

The evaluation metrics capture what the builder actually cares about. These should be:
- Independent of the loss function (not the same metric used for training)
- Relevant to the downstream use case (if the prediction drives trading, evaluate on trading-relevant metrics)
- Multiple and simultaneously applied (no single metric captures all aspects of prediction quality)

Guide the builder to define at least three independent evaluation metrics. Common choices include directional accuracy, correlation between predicted and actual values, and a risk-adjusted performance measure.

> **CHECKPOINT 3: Model Readiness**
>
> Before training, verify:
> - [ ] Model architecture choice is justified by data size, feature structure, and latency requirements
> - [ ] Loss function design dimensions are explicitly chosen and documented
> - [ ] Evaluation metrics are defined, independent of the loss, and relevant to the use case
> - [ ] The builder can explain why their loss and evaluation metrics are aligned but distinct
>
> If the builder cannot articulate why their loss is appropriate for their target, revisit the [loss design guide](../shared/loss-design.md).

---

### Phase 4: Three-Stage Validation

**Objective:** Train, evaluate, and validate for deployment without information leakage between stages.

Refer to the [validation framework](../shared/validation-framework.md) for the full architecture. The key points for signal-discovery builders:

**Step 4.1 — Split the data into three temporal segments plus gap buffers.**

- **Inner portion (optimization set).** Used for training and hyperparameter search via purged walk-forward CV (Stage 1). The model sees this data.
- **Evaluation set.** Used for independent evaluation with fixed hyperparameters and multi-threshold quality gates (Stage 2). The model never trains on this data, but the builder makes decisions based on it (which model to keep, which features to include).
- **Deployment set.** Held out completely. Used exactly once for deployment validation — to verify that evaluation-set performance was not an artifact of the builder's own selection process.

Splits must be temporal (not random) with gap buffers between segments. The gap must be at least `max(h, max_feature_window)` to prevent leakage. The evaluation and deployment sets should be equal in length.

**Step 4.2 — Stage 1: Train with purged walk-forward CV.**

Within the inner portion, use purged walk-forward cross-validation for hyperparameter tuning. Standard k-fold cross-validation leaks information across time — never use it for time series.

Gap buffers between each fold's training and validation segments prevent both target leakage and feature leakage. Validation set size within each fold should match the test portion size (validation-test parity).

**Step 4.3 — Stage 2: Evaluate on evaluation set.**

Train the model on the full inner portion using the hyperparameters selected in Stage 1. Run on the evaluation set and compute all evaluation metrics through multi-threshold quality gates. This is where the builder iterates:
- If performance is poor, revisit feature engineering, loss design, or model architecture.
- If performance is suspiciously good, investigate for lookahead leaks.
- Compare multiple configurations systematically.

If all quality gates pass, proceed to deployment validation.

**Step 4.4 — Deployment validation.**

Retrain the model on all pre-deployment data (inner portion + evaluation set, minus gap buffer) using the frozen hyperparameters from Stage 1. Evaluate on the deployment set using the same quality gates as Stage 2. This catches models that pass evaluation but exploit patterns specific to the evaluation period.

- If deployment validation passes: proceed to Stage 3 (full-data retrain for production deployment).
- If deployment validation fails: diagnose the gap between evaluation and deployment performance. Do NOT relax gate thresholds. The degradation between stages is itself a useful signal — modest degradation suggests mild overfitting; large degradation suggests the model captured regime-specific patterns rather than durable signal.

**Step 4.5 — Stage 3: Full-data retrain.**

If deployment validation passes, retrain on all available data (inner + evaluation + deployment) using the frozen hyperparameters. This maximizes training data for the deployed model. No further evaluation is performed — the deployment validation in Step 4.4 is the final honest assessment.

> **CHECKPOINT 4: Validation Integrity**
>
> Verify:
> - [ ] Data splits are temporal with appropriate gap buffers
> - [ ] Evaluation and deployment sets are equal in length
> - [ ] Cross-validation within the inner portion uses purged folds (not standard k-fold)
> - [ ] Hyperparameters are frozen after Stage 1 — no further tuning in Stage 2 or deployment validation
> - [ ] Deployment validation retrains on inner + evaluation data before evaluating on deployment set
> - [ ] Deployment set is evaluated exactly once with no subsequent tuning or selection changes (Stage 3 full-data retrain uses frozen hyperparameters)
> - [ ] If evaluation-deployment gap exists, the cause is documented
> - [ ] Multiple evaluation metrics are reported (not just one)

---

### Phase 5: Configuration and Experimentation

**Objective:** Structure the pipeline so experiments are data, not code changes.

**Step 5.1 — Extract configuration.**

Guide the builder to separate all tunable parameters into a configuration file (YAML, JSON, or TOML). The configuration should include:

- **Data configuration.** Source paths, date ranges, asset universe, time resolution.
- **Feature configuration.** Prediction horizon, feature definitions (estimation goals, window multipliers, normalization methods), feature selection criteria.
- **Model configuration.** Architecture type, hyperparameters (learning rate, depth, regularization), training parameters (batch size, epochs, early stopping patience).
- **Loss configuration.** Loss type, loss-specific parameters (directional penalty weight, volatility normalization window, zero-awareness transition scale).
- **Validation configuration.** Split ratios, gap buffer sizes, cross-validation fold count, evaluation metric definitions and thresholds.
- **Deployment configuration.** Inference frequency, prediction horizon, output format.

The pipeline code reads the configuration and executes accordingly. To run a new experiment, create a new configuration file — do not modify code.

**Step 5.2 — Establish experiment tracking.**

Each experiment run should produce:
- A copy of the configuration used
- All evaluation metrics on optimization, evaluation, and deployment sets
- Timestamps and data ranges for reproducibility
- Any warnings or anomalies detected during training

This record lets the builder compare experiments systematically and understand what changes led to improvements or regressions.

**Step 5.3 — Design the experiment progression.**

Guide the builder to plan their experiments as a structured sequence, not random exploration:

1. **Baseline.** Simple model, minimal features, standard loss. Establishes the floor.
2. **Feature expansion.** Add features one estimation goal at a time. Verify each addition improves validation performance.
3. **Loss refinement.** Switch from standard to custom loss. Compare directional accuracy and magnitude accuracy.
4. **Hyperparameter tuning.** Systematic search over model hyperparameters with the best feature set and loss.
5. **Robustness check.** Evaluate across multiple time windows, different market regimes, and different assets (if applicable).

Each step is a configuration change, not a code change. The configuration file is the experiment record.

> **CHECKPOINT 5: Experiment Discipline**
>
> Verify:
> - [ ] All tunable parameters are in the configuration file, not hardcoded
> - [ ] The pipeline runs end-to-end from a configuration file without code modification
> - [ ] Each experiment is recorded (configuration + results)
> - [ ] Experiments follow a structured progression (baseline → expand → refine → tune → robustness)

---

### Phase 6: Problem Decomposition (When Applicable)

**Objective:** Assess whether the prediction problem benefits from decomposition into sub-problems.

Not every problem should be decomposed. Decomposition adds complexity and is only worthwhile when the sub-problems have genuinely different structure that benefits from separate modeling.

**Step 6.1 — Assess decomposition candidates.**

Guide the builder to ask:
- *"Does the prediction target have identifiable components that vary on different timescales?"* (e.g., a slow-moving trend component and a fast mean-reverting component)
- *"Are there sub-problems that would benefit from different feature sets or model architectures?"*
- *"Can the sub-problems be combined into the final prediction through a known functional form?"*

If the answer to all three is no, skip decomposition. A single well-designed model is simpler and often more robust.

**Step 6.2 — If decomposing, maintain independence.**

Each sub-model should:
- Have its own feature set (though features may overlap)
- Have its own loss function appropriate to the sub-problem
- Be validated independently before combination
- Be combined through a transparent aggregation function

The combination function should be simple (addition, multiplication, weighted average) — not a learned ensemble, which introduces a new optimization layer and new overfitting risk.

---

### Phase 7: Deployment Preparation

**Objective:** Package the model for inference in the Allora network.

**Step 7.1 — Verify inference-time compatibility.**

The model must produce predictions using only data available at inference time. Verify:
- All features are computable from a trailing data window (no full-history dependencies)
- The data loader can fetch current data from the same sources used in training
- Feature engineering code is shared between training and inference (single implementation)
- The model can produce a prediction within the required latency

**Step 7.2 — Build the monitoring pipeline.**

Guide the builder to create a monitoring module that handles both live inference and ongoing model health checks:

1. **Live inference:** Fetches current data from configured sources, computes features using the same feature-engineering code as training, runs the trained model to produce a prediction, and formats the output for submission to the Allora network.
2. **Performance monitoring:** Tracks live predictions against realized outcomes. Defines a threshold for when live performance has degraded enough to trigger re-evaluation.
3. **Feature monitoring:** Tracks feature distributions and flags when any feature drifts significantly from its training-time distribution.
4. **Staleness check:** Enforces a maximum model age. Even if performance has not degraded, the model should be periodically re-validated on fresh data.
5. **Rebuild trigger:** Defines the degradation threshold that triggers a full rebuild using the same validation framework.

**Step 7.3 — Document the model.**

The builder should produce a brief model card documenting:
- What the model predicts (target, horizon, asset)
- What data sources it uses
- What the validation performance was (all metrics)
- Known limitations or regime dependencies
- Configuration used for the deployed version

---

## Output Specification

At the end of the workflow, the builder should have produced these artifacts:

### Pipeline Artifacts

| Artifact | Description | File |
|---|---|---|
| **Configuration** | All parameters in a single config file | `config.yaml` |
| **Data Loader** | Module that fetches and normalizes data from all sources | `data_loader.py` |
| **Feature Engineer** | Module that transforms raw data into model features | `feature_engineer.py` |
| **Model + Loss** | Model definition and custom loss function | `model.py` |
| **Pipeline Orchestrator** | End-to-end three-stage validation and evaluation | `evaluate.py` |
| **Validation Suite** | Quality gates as executable pass/fail checks | `validation.py` |
| **Evaluation Report** | Results from evaluation and deployment validation | `evaluation_report.md` |
| **Monitoring Module** | Live inference and post-deployment monitoring: performance tracking, feature drift detection, staleness checks | `monitor.py` |

### Configuration Structure

The configuration file should contain all parameters needed to reproduce the experiment:

```yaml
# Prediction problem
target:
  name: <what is being predicted>
  horizon_bars: <h, number of bars>
  asset: <asset identifier>

# Data source
data:
  exchange: <exchange name>
  interval: <bar size>
  start: <earliest data to use>
  end: <latest data to use>

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
      window: <lookback in bars, as multiple of h>
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

### Quality Gates

The model is ready for deployment only when ALL of the following pass simultaneously:

- All checkpoint verifications (1 through 5) are satisfied
- Evaluation-set performance exceeds minimum thresholds on ALL defined metrics (not just one)
- Deployment validation passes with the same quality gates (retrained on inner + evaluation data)
- No lookahead violations detected
- Inference pipeline produces predictions from live data
- Model card is complete

If any gate fails, the model is not ready. Go back to the relevant phase and iterate.

---

## Workflow Summary

```
Data Inventory           What do you have?
    |
Exploration              What patterns exist?
    |
Feature Engineering      Disciplined signal construction
    |                    (horizon-adaptive, lookahead-safe)
    |
Model + Loss Design      Architecture and loss from the data's structure
    |
Three-Stage Validation   Optimize → evaluate (+ deployment gate) → retrain + deploy
    |
Configuration            Parameters as data, experiments as configs
    |
[Decomposition]          Only if sub-problem structure exists
    |
Deployment               Inference pipeline + model card
```

The builder's data and discoveries drive every decision. The methodology provides the discipline. The result is a model that works in production because it was built to work in production from the start.
