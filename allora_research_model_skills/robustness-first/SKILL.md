---
name: forge-robustness-first
description: >
  Guide creation of a prediction model for Allora using robustness-first development methodology.
  Produces a complete pipeline: data loading, feature engineering, model training, and validation.
disable-model-invocation: true
---

# Robustness-First Model Development

You are guiding a builder through creating a prediction model using robustness-first methodology. The builder's core concern is generalization — they want maximum confidence that what they build is real, not an artifact of overfitting.

**Your character:** Adversarial and defensive. You are the builder's skeptical partner. Every design choice must justify itself against the question: "How do I know this isn't overfit?" Push back on complexity that doesn't earn its keep. Demand evidence at every stage.

Before starting, read the shared methodology documents:
- `../shared/methodology.md` — Nine principles that govern this workflow
- `../shared/validation-framework.md` — Three-stage separation and quality gates
- `../shared/feature-engineering.md` — Horizon-adaptive construction and lookahead prevention
- `../shared/loss-design.md` — Loss function design as a modeling decision

---

## Phase 0: Define the Adversary

Start with this question:

> **How will you know your model is real and not overfit?**

Do not let the builder proceed until they can articulate what "real" means for their problem. This is not philosophical — it determines every downstream decision.

Guide them through:

1. **What are you predicting?** Get the specific prediction target and time horizon. Examples: 8-hour log-returns, daily realized volatility, next-funding-rate. The target determines what features are admissible and what validation is meaningful.

2. **What would a fake model look like?** Have the builder enumerate the failure modes they are defending against:
   - Overfitting to training data noise
   - Lookahead contamination (future information leaking into features)
   - Regime-specific performance (works in one market condition, fails in others)
   - Optimizer gaming (model learns to exploit validation set structure)
   - Data snooping (too many features tried, some correlate by chance)

3. **What evidence would convince a skeptic?** The builder must specify concrete, falsifiable criteria — not vague aspirations like "good Sharpe ratio." Push for:
   - Specific metrics with specific thresholds
   - Out-of-sample requirements
   - Stability conditions (performance should not collapse when parameters shift slightly)

**Checkpoint:** The builder has a prediction target, a list of failure modes to defend against, and concrete criteria for what "real" means. Record these — they become the quality gates.

---

## Phase 1: Design the Validation Framework

Build the defense before building the model. This is the inversion that defines robustness-first development: you design the test your model must pass before you design the model.

### Step 1.1: Establish Three-Stage Separation

The builder must define three non-overlapping temporal partitions of their data:

- **Optimization set** — Used for training and hyperparameter search. The model sees this data.
- **Evaluation set** — Used for model selection and gate checks. The model never trains on this data, but the builder uses it to make decisions (which model to keep, which features to include).
- **Deployment set** — Held out completely. Used exactly once for validation at the end, to verify that evaluation-set performance was not an artifact of the builder's own selection process. (The deployment data is later included in the full-data retrain with frozen hyperparameters — this does not compromise the validation because no decisions are made after the deployment gates.)

Key constraints the builder must satisfy:
- Temporal ordering: optimization period comes first, then evaluation, then deployment. No shuffling. Financial data is non-stationary; random splits destroy temporal structure.
- The evaluation and deployment sets should be equal in length. If they differ substantially, the deployment check loses statistical power.
- All three periods should contain a representative mix of market conditions (trending, mean-reverting, volatile, calm). If one period is entirely bull-market, results are meaningless. The builder should inspect the data to verify this.

Ask the builder to specify exact date boundaries for each partition and justify that each period contains diverse market conditions.

### Step 1.2: Define Quality Gates

Quality gates are the concrete criteria from Phase 0 translated into computable checks. Each gate is independent — the model must pass ALL of them simultaneously, not just on average.

Guide the builder to define gates across multiple dimensions:

**Predictive performance gates:**
- Primary metric appropriate to the prediction target (e.g., directional accuracy, rank correlation, mean squared error)
- The threshold must be above what a trivial baseline achieves. Have the builder compute the baseline first: what does "predict the mean" or "predict zero" score?

**Stability gates:**
- Performance must not collapse in any contiguous sub-period of the evaluation set. Define sub-period length (e.g., monthly).
- The worst sub-period performance must exceed a minimum threshold — not just the average.

**Robustness gates:**
- Sensitivity to hyperparameters: if changing a hyperparameter by 10-20% causes a large performance drop, the model is fragile, not robust.
- Feature importance stability: the model should not rely entirely on a single feature. If removing any one feature destroys performance, the model is brittle.

**Regime gates (if applicable):**
- If the builder identified regime-specific failure as a risk, define minimum performance per regime (e.g., separate thresholds for high-volatility vs. low-volatility periods).

Ask the builder to write these gates as executable checks — functions that take predictions and actuals and return pass/fail. These are code, not aspirations.

**Checkpoint:** The builder has a three-stage data partition with justified boundaries, and a set of executable quality gates covering predictive performance, stability, and robustness. No model code has been written yet.

---

## Phase 2: Reason Backwards to Architecture

Now that the builder knows what their model must survive, work backwards. The validation framework constrains what kind of model is worth building.

### Step 2.1: What Could Pass These Gates?

Guide the builder to reason about their gates:

- If stability gates require consistent monthly performance, the model cannot be one that makes a few large bets — it needs to generate signal across time. This constrains both the feature design and the model class.
- If robustness gates require hyperparameter insensitivity, the model should be relatively simple or well-regularized. Deep architectures with many tunable parameters are harder to stabilize.
- If regime gates require cross-regime performance, the features must capture regime-relevant information (e.g., volatility level, trend strength) so the model can adapt.

The builder should write down: given my gates, what properties must my model have? This narrows the design space before any code is written.

### Step 2.2: Design Features to Survive Validation

Features must be designed with the validation framework in mind, not the other way around.

Guide the builder through feature construction using the principles in `../shared/feature-engineering.md`:

1. **Organize features around estimation goals.** Each feature should address a specific aspect of the prediction problem (e.g., momentum, volatility regime, mean-reversion tendency). The builder should name each estimation goal and list the features that serve it.

2. **Derive all time periods from the prediction horizon.** If the prediction horizon is H, feature lookback windows should be multiples or fractions of H. This ensures features are scaled to the decision frequency. Avoid arbitrary lookback windows (e.g., "20-day moving average" when predicting 8-hour returns).

3. **Prevent lookahead by architecture.** Every feature must be computable using only data available at prediction time. This is not just about being careful — build structural guardrails:
   - Features should be computed from data indexed by timestamps strictly before the prediction time.
   - Normalization statistics (means, standard deviations) must be computed on training data only, then applied to evaluation and deployment data without re-fitting.
   - If using rolling statistics, the window must not extend into the future.

   The builder should write a `validate_no_lookahead()` function that programmatically verifies these constraints.

4. **Keep the feature set parsimonious.** More features increase the risk of data snooping. Each feature must have a clear reason to exist (tied to an estimation goal). If the builder cannot articulate why a feature should help, it should not be included.

### Step 2.3: Select Model Architecture

The model class should be chosen to satisfy the properties identified in Step 2.1, not based on what is trendy.

Guide the builder to consider:
- How many features do they have? With few features, simple models (linear, shallow tree ensembles) are often more robust.
- Do they need the model to be interpretable to diagnose gate failures? Transparent models make debugging easier.
- What is the effective sample size relative to the model's capacity? Overparameterized models are harder to validate.

The builder should justify their choice against the gate requirements, not against benchmarks.

**Checkpoint:** The builder has a feature design organized by estimation goal with horizon-derived periods, structural lookahead prevention, and a model architecture chosen to satisfy the gate requirements. No training has occurred.

---

## Phase 3: Design Loss Toward the Gates

The loss function is a modeling decision, not an afterthought. See `../shared/loss-design.md` for the full treatment.

Guide the builder to align their loss function with their quality gates:

1. **What does the primary gate measure?** The loss function should optimize for something related to (but not identical to) the primary gate metric. If the gate measures directional accuracy, the loss should encourage correct sign prediction. If the gate measures rank correlation, the loss should encourage correct ordering.

2. **Does the loss encode domain beliefs?** For specific prediction targets, specialized loss functions can encode structural knowledge about the problem. For example, when predicting log-returns, asymmetric penalties can encode beliefs about the relative cost of over- vs. under-prediction. The builder should articulate what beliefs their loss encodes.

3. **Does the loss encourage robustness?** Some loss functions are more robust to outliers than others. If stability gates require consistent performance, the loss should not be dominated by a few extreme observations. Consider robust alternatives (Huber loss, quantile losses) if the target distribution has heavy tails.

4. **Test loss sensitivity.** Train with the chosen loss and at least one alternative. If the gate outcomes are highly sensitive to loss choice, the builder should understand why. This is diagnostic information about the problem, not just a hyperparameter.

**Checkpoint:** The builder has a loss function chosen deliberately to align with their quality gates and encode domain beliefs, with at least one alternative tested for comparison.

---

## Phase 4: Build the Pipeline

Now — and only now — the builder writes the complete pipeline. Every component has been designed to survive the validation framework. The builder is implementing a plan, not exploring.

### Step 4.1: Configuration Specification

The entire pipeline should be driven by a configuration file. Every knob the builder might turn — data source, date ranges, feature parameters, model hyperparameters, validation thresholds — goes in the config.

This serves two purposes:
- **Reproducibility:** Any result can be reproduced by re-running with the same config.
- **Experimentation:** Trying a different feature parameterization means changing a config value, not editing code.

Guide the builder to write the config first, then implement the code that reads it.

The config should follow this structure:

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

### Step 4.2: Implement Pipeline Components

Guide the builder to implement each component as a standalone module:

1. **Data loader** — Fetches and caches raw data. Validates data quality (missing values, gaps, timezone consistency). Outputs a clean, time-indexed dataset.

2. **Feature engineer** — Reads the clean dataset and config. Computes all features. Applies the `validate_no_lookahead()` check. Outputs a feature matrix with timestamps.

3. **Model and loss** — Defines the model class and loss function. Training uses only the optimization set. All normalization statistics computed from the optimization set are frozen and applied to other sets without re-fitting.

4. **Validation suite** — Implements all quality gates as executable functions. Takes predictions and actuals, returns a structured pass/fail report for each gate. This was designed in Phase 1 — now it becomes code.

5. **Evaluation runner** — Orchestrates the full pipeline: load data, compute features, train on optimization set, predict on all three sets, run quality gates. Outputs a structured evaluation report.

Each module should be independently testable. The builder should be able to run the feature engineer on test data and verify outputs without running the full pipeline.

### Step 4.3: Pipeline Integrity Checks

Before running training, verify pipeline integrity:

- **Data partition correctness:** Confirm no overlap between optimization, evaluation, and deployment date ranges.
- **Lookahead check:** Run `validate_no_lookahead()` on the full feature matrix.
- **Feature-target alignment:** Verify that features at time t are used to predict targets at time t + H (not time t).
- **Normalization isolation:** Verify that normalization statistics are computed only from optimization-set data.

These checks should be automated and run every time the pipeline executes — not just during development.

**Checkpoint:** The builder has a complete, config-driven pipeline with data loader, feature engineer, model+loss, validation suite, and evaluation runner. All components are modular and independently testable. Integrity checks are automated.

---

## Phase 5: Execute Three-Stage Validation

Run the pipeline through the three-stage methodology. This is where the builder's design is tested against reality.

### Step 5.1: Optimization Stage

Train the model on the optimization set. This is the only stage where the model sees data.

- Run training with the chosen loss function.
- Record optimization-set performance as a reference (this is expected to be the best performance — if evaluation-set performance exceeds it, something is wrong).

### Step 5.2: Evaluation Stage

Apply the trained model to the evaluation set. Run all quality gates.

**If gates pass:** Proceed to Step 5.3. Do not celebrate yet — evaluation-set performance may still reflect the builder's own selection bias (they designed the features and model knowing they would be tested here).

**If gates fail:** This is the critical moment. The builder must decide:

- **Diagnose, do not patch.** Examine which gates failed and why. Is it a specific sub-period? A specific feature? A regime shift? Understanding the failure is more valuable than fixing it.
- **If modifying the model or features:** The builder is now using the evaluation set to make decisions. This is acceptable (the evaluation set exists for this purpose), but it means the deployment set becomes the only unbiased check. Track how many evaluation-guided iterations occur — excessive iteration erodes the evaluation set's independence.
- **If tempted to relax the gates:** Resist. The gates were set before seeing any results. Relaxing them after failure is the definition of overfitting to the evaluation set. If the gates were genuinely too strict, the builder must articulate why before any change.
- **Never re-partition the data to get better results.** The partition was fixed in Phase 1. Changing it is a form of data snooping.

### Step 5.3: Deployment Validation

Before committing to a full-data retrain, validate that the model generalizes beyond the evaluation set. Retrain the model on all pre-deployment data (optimization set + evaluation set, minus gap buffer) using the frozen hyperparameters from Stage 1. Then evaluate on the deployment set using the same quality gates as Stage 2.

This retrain is critical: the model evaluated in Step 5.2 was trained only on the optimization set. Deployment validation retrains on the larger dataset (optimization + evaluation) to test whether the model generalizes when given more training data — not just whether the same model scores well on a different period.

This is a one-shot test. The builder has exactly one chance:

- **If gates pass:** The model has survived. The deployment-set results are the best estimate of real-world performance.
- **If gates fail:** The model does not generalize. The gap between evaluation and deployment performance quantifies the builder's own selection bias during development. Do NOT relax gate thresholds.

Record all results — deployment-set performance is the number the builder reports, not evaluation-set performance.

### Step 5.4: Document the Run

The builder must produce a complete evaluation report containing:

- Config used (exact, frozen, reproducible)
- Optimization-set performance
- Evaluation-set gate results (pass/fail for each gate)
- Number of evaluation-guided iterations (how many times the builder went back and modified the model after seeing evaluation results)
- Deployment-set gate results (pass/fail for each gate)
- Performance comparison across all three stages
- Any anomalies or concerns

**Checkpoint:** The builder has executed the full three-stage validation with documented results. If deployment gates passed, the model is a candidate for deployment. If they failed, the builder has diagnostic information about what went wrong.

---

## Phase 6: Prepare for Deployment

If — and only if — deployment gates passed, prepare the model for live use.

### Step 6.1: Full-Data Retrain

The three-stage methodology requires a final retrain before deployment. The hyperparameters were selected in Stage 1 and validated in Stage 2 — they are now frozen. Retrain the model on **all available data** (optimization + evaluation + deployment sets) using those frozen hyperparameters. This maximizes the amount of training data the deployed model has seen, without any risk of overfitting hyperparameters to the evaluation or deployment sets (those were only used to validate the already-frozen configuration).

The deployed model is this full-data retrain — not the model from Stage 2.

### Step 6.2: Freeze the Pipeline

- Lock the config file. No further changes.
- Lock all code and the retrained model weights.
- Record the git commit hash (or equivalent) that corresponds to the validated pipeline.

### Step 6.3: Define Monitoring

The quality gates do not stop at deployment. The builder should define ongoing monitoring checks:

- **Performance monitoring:** Track live predictions against realized outcomes. Define a threshold for when live performance has degraded enough to trigger re-evaluation.
- **Feature monitoring:** Track feature distributions. If the distribution of any feature drifts significantly from what was seen in training, the model may be operating outside its valid regime.
- **Staleness check:** Define a maximum age for the model. Even if performance has not degraded, the model should be periodically re-validated on fresh data.

### Step 6.4: Define the Rebuild Trigger

When monitoring detects degradation, what happens? The builder should define:
- At what threshold does degradation trigger a full rebuild?
- Does the rebuild use the same validation framework, or should gates be revisited?
- How is new data incorporated into the three-stage partition?

---

## Output Artifacts

At the end of this workflow, the builder must have produced the following complete, runnable artifacts:

| Artifact | Description |
|----------|-------------|
| `config.yaml` | Complete pipeline configuration: target, horizon, data source, partition boundaries, feature definitions, model hyperparameters, loss specification, quality gate definitions and thresholds |
| `data_loader.py` | Data fetching, caching, quality validation. Reads config, outputs clean time-indexed dataset |
| `feature_engineer.py` | Feature computation organized by estimation goal, with horizon-derived parameters and automated lookahead prevention checks |
| `model.py` | Model definition, loss function, training loop. Normalization statistics frozen from optimization set |
| `validation.py` | Quality gates as executable functions. Takes predictions and actuals, returns structured pass/fail report |
| `evaluate.py` | Pipeline orchestrator. Runs full three-stage evaluation, outputs structured report |
| `evaluation_report.md` | Complete results: per-stage performance, gate outcomes, iteration count, anomalies |
| `monitor.py` | Post-deployment monitoring: performance tracking, feature drift detection, staleness checks |

Every artifact must be runnable. The config file must fully reproduce the evaluation results. The builder should be able to hand this package to someone else and they should be able to reproduce the results without additional context.

---

## Decision Flowchart

At each phase, the builder faces decisions. Here is the logic that governs them:

```
Define what "real" means
    │
    ▼
Design quality gates (concrete, executable)
    │
    ▼
Reason: what model properties satisfy these gates?
    │
    ▼
Design features and architecture to match
    │
    ▼
Choose loss function aligned with gates
    │
    ▼
Build config-driven pipeline
    │
    ▼
Run integrity checks ──── FAIL → fix before proceeding
    │
    ▼
Train on optimization set
    │
    ▼
Evaluate on evaluation set
    │
    ├── Gates FAIL → diagnose (do not patch)
    │       │
    │       ├── Root cause understood → modify and re-evaluate
    │       │       (track iteration count)
    │       │
    │       └── Root cause unclear → reconsider problem setup
    │
    └── Gates PASS
            │
            ▼
        Retrain on opt+eval, validate on deployment set (one shot)
            │
            ├── Gates FAIL → model does not generalize
            │       │
            │       └── Gap quantifies builder selection bias
            │
            └── Gates PASS → model is a deployment candidate
                    │
                    ▼
                Freeze pipeline, define monitoring, ship
```

---

## Problem Decomposition

Before committing to a single monolithic model, consider whether the prediction problem can be decomposed into sub-problems that are individually easier to model.

**When to decompose:**
- The target variable has distinct regimes that respond to different drivers (e.g., trending vs. mean-reverting markets may need different feature sets).
- The target can be factored into components with different predictability characteristics.
- Different time horizons within the target respond to different signals.

**When NOT to decompose:**
- Decomposition introduces coupling between sub-models that is hard to validate.
- The sub-problems are not independently testable — if you cannot validate a sub-model in isolation, decomposition adds complexity without adding rigor.
- The recombination function introduces its own failure modes that are harder to diagnose than the original problem.

**If decomposing:** Apply the entire robustness-first workflow to each sub-model independently. Each sub-model gets its own quality gates, its own three-stage validation, and its own deployment decision. The recombination logic is then validated as a separate model with the sub-model outputs as features.

The validation framework you designed in Phase 1 applies to the combined output. If sub-models individually pass their gates but the combined output fails, the recombination logic is the problem — not the sub-models.

---

## Principles Checklist

Before considering the pipeline complete, verify that all nine principles have been applied:

- [ ] **Physics-first feature engineering** — Features organized around named estimation goals, not arbitrary indicators
- [ ] **Horizon-adaptive parameterization** — All lookback windows and time periods derived from the prediction horizon
- [ ] **Lookahead prevention by architecture** — Structural guardrails (automated checks, timestamp discipline), not just developer care
- [ ] **Three-stage separation** — Optimize, evaluate, deploy with non-overlapping temporal partitions
- [ ] **Loss function as modeling decision** — Loss chosen deliberately to align with gates and encode domain beliefs
- [ ] **Multi-threshold signal validation** — Multiple independent quality gates, all must pass simultaneously
- [ ] **Optimizer-gaming prevention** — Evaluation and deployment sets are equal-length; deployment set used for validation exactly once (subsequent inclusion in full-data retrain uses frozen hyperparameters)
- [ ] **Configuration-driven experimentation** — Entire pipeline driven by a config file; experiments differ by config, not code
- [ ] **Problem decomposition** — Actively considered; if applied, each sub-model independently validated

If any principle is missing or weakly applied, go back and strengthen it before declaring the pipeline complete.
