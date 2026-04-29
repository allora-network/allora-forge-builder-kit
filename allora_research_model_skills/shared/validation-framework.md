# Validation Framework: Three-Stage Architecture and Quality Gates

This document specifies how to evaluate financial prediction models honestly. It implements Principles 4 (three-stage separation), 6 (multi-threshold signal validation), and 7 (optimizer-gaming prevention) from the [methodology](./methodology.md).

The core problem: evaluating a model on data that influenced its construction produces optimistically biased performance estimates. This bias is not small — it routinely turns worthless models into apparent successes. The three-stage architecture eliminates this bias through strict data separation, and multi-threshold quality gates ensure that models meeting the bar are genuinely useful.

---

## Why Standard Cross-Validation Fails for Time Series

Standard k-fold cross-validation assumes samples are exchangeable — any sample can appear in any fold. Financial time series violate this assumption fundamentally:

1. **Temporal autocorrelation.** Adjacent samples share information. If sample `t` is in the training set and sample `t+1` is in the test set, the model has already seen data highly correlated with the test point.

2. **Feature lookback contamination.** Features computed with rolling or expanding windows incorporate historical data. A feature at time `t` depends on data from `t-W` to `t` (for window size `W`). If any of those historical points are in the test set of a different fold, information flows from test to train.

3. **Non-stationarity.** The data-generating process changes over time. A random split mixes samples from different regimes, allowing the model to interpolate between regimes rather than extrapolate — which is what it must do in production.

These are not edge cases. They are fundamental properties of financial time series. Any validation scheme that ignores them produces unreliable performance estimates.

---

## Stage 1: Inner Optimization (Hyperparameter Selection)

### Purpose

Find good hyperparameters through systematic search. This stage consumes data — the selected hyperparameters encode information about the training data, so inner CV scores are optimistically biased estimates of true performance.

### Purged Walk-Forward Cross-Validation

Walk-forward CV respects the temporal ordering of financial data. Each fold uses a contiguous block of past data for training and a subsequent contiguous block for testing, advancing through time:

```
Fold 1:  [====TRAIN====]--gap--[=TEST=]
Fold 2:       [====TRAIN====]--gap--[=TEST=]
Fold 3:            [====TRAIN====]--gap--[=TEST=]
```

**Purging** removes samples from the training set that are temporally close to the test set. **Gap buffers** insert an empty zone between training and test data. Together, they prevent information leakage through temporal autocorrelation and feature lookback windows.

### Gap Buffer Sizing

The gap buffer must be at least as large as the larger of:

- **The prediction horizon.** If you predict `H` steps ahead, the target at time `t` depends on data at time `t+H`. Without a gap of at least `H`, training samples near the boundary have targets that overlap with the test period.

- **The maximum feature lookback window.** If your longest feature uses a window of `W` steps, then a test sample at time `t` depends on data from `t-W` to `t`. Without a gap of at least `W`, training could include samples that share lookback data with test samples.

In practice, use the maximum of both, plus a safety margin. Since horizon-adaptive parameterization (Principle 2) expresses all periods as multiples of `H`, the maximum feature lookback is typically some multiple `k * H`, making the gap buffer `k * H` plus margin.

**Insufficient gap buffers are worse than no CV at all** — they create the illusion of honest evaluation while leaking information. When in doubt, make the gap larger. The cost is fewer effective training samples per fold; the benefit is honest scores.

### Walk-Forward Design Choices

Several structural choices affect the quality of inner CV:

**Expanding vs. sliding window training.** Expanding windows use all data up to the current fold; sliding windows use a fixed-size lookback. Expanding windows give the model more data (better for small datasets) but assume older data is still relevant (worse in non-stationary markets). Sliding windows are more adaptive but require enough data per window for stable training.

**Number of folds.** More folds give lower-variance CV score estimates but each fold has less unique test data. Fewer folds give noisier estimates but each fold is more independent. A reasonable starting point is 5-10 folds, ensuring each fold has enough test samples for the binomial tests used in validation (Principle 6).

**Test fold size.** Each test fold should contain enough samples for statistical significance. For directional accuracy, a binomial test needs roughly 400+ predictions to have a reasonable chance of detecting a 55% accuracy rate, and 780+ for 80% statistical power (see the Statistical Rigor section below for the full derivation). Adjust based on your expected signal strength — weaker signals need more samples to detect.

---

## Stage 2: Outer Evaluation (Honest Assessment)

### Purpose

Obtain an unbiased estimate of out-of-sample performance using data that was never seen during Stage 1 — not even indirectly through hyperparameter selection.

### Design Requirements

**Independent time windows.** The outer evaluation uses time windows that are strictly later than (or otherwise separated from) all data used in Stage 1. These windows were not available to the optimizer in any form.

**Fixed hyperparameters.** All hyperparameters are fixed to the values selected in Stage 1. No further tuning, no "just one more adjustment." Every modification after seeing outer evaluation data converts the outer evaluation into another round of inner optimization.

**Fresh model training per window.** For each outer evaluation window, retrain the model from scratch on the training portion of that window (using the fixed hyperparameters), then evaluate on the held-out portion. This simulates what would happen in production: the model is trained on available history and evaluated on the subsequent period.

**Multiple independent windows.** A single out-of-sample window is insufficient — performance on one window could be due to favorable market conditions. Multiple windows across different time periods provide confidence that performance is robust across regimes.

### How to Partition Data

The simplest approach: divide your full dataset into an inner portion (for Stage 1) and an outer portion (for Stage 2) along the time axis. The inner portion is earlier; the outer portion is later. A gap buffer separates them, sized the same way as the inner CV gap buffers.

A more sophisticated approach uses nested walk-forward evaluation, where the outer evaluation itself walks forward through multiple windows, each time using all prior data for Stage 1 optimization and the next window for outer evaluation. This is computationally expensive but provides the most realistic simulation of production deployment.

---

## Optimizer-Gaming Prevention

Even with strict Stage 1 / Stage 2 separation, the optimizer can implicitly game the evaluation through validation set design. Three mechanisms close these loopholes.

### Validation-Test Parity

**Principle:** The validation set used for early stopping during model training should be the same size as the test set used for evaluation.

**Why this matters:** Early stopping monitors the validation loss and stops training when it begins to degrade. If the validation set is much smaller than the test set, the validation loss is noisier — the model may stop at a point that happens to look good on the noisy validation signal but performs poorly on the smoother test signal (or vice versa). Matching sizes ensures the statistical properties of the signal the model optimizes against match the signal it is evaluated on.

**Implementation:** When you define your walk-forward folds, ensure the validation portion (used during training) is drawn from the end of the training block and has the same number of samples as the test portion.

### Constant Total Test Coverage

**Principle:** When exploring different test set sizes as a hyperparameter, hold the total number of test samples constant across configurations.

**Why this matters:** If you double the test window size and also double the total test data (by using the same number of windows), you are changing two things simultaneously: the resolution of individual tests and the total data used for evaluation. This confounds the comparison.

To maintain constant coverage: if you double the test window size, halve the number of test windows. The product `test_size * num_windows` stays constant. This ensures that differences in performance across configurations reflect the effect of test window size, not the effect of having more or less test data overall.

**Formula:** `num_test_windows = const_test_epochs / test_size`, where `const_test_epochs` is a fixed constant representing the total test coverage budget.

### Progressive Range Narrowing

**Principle:** Run large hyperparameter searches in stages, narrowing the search range based on top-performing configurations.

**Why this matters:** A single large search across all hyperparameter combinations risks the optimizer memorizing configurations that happen to perform well on specific data windows. Progressive narrowing mitigates this:

1. Run a broad search across the full parameter space.
2. Identify the top-performing configurations (e.g., top 20%).
3. Narrow the search ranges to the region spanned by these top configurations.
4. Run a finer search within the narrowed ranges.

This converges toward robust configurations (those that perform well across the parameter space) rather than isolated lucky points.

---

## Multi-Threshold Signal Validation

A model is accepted only if it simultaneously passes all quality gates. These gates are independent — they test different aspects of prediction quality that cannot substitute for each other.

### Gate 1: Directional Accuracy

**What it tests:** Does the model predict the correct sign of the target more often than chance?

**How to measure:** Fraction of predictions where `sign(prediction) == sign(actual)`. Baseline is 50% for a symmetric target distribution (adjust if the target is significantly asymmetric).

**Why it matters:** This is the minimum bar for predictive value. A model that cannot predict direction is not useful, regardless of its other properties.

### Gate 2: Statistical Significance

**What it tests:** Is the observed directional accuracy distinguishable from random guessing?

**How to measure:** Apply a binomial test. Under the null hypothesis of random guessing, the number of correct directional predictions follows a Binomial(n, 0.5) distribution (or Binomial(n, p_base) if the baseline is not 50%). Compute the p-value for the observed accuracy.

**Confidence intervals.** Report confidence intervals for directional accuracy, not just point estimates. An accuracy of 54% with a 95% CI of [48%, 60%] tells a very different story than 54% with a CI of [52%, 56%].

**Why p-values alone are insufficient:** A p-value tells you whether the result is distinguishable from chance, not whether it is large enough to be useful. A model with 50.1% accuracy on 1 million predictions will have a tiny p-value but is practically worthless. Always pair statistical significance with practical significance (the other gates).

### Gate 3: Calibration

**What it tests:** Are prediction magnitudes informative? When the model predicts a large move, does a larger move actually occur?

**How to measure:** Partition predictions by magnitude (e.g., quintiles) and verify that actual target magnitudes increase monotonically across quintiles. Alternatively, compute rank correlation between absolute prediction magnitude and absolute actual magnitude.

**Why it matters:** A model that always predicts the same magnitude (just varying sign) passes the directional accuracy gate but carries no information about *how much* the target will move. Calibration ensures the model's confidence levels are meaningful.

### Gate 4: Financial Improvement

**What it tests:** Does the model improve financial outcomes relative to a simple baseline?

**How to measure:** This is application-specific. Possible baselines include: no-trade (always predict zero), persistence (predict the same as the last observation), or simple moving average. The improvement metric should reflect your actual use case — information coefficient, Sharpe ratio improvement, PnL improvement, or whatever measures the value the model adds.

**Why it matters:** A model can be statistically significant, directionally accurate, and well-calibrated, yet still not improve on a simple baseline if the simple baseline already captures most of the available signal. This gate ensures the model adds real value.

### Composing the Gates

A model passes validation if and only if it passes ALL gates simultaneously. The logic is:

```
passed = (
    directional_accuracy > threshold_direction
    AND p_value < threshold_significance
    AND calibration_score > threshold_calibration
    AND financial_improvement > threshold_improvement
)
```

**Setting thresholds.** Thresholds should be set based on what would make the model useful in practice, not based on what current models achieve. This requires domain judgment: what directional accuracy is commercially meaningful? What financial improvement justifies deployment? Set these before evaluating any model to avoid hindsight bias.

**Threshold independence.** Each gate tests a genuinely different property. A model cannot pass by being excellent on one gate and poor on others. This is by design — each gate catches a different class of failure that the others miss.

---

## Statistical Rigor

### Binomial Tests for Directional Accuracy

The binomial test is the appropriate significance test for directional accuracy because each prediction is a binary outcome (correct or incorrect direction) with a known baseline probability.

**Requirements:**
- Each prediction must be approximately independent. With purged walk-forward CV, this is ensured by gap buffers that remove temporal dependence.
- The baseline probability must be correctly specified. For symmetric targets, use 0.5. For asymmetric targets (e.g., markets with upward drift), compute the empirical baseline from the target distribution.

**Sample size considerations:** The minimum number of predictions needed depends on the signal strength you want to detect. To detect 55% accuracy (vs. 50% baseline) with 95% confidence and 80% power, you need approximately 780 predictions. Weaker signals need more data. Plan your test fold sizes accordingly.

### Confidence Intervals

Always report confidence intervals for key metrics, not just point estimates. For directional accuracy, the Wilson score interval is preferred over the normal approximation (it has better coverage properties for probabilities near 0 or 1, and for small sample sizes).

For financial improvement metrics, bootstrap confidence intervals are appropriate — they make no distributional assumptions and handle the complex dependencies in financial returns.

### Multiple Testing

If you evaluate multiple model variants and select the best, you face a multiple testing problem — the more variants you try, the more likely one passes by chance. Account for this in Stage 2 by:

1. **Pre-registering** which model variant you will evaluate in Stage 2 before looking at outer evaluation data.
2. If evaluating multiple variants, applying a Bonferroni correction (or less conservative alternatives like Holm-Bonferroni) to your significance thresholds.
3. Reporting the number of configurations evaluated alongside the final results.

---

## Deployment Validation

Before committing to a full-data retrain, validate that the model generalizes beyond the evaluation set. This catches models that pass Stage 2 but exploit patterns specific to the evaluation period.

**Three-partition approach:** Divide the outer portion into an evaluation set and a deployment set of equal size, separated by a gap buffer. Stage 2 evaluates on the evaluation set. Deployment validation retrains on all pre-deployment data (inner portion + evaluation set, minus gap buffer) and evaluates on the deployment set using the same quality gates.

- If deployment validation passes: proceed to Stage 3.
- If deployment validation fails: diagnose the gap between evaluation and deployment performance. Do NOT relax gate thresholds. The degradation between stages is itself a useful signal — modest degradation suggests mild overfitting; large degradation suggests the model captured regime-specific patterns rather than durable signal.

## Production Deployment (Stage 3)

Once a model passes all quality gates on both evaluation and deployment sets:

1. **Retrain on all available data** using the selected hyperparameters. This maximizes the information available to the deployed model.

2. **Lock the configuration.** The deployed model runs on the exact config that passed validation. No manual tweaks, no "I think this parameter should be slightly different."

3. **Monitor for degradation.** Production models degrade over time as market conditions change. Define monitoring metrics and degradation thresholds that trigger re-evaluation. The monitoring metrics should align with your validation quality gates — if the model would no longer pass the gates on recent data, it needs retraining or replacement.

4. **Scheduled re-evaluation.** Periodically run the full three-stage pipeline on updated data. This catches gradual degradation that single-metric monitoring might miss.

---

## Summary: The Validation Pipeline

```
Full Dataset
  |
  |-- [Inner Portion] -------> Stage 1: Purged Walk-Forward CV
  |                              |
  |                              |--> Hyperparameter search
  |                              |--> Validation-test parity
  |                              |--> Progressive range narrowing
  |                              |--> Output: selected hyperparameters
  |
  |-- [gap buffer]
  |
  |-- [Evaluation Set] ------> Stage 2: Independent Evaluation
  |                              |
  |                              |--> Fixed hyperparameters from Stage 1
  |                              |--> Multi-threshold quality gates
  |                              |--> FAIL --> Back to Stage 1
  |
  |-- [gap buffer]
  |
  |-- [Deployment Set] ------> Deployment Validation
                                 |
                                 |--> Retrain on inner + evaluation data
                                 |--> Same quality gates
                                 |--> FAIL --> Diagnose, do not relax gates
                                 |--> PASS --> Stage 3: Full-data retrain + deploy
```

Every element of this pipeline exists to prevent a specific failure mode. Removing any element opens a pathway for false confidence in model quality. The pipeline is demanding because the alternative — deploying models based on optimistic backtests — is how most quantitative strategies fail.
