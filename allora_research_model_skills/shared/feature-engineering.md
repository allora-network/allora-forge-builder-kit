# Feature Engineering

Features are the language your model uses to understand the world. Poorly designed features produce models that appear to work in backtesting but fail in production — usually because they encode future information, overfit to a specific time regime, or measure nothing meaningful. This document covers three principles that prevent these failure modes: physics-first design, horizon-adaptive parameterization, and lookahead prevention by construction.

These principles apply regardless of your prediction target (returns, volatility, funding rates, sentiment scores) and regardless of your data sources (price data, on-chain metrics, social feeds, order book snapshots).

## Physics-First Feature Design

Every feature must answer a clear estimation question about the state of the system at time *t*. "What is the recent momentum?" is an estimation question. "Feature #47 from a blog post" is not. If you cannot articulate what physical or economic quantity a feature measures, remove it — the model cannot learn a meaningful relationship from noise dressed up as signal.

### Estimation Goals as the Organizing Principle

Group your features by what they estimate, not by how they are computed. A useful taxonomy for financial data:

| Estimation Goal | What It Captures | Example Measurements |
|---|---|---|
| Trend | Directional persistence over a timescale | Returns over a window, slope of a moving average, rate of change |
| Mean Reversion | Distance from a local equilibrium | Deviation from a rolling mean, z-score relative to recent history |
| Volatility | Magnitude of recent fluctuations | Standard deviation of returns, range-based estimators, realized variance |
| Microstructure | Short-lived supply/demand imbalance | Spread dynamics, order flow imbalance, trade arrival rate |
| Cross-Asset | Relative behavior between instruments | Return differentials, correlation changes, basis spreads |
| External Signal | Information from outside the price process | Sentiment scores, on-chain activity metrics, funding rates |

This organization serves two purposes. First, it forces you to justify each feature's existence — if a feature doesn't map to an estimation goal, it probably doesn't belong. Second, it makes feature selection principled: you can reason about which estimation goals are relevant for your prediction target rather than performing blind search over hundreds of unnamed columns.

### From Estimation Goal to Feature

The path from an estimation goal to a concrete feature involves three decisions:

1. **What to measure.** Choose the raw quantity that reflects your estimation goal. For a trend estimate, this might be a price return. For a volatility estimate, it might be squared returns or a high-low range.

2. **Over what window.** Every measurement requires a lookback period. This is where horizon-adaptive parameterization (next section) becomes critical — do not pick arbitrary windows.

3. **How to normalize.** Raw measurements vary enormously across assets and time regimes. Normalizing by recent volatility, by a rolling percentile rank, or by a z-score makes features comparable and helps the model generalize. A 2% daily return means something very different for a stablecoin versus a small-cap token.

A well-designed feature makes all three decisions explicit and justifiable.

### Extending to Non-Price Data

The physics-first principle applies identically to alternative data sources. The data changes; the discipline does not.

**Social sentiment data.** The estimation goal might be "crowd directional conviction" or "attention regime change." Measurements could include aggregated sentiment scores over a window, rate of change in post volume, or divergence between sentiment and recent price action. The same questions apply: what are you measuring, over what window, and how do you normalize across different activity levels?

**On-chain metrics.** Estimation goals include "capital flow direction" (exchange inflows/outflows), "holder conviction" (wallet age distributions), or "network utilization pressure" (gas prices, transaction counts). Each produces features through the same three-decision framework.

**Order book data.** Estimation goals might be "instantaneous supply/demand imbalance" or "depth resilience." The physics here is microstructure — features should measure quantities that have economic meaning at the timescale you operate on.

The temptation with novel data sources is to throw raw columns at the model and hope it learns. Resist this. A raw sentiment score is not a feature — it is an ingredient. Transform it through the same estimation-goal framework: what does it measure, over what window, normalized how?

## Horizon-Adaptive Parameterization

Every feature involves a lookback window or period parameter. The most common mistake in feature engineering is hardcoding these: "use a 14-period RSI" or "compute 20-day volatility." These choices embed assumptions about the relevant timescale — assumptions that break when the prediction horizon changes.

### The Core Idea

All feature periods should be derived from a single parameter: the prediction horizon *h*. When *h* changes, every feature window auto-scales.

A feature set spans multiple timescales relative to the horizon:

- **Sub-horizon windows** capture recent dynamics: `h/4`, `h/2`. These see fine-grained structure within the prediction period.
- **Horizon-scale windows** capture the regime the model is predicting over: `h`, `3h/2`. These match the natural timescale of the target.
- **Super-horizon windows** capture the broader context: `2h`, `5h`, `10h`. These provide the slow-moving backdrop against which the horizon-scale dynamics play out.

For example, if your prediction horizon is 24 one-hour bars:
- Sub-horizon: 6-bar, 12-bar windows
- Horizon-scale: 24-bar, 36-bar windows
- Super-horizon: 48-bar, 120-bar, 240-bar windows

If the horizon changes to 8 bars, every window scales automatically: 2, 4, 8, 12, 16, 40, 80. No manual retuning required.

### Why This Matters

Consider what happens without horizon-adaptive parameterization. You build a model for 24-hour prediction with features using 14-bar and 50-bar windows. These work well. Now you adapt the model to 4-hour prediction. The 14-bar and 50-bar windows now represent 14 hours and 50 hours — the 50-bar window is more than 12 times the prediction horizon. It is measuring structure that is almost entirely irrelevant to what happens in the next 4 hours, while the sub-horizon resolution is too coarse.

The model might still "work" in backtesting because the optimizer can learn to ignore the irrelevant features. But you have wasted model capacity and introduced noise that hurts generalization.

### Implementation Pattern

Define your feature configuration as functions of the horizon, not as literal values:

```
Feature: rolling_trend
  windows: [h/4, h/2, h, 2h, 5h]
  measurement: return over window
  normalization: divide by rolling volatility at window scale
```

When you change the horizon from `h=24` to `h=8`, the feature specification remains identical — only the derived window values change. This is the power of parameterization: your feature set is a *recipe*, not a list of numbers.

### Edge Cases

Some features have natural minimum windows below which they become meaningless (you cannot estimate a correlation from 2 observations). When `h/4` would produce a window shorter than the minimum meaningful lookback, clamp to the minimum. Document these minimums explicitly in your feature definitions.

Similarly, some data sources have inherent timescales that override horizon scaling. Daily settlement events occur once per day regardless of your prediction horizon. Features measuring these events should be parameterized by the event frequency, not the prediction horizon. Use horizon-adaptive scaling for features where the relevant timescale is a modeling choice, and fixed scaling where the data physics dictates the timescale.

## Lookahead Prevention by Construction

Lookahead — using future information to predict the future — is the single most common cause of backtesting results that do not reproduce in production. It is also the hardest bug to find because it does not cause errors. The model trains, the metrics look good, often *suspiciously* good, and everything seems fine until live deployment.

The solution is not "be careful." Developer discipline does not scale. Instead, build structural guardrails that make lookahead impossible by construction.

### Principle: Make the Wrong Thing Impossible

Every operation in your feature pipeline should be past-only by construction, not by convention. This means:

1. **Rolling windows end at *t*, never extend past it.** A rolling mean computed at time *t* must use data from `[t-w, t]`, not `[t-w/2, t+w/2]` or any other window that includes future data. This seems obvious, but centering operations in standard libraries default to symmetric windows. Verify your rolling functions use a trailing window.

2. **Target shift is applied once, globally, at the start of pipeline construction.** The target at time *t* is the value you want to predict, observed at time *t+h*. Compute `target[t] = f(raw_data[t+h])` once, early in your pipeline, and then build all features from the raw data at time *t*. This single shift is the *only* operation that touches future data. Everything downstream operates on past-only quantities.

3. **Raw target columns are dropped from the model input.** After constructing the target, remove the raw future-looking column from the feature matrix. This prevents accidental leakage through transformations of the target that end up as features. If your feature matrix contains a column that has any temporal relationship to the target beyond what the model should learn, you have a leak.

4. **Features computed from external data inherit the same discipline.** If you join an external dataset (sentiment scores, on-chain metrics) to your time index, verify the join uses as-of logic: for each timestamp *t*, use the most recent observation at or before *t*. Never use the closest observation regardless of direction — that introduces forward-looking information for half the joins.

### Purged Cross-Validation

Standard k-fold cross-validation allows information leakage at fold boundaries. If your training set contains data up to day 100 and your validation set starts at day 101, any feature with a lookback window longer than 1 day will have training-set information bleeding into validation-set features.

Purged cross-validation inserts a gap buffer between training and validation folds. The gap should be at least as large as:
- The prediction horizon *h* (to prevent target leakage)
- The longest feature lookback window (to prevent feature leakage)

In practice, use a gap of `max(h, max_feature_window)`. Observations within the gap are excluded from both training and validation. This wastes some data but guarantees temporal separation.

For expanding-window or rolling-window validation (often more appropriate for financial data than k-fold), the gap principle still applies: insert a buffer between the end of the training window and the start of the evaluation window.

### Feature Validation Checklist

Before training any model, verify every feature passes these checks:

**Temporal integrity:**
- [ ] All rolling operations use trailing (past-only) windows
- [ ] Target shift is computed exactly once and uses the correct horizon
- [ ] No raw target or future-value columns remain in the feature matrix
- [ ] External data joins use as-of (point-in-time) logic

**Inference-time computability:**
- [ ] Every feature can be computed using only data available at prediction time
- [ ] No feature requires data that arrives with a delay longer than your prediction frequency
- [ ] Features from external sources account for the source's publication lag

**Stationarity and scaling:**
- [ ] Features are normalized or transformed to be approximately stationary
- [ ] Normalization parameters (rolling mean, rolling std) are computed from past data only
- [ ] No feature has a trend component that would cause distribution shift over time

**Implementation consistency:**
- [ ] The feature computation code used in training is identical to the code used in inference
- [ ] There is a single feature-engineering function called from both training and inference paths
- [ ] No training-only preprocessing (e.g., global z-scoring across the full dataset) that cannot be replicated in real-time

This last point deserves emphasis. A common pattern is to compute normalizing statistics (mean, standard deviation) across the entire training dataset and apply them to features. This is fine during training but impossible during inference — you do not have the future data needed to compute the global statistics. Use rolling normalizations computed from a trailing window, which are identical in both contexts.

### Common Leakage Patterns

These are the patterns that appear most often in practice. Each is subtle enough to pass code review and produce plausible-looking results.

**Survivorship bias in universe construction.** If you construct your asset universe based on data availability at the end of the backtest period (e.g., "tokens that have 2 years of data"), you exclude assets that delisted or failed during the period. The remaining assets are the survivors, and your model learns to predict in a world that only contains winners. Construct your universe as-of each point in time.

**Information in the index.** If your dataset's time index is filtered (e.g., only timestamps where trading occurred, or only timestamps where a certain condition holds), the filter itself may encode future information. A model can learn to exploit the fact that certain timestamps are present or absent. Use a regular time grid and handle missing data explicitly.

**Leakage through feature selection.** If you select features based on their correlation with the target across the full dataset, you have used future information to choose your features. Feature selection must happen within the training fold only, re-evaluated at each cross-validation split.

**Normalization across time.** Computing a z-score using the mean and standard deviation of the entire time series (including future observations) leaks future information into every feature value. Always normalize using trailing windows.

## Putting It Together

The three principles reinforce each other:

1. **Physics-first design** ensures every feature measures something meaningful, reducing the search space and making the model's behavior interpretable.

2. **Horizon-adaptive parameterization** ensures the feature set automatically adjusts to the prediction timescale, eliminating a major source of manual tuning and cross-horizon inconsistency.

3. **Lookahead prevention** ensures the features your model sees during training are identical to what it will see during inference, closing the gap between backtest performance and live performance.

When you add a new data source or extend to a new prediction target, apply all three: define the estimation goals the new data serves, parameterize its lookback windows relative to the horizon, and verify every operation is past-only. The discipline is the same regardless of what you are building.

For how these features integrate into the broader model development workflow — including validation, loss design, and deployment — see the [methodology overview](methodology.md) and [validation framework](validation-framework.md).
