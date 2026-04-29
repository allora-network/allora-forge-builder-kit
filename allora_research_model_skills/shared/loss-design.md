# Loss Design

The loss function is the only thing your model directly optimizes. Every other metric — directional accuracy, Sharpe ratio, profit factor — is a downstream consequence of what the loss rewards and penalizes. Choosing MSE because it is the default is like choosing a car's steering by ignoring it: you will end up somewhere, but probably not where you intended.

This document treats loss design as a first-class modeling decision. We cover the design dimensions, walk through CZAR (Composite Zero-Agnostic Returns) loss as a concrete example for log-returns prediction, then generalize to other prediction targets.

## Loss as a Modeling Decision

A loss function encodes your beliefs about what constitutes a good prediction. These beliefs include:

- **What errors matter most.** Is predicting the wrong direction worse than being off by a constant magnitude? Should large errors be penalized quadratically (MSE), linearly (MAE), or according to some other schedule?

- **What context qualifies an error.** A 1% prediction error during a calm market may be unacceptable, while a 5% error during a volatility spike may be reasonable. Should the loss normalize errors by local conditions?

- **What the prediction will be used for.** If the prediction drives a trading decision, directional accuracy matters more than point-estimate precision. If it drives a risk model, underestimating tail events is far worse than overestimating them.

Standard losses (MSE, MAE, Huber) encode none of these beliefs. They treat all errors as symmetric, context-free, and equally important regardless of direction. For many financial prediction tasks, this is the wrong set of assumptions.

## Design Dimensions

When designing a custom loss for financial prediction, consider these three dimensions. They are independent — you can combine them freely.

### Directional Asymmetry

For predictions that drive trading decisions, the direction of the error is often more important than its magnitude. Predicting +2% when the actual move is -1% (wrong direction by 3 percentage points) is worse than predicting +2% when the actual move is +5% (right direction, off by 3 percentage points).

A directionally-asymmetric loss applies a higher penalty when the predicted and actual values have opposite signs. The simplest approach multiplies the base loss by a scaling factor when the prediction and target disagree on sign:

```
loss_i = base_loss(pred_i, target_i) * (1 + alpha * wrong_direction_i)
```

where `wrong_direction_i = 1` when `sign(pred_i) != sign(target_i)` and `0` otherwise. The parameter `alpha` controls how much more the model is penalized for directional errors. Setting `alpha = 0` recovers the symmetric base loss.

The difficulty is gradient behavior at the sign boundary. A hard sign function has zero gradient, which means the optimizer receives no signal about *how close* a prediction was to the correct direction. Smooth approximations (e.g., `tanh(pred * target * scale)`) provide gradient flow through the transition, which helps the optimizer learn the boundary.

### Volatility Normalization

Financial return magnitudes vary enormously across time — a 1% daily return might be a 0.5-sigma event in a calm market or a 0.1-sigma event during a crisis. If the loss treats all 1% errors equally, the model will be dominated by high-volatility periods in the training data, because those periods produce the largest raw errors.

Normalizing the loss by local volatility puts all time periods on an equal footing:

```
normalized_error_i = (pred_i - target_i) / local_vol_i
```

where `local_vol_i` is a trailing estimate of volatility at time *i* (e.g., rolling standard deviation of returns over a recent window). This means the model optimizes for *risk-adjusted* prediction quality rather than raw magnitude.

The choice of volatility estimator and its lookback window matters. Use the same horizon-adaptive parameterization discussed in the [feature engineering](feature-engineering.md) guide: the volatility normalization window should scale with the prediction horizon.

A practical consideration: ensure `local_vol_i` has a floor (minimum value) to prevent division-by-zero in extremely quiet markets. This floor should be small enough to be reached only in degenerate cases.

### Near-Zero Awareness

For return-based prediction targets, there is a regime where directional asymmetry becomes counterproductive: near-zero returns. When the actual return is essentially zero (below the market's noise floor), the "direction" is meaningless — it is determined by microstructure noise, not by any predictable process.

Penalizing the model for getting the "wrong" direction on a +0.001% return teaches the model to fit noise. A well-designed loss should recognize that near-zero returns carry no directional information and reduce or eliminate the directional penalty in this regime.

This can be implemented through a continuous weighting function that transitions from "direction matters" (far from zero) to "direction is noise" (near zero). The transition scale should reflect the market's noise floor — itself a function of volatility.

## CZAR Loss: A Concrete Example for Log-Returns

CZAR (Composite Zero-Agnostic Returns) loss is a purpose-built loss function for log-returns prediction that combines all three design dimensions into a single piecewise objective built on the Cauchy (Lorentzian) kernel.

### Motivation

Log-returns have specific properties that standard losses handle poorly:

1. **Direction matters economically.** The sign of a log-return maps directly to a trading decision (long vs. short). A wrong-sign prediction is categorically worse than an imprecise same-sign prediction. MSE treats them equivalently.

2. **The zero-return problem.** A large fraction of short-horizon returns are effectively zero — they fall within the bid-ask spread or are indistinguishable from microstructure noise. A model that tries to predict the direction of these returns is fitting noise.

3. **Volatility clustering.** The magnitude of log-returns varies by orders of magnitude across market regimes. A loss that weights all observations equally will be dominated by extreme periods.

### Design Overview

**Volatility normalization.** All inputs are z-scored before loss computation: `z = (y - mean) / std`, where `std` is the local rolling standard deviation and `mean` is the local mean (typically zero for short horizons). This makes the loss regime-invariant.

**Core kernel.** The loss is built on the Cauchy derivative `f(x) = 1 / (1 + x²)` and its antiderivative `arctan(x)`. Bounded gradients give robustness to outliers with smooth transitions. The `alpha ∈ [0, 1]` parameter controls MSE-like curvature at the origin via a horizontal shift `δ = alpha / √3`.

**Three-region piecewise structure.** Let `s = sign(z_true)` and `u = s · z_pred`. The loss partitions based on sign agreement and magnitude:
- **Region 1 (`u ≤ 0`) — Wrong sign.** Steepest penalty. Quadratic + linear directional term.
- **Region 2 (`0 < u ≤ |z_true|`) — Right sign, undershoot.** Arctan transition with decreasing gradient as prediction approaches truth.
- **Region 3 (`u > |z_true|`) — Right sign, overshoot.** Quadratic in overshoot, with curvature that decreases for large `|z_true|`.

**Zero-agnostic softening.** The `epsilon` parameter (in std units) smoothly reduces loss as `|z_true| → 0`, preventing the model from fitting noise in near-zero returns. The `tau` parameter controls transition smoothness via a softplus hinge.

### Parameters

| Parameter | Role | Typical range |
|-----------|------|---------------|
| `std` | Local volatility for z-scoring | Rolling std of returns over a horizon-adaptive window |
| `mean` | Local mean for centering | Rolling mean, or 0 for short horizons |
| `alpha` | MSE curvature at origin | `[0, 1]`. 0 = linear only, 1 = maximum curvature |
| `epsilon` | Zero-agnostic scale (in std units) | ~1. Returns within ~epsilon standard deviations of zero are softened |
| `tau` | Hinge smoothness for zero-agnostic transition | ~0.05. Smaller = sharper transition |

### Reference Implementation

The full implementation below includes the loss function, analytical gradient, and analytical Hessian. The gradient and Hessian are essential for use as a custom objective in gradient boosting frameworks (LightGBM, XGBoost). Note the use of pseudo-gradients and pseudo-Hessians in some regions for numerical stability.

```python
import numpy as np


def derivative(x):
    return 1.0 / (1.0 + x**2)


def antiderivative(x):
    return np.arctan(x)


def double_derivative(x):
    return 2.0 * np.abs(x) / (1.0 + x**2)**2


def eps_effective(eps, delta):
    # Rescale epsilon so that 1 - loss(z_true, 0) / loss(0, epsilon) crosses zero at epsilon
    if abs(delta) == 0:
        return np.arctan(eps)

    A = (1 + delta**2) * (antiderivative(eps + delta) - antiderivative(delta))
    beta = delta / (1 + delta**2)  # coefficient on eps_eff^2 in loss(0, eps_eff, 1)

    # Solve beta*x^2 + x - A = 0 for positive x
    return (-1 + np.sqrt(1 + 4 * beta * A)) / (2 * beta)


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def norm_smooth(z_true, eps, delta, tau):
    # Minimum value of the normalisation at z_true, set by the limit that loss(z_true,0)
    # does not decrease as z_true increases.
    # Simplified from: 1 - loss(z_true, 0) / loss(0, epsilon)
    a = np.abs(z_true)
    d2p1 = delta**2 + 1
    num = d2p1 * (antiderivative(a + delta) - antiderivative(delta))
    denom = eps + delta / d2p1 * eps**2
    norm_min = 1.0 - num / denom

    if tau <= 0:
        # Hard transition
        return np.maximum(norm_min, 0.0)

    # Smooth transition when norm drops below zero
    # Scale tau_eff by |norm_inf| so the asymptote is invariant across eps, delta
    # Asymptotic value of norm_min as |z_true| -> inf
    num_inf  = d2p1 * (0.5*np.pi - antiderivative(delta))
    norm_inf = 1.0 - num_inf / denom
    tau_eff = np.abs(tau) * np.abs(norm_inf)
    return softplus(norm_min / tau_eff) / softplus(1 / tau_eff)


def czar_loss(y_true, y_pred, std, mean=0, alpha=1, epsilon=1, tau=0.05):
    """
    Composite Zero-Agnostic Return Loss

    Asymmetric, piecewise function that is
        * Linear (alpha=0) or quadratic (alpha>0) when y_pred has opposite sign to y_true
        * Linear (alpha=0) or quadratic (alpha>0) when |y_pred| > |y_true|, with a decreasing
          gradient as |z_true| increases
        * Arctangent transition from 0 < |y_pred| < |y_true|

    Args:
        y_true: True returns
        y_pred: Predicted returns
        std: Standard deviation of true returns
        mean: Mean of true returns
        alpha: MSE term constant (alpha=0 is linear only, alpha=1 is maximum gradient)
        epsilon: Loss softening scale, in units of standard deviation. Optimum is eps~1
        tau: Scaling for softening hinge function
    Returns:
        Value of loss
    """

    if alpha < 0 or alpha > 1:
        raise ValueError(f'alpha must be between 0 and 1, got {alpha}')

    z_true = (y_true - mean) / std
    z_pred = (y_pred - mean) / std

    s = np.where(z_true == 0, 1, np.sign(z_true))
    s_pred = np.where(z_pred == 0, 1, np.sign(z_pred))
    a = np.abs(z_true)
    u = s * z_pred

    # Apply horizontal shift to function for smooth change in gradient
    # Alpha should be between 0 and 1. 1/sqrt(3) shifts to the peak of the hessian function
    delta = alpha / np.sqrt(3)
    d2p1 = delta**2 + 1

    d_true = z_true + s * delta
    d_pred = z_pred + s_pred * delta

    h1 = d2p1 * double_derivative(delta)
    h3 = d2p1 * double_derivative(d_true)

    # Region 1: opposite sign (u <= 0): grad = -s + MSE term
    # Constant so that the middle branch hits zero at z_pred = z_obs
    C = s * d2p1 * (antiderivative(d_true) - antiderivative(s * delta))
    L1 = 0.5 * h1 * z_pred**2 - s * z_pred + C

    # Region 2: same sign, before threshold (0 < u <= a): grad = -s * antiderivative(z_pred)
    # antiderivative(d_true) term so that the middle branch hits zero at z_pred = z_obs
    L2 = s * d2p1 * (antiderivative(d_true) - antiderivative(d_pred))

    # Region 3: past threshold (u > a): grad = s * derivative(z_obs) + MSE term
    dz = z_pred - z_true
    L3 = 0.5 * np.minimum(h3, h1) * dz**2 + s * d2p1 * derivative(d_true) * dz

    # Softening term
    if epsilon > 0:
        eps_eff = eps_effective(epsilon, delta)
        softening_0 = czar_loss(0, eps_eff, 1., epsilon=0, alpha=alpha)
        norm = norm_smooth(z_true, eps_eff, delta, tau)
        Lsoft = norm * softening_0
    else:
        Lsoft = 0

    return np.where(u <= 0, L1, np.where(u <= a, L2, L3)) + Lsoft


def czar_gradient(y_true, y_pred, std, mean=0, alpha=1):
    z_true = (y_true - mean) / std
    z_pred = (y_pred - mean) / std

    s = np.where(z_true == 0, 1, np.sign(z_true))
    s_pred = np.where(z_pred == 0, 1, np.sign(z_pred))
    a = np.abs(z_true)
    u = s * z_pred

    delta = alpha / np.sqrt(3)
    d2p1 = delta**2 + 1

    d_true = z_true + s * delta
    d_pred = z_pred + s_pred * delta

    h1 = d2p1 * double_derivative(delta)
    h3 = d2p1 * double_derivative(d_true)

    # Region 1: opposite sign (u <= 0)
    # Pseudo gradient for numerical stability:
    G1 = h1 * z_pred - np.sign(z_true)

    # Region 2: same sign, before threshold (0 < u <= a)
    G2 = -s * d2p1 * derivative(d_pred)

    # Region 3: past threshold (u > a)
    # Pseudo gradient for numerical stability:
    G3 = np.minimum(h3, h1) * (z_pred - z_true)

    return np.where(u <= 0, G1, np.where(u <= a, G2, G3)) / std


def czar_hessian(y_true, y_pred, std, mean=0, alpha=1):
    z_true = (y_true - mean) / std
    z_pred = (y_pred - mean) / std

    s = np.where(z_true == 0, 1.0, np.sign(z_true))
    s_pred = np.where(z_pred == 0, 1.0, np.sign(z_pred))
    a = np.abs(z_true)
    u = s * z_pred

    delta = alpha / np.sqrt(3)
    d2p1 = delta**2 + 1

    d_true = s * (np.abs(z_true) + delta)
    d_pred = s_pred * (np.abs(z_pred) + delta)

    # Region 1: opposite sign (u <= 0)
    h1 = d2p1 * double_derivative(delta)
    H1 = np.full_like(d_pred, h1)

    # Region 2: same sign, before threshold (0 < u <= a)
    # Pseudo hessian for numerical stability:
    H2 = (1.0 + d_pred**2) * double_derivative(d_pred)

    # Region 3: past threshold (u > a)
    # Consistent with H2 pseudo hessian:
    h3 = (1.0 + d_true**2) * double_derivative(d_true)
    H3 = np.full_like(d_pred, np.minimum(h1, h3))

    return np.where(u <= 0, H1, np.where(u <= a, H2, H3)) / std**2
```

### Why This Works for Log-Returns Specifically

CZAR is designed for log-returns and should not be blindly applied to other prediction targets. The reasons it works here are specific to the problem:

- Log-returns are naturally centered around zero, making the three-region sign-based partitioning well-defined.
- The direction of a log-return directly maps to a trading decision (long vs. short), making the directional asymmetry (Region 1 penalty) economically meaningful.
- Volatility normalization of log-returns produces approximately standard-normal residuals, so `epsilon ≈ 1` has a natural interpretation as "within one standard deviation of zero."
- The Cauchy kernel's bounded gradients provide robustness to the fat-tailed distribution of z-scored returns.

For other prediction targets, the design principles transfer but the specific mechanism does not. See the next section for guidance on adapting loss design to different targets.

## Loss Design for Other Prediction Targets

The three design dimensions (directional asymmetry, volatility normalization, near-zero awareness) are general, but their application depends on the prediction target's properties.

### Volatility Prediction

When predicting future volatility (realized variance, conditional standard deviation), the error structure is inherently asymmetric:

- **Underestimating volatility is more dangerous than overestimating it.** Underestimation leads to undersized hedges and unexpected losses. Overestimation leads to conservative positioning — suboptimal but not catastrophic.

- **Volatility has a positive floor.** Unlike returns, volatility cannot be negative and is rarely near zero. The near-zero awareness dimension is less relevant.

- **Volatility is right-skewed.** Large volatility spikes are common; a loss that penalizes overestimation and underestimation equally will underfit the right tail.

A volatility-specific loss might apply an asymmetric penalty: standard loss when overpredicting, scaled-up loss when underpredicting. The scaling factor encodes your risk preference — how much worse is underestimation than overestimation?

### Funding Rate Prediction

Funding rates have distinctive properties:

- **Sign changes are critical events.** A funding rate that crosses zero signals a shift from one side of the market paying the other. Capturing these transitions is often more valuable than precise magnitude estimation.

- **Persistent regimes.** Funding rates tend to stay positive or negative for extended periods, then abruptly flip. A loss that emphasizes sign-change detection might weight observations near the zero crossing more heavily.

- **Bounded magnitude.** Most funding rates operate within known bounds (set by exchange rules). The tail behavior differs from returns — extreme values are clamped, not fat-tailed.

A funding-rate loss might combine a base magnitude loss with a transition-sensitive component that increases the penalty for predictions that miss a sign change while reducing the penalty for magnitude errors during stable-sign periods.

### Sentiment or Categorical Scores

When the prediction target is ordinal (bullish/neutral/bearish) or bounded (a sentiment score on [0, 1]):

- Standard classification losses (cross-entropy) may apply directly for ordinal targets.
- For bounded continuous scores, the loss should respect the bounds — predictions near the boundaries should not be penalized for clipping.
- If the score drives a trading decision, the relevant question is not "how accurate is the score?" but "how reliably does a threshold crossing predict a regime change?"

## Loss and Evaluation: Aligned but Distinct

The loss function is what the optimizer sees during training. Evaluation metrics are what you use to decide if the model is good. These should be aligned in spirit but are not identical, and conflating them causes subtle problems.

### Why They Differ

**The loss must be differentiable.** Evaluation metrics do not. You might care about the percentage of correct directional predictions (accuracy), but accuracy has zero gradient almost everywhere. The loss needs a smooth surrogate that pushes the model toward higher accuracy while remaining optimizable.

**The loss operates per-sample.** Evaluation metrics often aggregate across the full validation set. Sharpe ratio, for example, is a portfolio-level metric that depends on the joint distribution of predictions. You cannot decompose it into independent per-sample contributions for gradient computation.

**The loss is trained on; evaluation is measured on.** If you use the same metric for both, and that metric has any exploitable structure, the optimizer will find a degenerate solution that scores well on the metric without producing useful predictions. Keep the training loss and the evaluation metrics related but not identical.

### The Alignment Principle

A well-designed loss should produce models that score well on your evaluation metrics, even though the metrics are not what the model directly optimizes. If you find that your loss is decreasing during training but your evaluation metrics are not improving (or are getting worse), the loss and evaluation are misaligned — the loss is rewarding something the evaluation does not value.

When this happens, the fix is usually in the loss, not the evaluation. The evaluation metrics capture what you actually care about. The loss is your imperfect attempt to make that objective differentiable and per-sample decomposable. Iterate on the loss design until alignment improves.

## Custom Loss Implementation

Designing a good loss is worthless if the implementation is numerically broken. These considerations apply to any custom loss:

### Gradient Availability

Automatic differentiation handles most cases, but verify:

- **Non-differentiable operations.** Hard sign functions, argmax, and indicator functions have zero or undefined gradients. Use smooth approximations: `tanh(x * scale)` instead of `sign(x)`, soft-threshold functions instead of hard thresholds.
- **Conditional branches.** If your loss has `if/else` branches based on the prediction or target values, ensure gradients flow through all branches. Some frameworks handle this automatically; others may silently return zero gradients for inactive branches.
- **Gradient magnitude.** Custom losses can produce gradients that are orders of magnitude larger or smaller than standard losses. This interacts with learning rate selection. Scale your loss so that its gradients are in a reasonable range, or adjust the learning rate accordingly.

### Numerical Stability

- **Division by near-zero values.** Volatility normalization requires dividing by a volatility estimate. Add a floor: `max(vol, epsilon)` where `epsilon` is small but not negligible (e.g., 1e-8).
- **Log of near-zero values.** If your loss involves logarithms, guard against `log(0)` with a similar floor.
- **Large exponents.** Exponential weighting can produce Inf or NaN. Use clamped exponents or log-sum-exp formulations for numerical safety.
- **Mixed precision.** If training in half precision (float16), the reduced dynamic range makes numerical issues more likely. Test your custom loss in the precision you will actually train in.

### Batch Behavior

- **Per-sample decomposition.** Most training frameworks expect the loss to decompose as a mean (or sum) over the batch. If your loss has inter-sample dependencies (e.g., a ranking component), handle batch aggregation explicitly.
- **Batch size sensitivity.** Volatility normalization computed per-batch rather than per-sample can introduce batch-size-dependent behavior. Prefer pre-computed volatility normalization (computed during feature engineering and stored as a column) over within-loss computation.
- **NaN propagation.** A single NaN in the batch can poison the entire batch loss. Guard against NaN-producing edge cases (division by zero, log of negative values) at the sample level, not the batch level.

## Summary

Loss design is a chain of deliberate decisions:

1. **Start from your use case.** What does a "good prediction" mean for your downstream application? What errors are catastrophic versus tolerable?
2. **Choose your design dimensions.** Directional asymmetry, volatility normalization, and near-zero awareness are the three main levers. Not all apply to every problem.
3. **Implement carefully.** Gradient availability, numerical stability, and batch behavior are implementation concerns that can silently break a well-designed loss.
4. **Validate alignment.** Monitor both the training loss and your evaluation metrics. If they diverge, iterate on the loss design.

For log-returns prediction, CZAR provides a principled starting point that addresses the specific challenges of that target. For other targets, use the design dimensions as a framework to reason about what your loss should reward and penalize — then build accordingly.

For how loss design integrates with the broader model development workflow, see the [methodology overview](methodology.md) and [validation framework](validation-framework.md).
