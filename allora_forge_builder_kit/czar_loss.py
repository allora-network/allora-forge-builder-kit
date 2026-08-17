"""
CZAR Loss (Composite Zero-Agnostic Returns)
============================================

A piecewise loss built on the Cauchy kernel that:
- Z-scores by local volatility
- Applies steep wrong-sign penalties
- Uses bounded arctan transitions for same-sign predictions
- Smoothly reduces loss near zero returns

Provides gradient and hessian for use as a custom LightGBM objective.
"""

import numpy as np


def derivative(x):
    return 1.0 / (1.0 + x**2)


def antiderivative(x):
    return np.arctan(x)


def double_derivative(x):
    return 2.0 * np.abs(x) / (1.0 + x**2)**2


def eps_effective(eps, delta):
    if abs(delta) == 0:
        return np.arctan(eps)
    A = (1 + delta**2) * (antiderivative(eps + delta) - antiderivative(delta))
    beta = delta / (1 + delta**2)
    return (-1 + np.sqrt(1 + 4 * beta * A)) / (2 * beta)


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def norm_smooth(z_true, eps, delta, tau):
    a = np.abs(z_true)
    d2p1 = delta**2 + 1
    num = d2p1 * (antiderivative(a + delta) - antiderivative(delta))
    denom = eps + delta / d2p1 * eps**2
    norm_min = 1.0 - num / denom

    if tau <= 0:
        return np.maximum(norm_min, 0.0)

    num_inf = d2p1 * (0.5 * np.pi - antiderivative(delta))
    norm_inf = 1.0 - num_inf / denom
    tau_eff = np.abs(tau) * np.abs(norm_inf)
    return softplus(norm_min / tau_eff) / softplus(1 / tau_eff)


def czar_loss(y_true, y_pred, std, mean=0, alpha=1, epsilon=1, tau=0.05):
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    std = np.where(np.isfinite(std) & (std > 0), std, 1e-8)
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

    C = s * d2p1 * (antiderivative(d_true) - antiderivative(s * delta))
    L1 = 0.5 * h1 * z_pred**2 - s * z_pred + C
    L2 = s * d2p1 * (antiderivative(d_true) - antiderivative(d_pred))
    dz = z_pred - z_true
    L3 = 0.5 * np.minimum(h3, h1) * dz**2 + s * d2p1 * derivative(d_true) * dz

    if epsilon > 0:
        eps_eff = eps_effective(epsilon, delta)
        softening_0 = czar_loss(0, eps_eff, 1.0, epsilon=0, alpha=alpha)
        norm = norm_smooth(z_true, eps_eff, delta, tau)
        Lsoft = norm * softening_0
    else:
        Lsoft = 0

    return np.where(u <= 0, L1, np.where(u <= a, L2, L3)) + Lsoft


def czar_gradient(y_true, y_pred, std, mean=0, alpha=1):
    std = np.where(np.isfinite(std) & (std > 0), std, 1e-8)
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

    G1 = h1 * z_pred - np.sign(z_true)
    G2 = -s * d2p1 * derivative(d_pred)
    G3 = np.minimum(h3, h1) * (z_pred - z_true)

    return np.where(u <= 0, G1, np.where(u <= a, G2, G3)) / std


def czar_hessian(y_true, y_pred, std, mean=0, alpha=1):
    std = np.where(np.isfinite(std) & (std > 0), std, 1e-8)
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

    h1 = d2p1 * double_derivative(delta)
    H1 = np.full_like(d_pred, h1)
    H2 = (1.0 + d_pred**2) * double_derivative(d_pred)
    h3 = (1.0 + d_true**2) * double_derivative(d_true)
    H3 = np.full_like(d_pred, np.minimum(h1, h3))

    return np.where(u <= 0, H1, np.where(u <= a, H2, H3)) / std**2


def make_czar_objective(std, mean=0, alpha=1):
    """
    Create a LightGBM-compatible custom objective using CZAR loss.
    
    Args:
        std: Rolling volatility for z-scoring (scalar or array matching training data)
        mean: Mean for z-scoring (usually 0 for returns)
        alpha: CZAR alpha parameter (0-1, controls MSE curvature)
    
    Returns:
        objective function compatible with LightGBM's fobj parameter
    """
    def objective(y_true_or_dataset, y_pred):
        # Handle both LightGBM Dataset objects and raw arrays
        if hasattr(y_true_or_dataset, 'get_label'):
            y_true = y_true_or_dataset.get_label()
        else:
            y_true = np.asarray(y_true_or_dataset)
        grad = czar_gradient(y_true, y_pred, std=std, mean=mean, alpha=alpha)
        hess = czar_hessian(y_true, y_pred, std=std, mean=mean, alpha=alpha)
        # Clip hessian to avoid numerical issues
        hess = np.maximum(hess, 1e-6)
        return grad, hess
    return objective
