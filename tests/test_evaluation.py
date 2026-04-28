"""
Tests for PerformanceEvaluator — validates the 7 primary metrics
against known-answer synthetic data.

Run with:
    pytest tests/test_evaluation.py -v
"""

import numpy as np
import pytest

from allora_forge_builder_kit import PerformanceEvaluator


@pytest.fixture
def evaluator():
    return PerformanceEvaluator()


# ── Directional Accuracy ────────────────────────────────────────────

def test_perfect_direction_gives_da_1(evaluator):
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 20)
    y_pred = np.array([0.005, -0.01, 0.02, -0.005, 0.01] * 20)
    m = evaluator.calculate_directional_metrics(y_true, y_pred)
    assert m["directional_accuracy"] == 1.0
    assert m["da_pvalue"] < 0.05


def test_random_direction_gives_da_near_half(evaluator):
    rng = np.random.RandomState(42)
    y_true = rng.randn(1000)
    y_pred = rng.randn(1000)
    m = evaluator.calculate_directional_metrics(y_true, y_pred)
    assert 0.45 <= m["directional_accuracy"] <= 0.55
    assert m["da_pvalue"] > 0.05


def test_opposite_direction_gives_low_da(evaluator):
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 20)
    y_pred = -y_true
    m = evaluator.calculate_directional_metrics(y_true, y_pred)
    assert m["directional_accuracy"] == 0.0
    assert m["da_pvalue"] >= 0.5


def test_da_ci_bounds_are_ordered(evaluator):
    rng = np.random.RandomState(7)
    y_true = rng.randn(200)
    y_pred = y_true * 0.3 + rng.randn(200) * 0.7
    m = evaluator.calculate_directional_metrics(y_true, y_pred)
    assert m["da_ci_lower"] <= m["directional_accuracy"] <= m["da_ci_upper"]


def test_n_eff_reduces_with_autocorrelation(evaluator):
    rng = np.random.RandomState(1)
    n = 500
    # AR(1) true returns with high persistence create streaks of same sign.
    # A constant-sign prediction is correct during positive streaks and
    # wrong during negative streaks, producing autocorrelated correctness.
    y_true = np.zeros(n)
    for i in range(1, n):
        y_true[i] = 0.9 * y_true[i - 1] + rng.randn() * 0.01
    y_pred = np.full(n, 0.001)
    m = evaluator.calculate_directional_metrics(y_true, y_pred)
    assert m["da_n_effective"] < m["da_n_samples"]


# ── Pearson Correlation ─────────────────────────────────────────────

def test_perfect_correlation(evaluator):
    y_true = np.linspace(-1, 1, 100)
    y_pred = y_true * 2.0
    m = evaluator.calculate_correlation_metrics(y_true, y_pred)
    assert m["pearson_r"] > 0.99
    assert m["pearson_pvalue"] < 0.001


def test_no_correlation(evaluator):
    rng = np.random.RandomState(99)
    y_true = rng.randn(500)
    y_pred = rng.randn(500)
    m = evaluator.calculate_correlation_metrics(y_true, y_pred)
    assert abs(m["pearson_r"]) < 0.15


# ── WRMSE Improvement ──────────────────────────────────────────────

def test_perfect_predictions_give_full_wrmse_improvement(evaluator):
    y_true = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
    y_pred = y_true.copy()
    m = evaluator.calculate_wrmse_improvement(y_true, y_pred)
    assert m["wrmse_model"] == 0.0
    assert m["wrmse_improvement"] == 1.0


def test_zero_predictions_give_zero_wrmse_improvement(evaluator):
    y_true = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
    y_pred = np.zeros_like(y_true)
    m = evaluator.calculate_wrmse_improvement(y_true, y_pred)
    assert abs(m["wrmse_improvement"]) < 1e-10


# ── CZAR Improvement ───────────────────────────────────────────────

def test_perfect_direction_gives_czar_1(evaluator):
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 10)
    y_pred = y_true * 0.5
    m = evaluator.calculate_czar_improvement(y_true, y_pred)
    assert abs(m["czar_improvement"] - 1.0) < 1e-10


def test_opposite_direction_gives_negative_czar(evaluator):
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 10)
    y_pred = -y_true
    m = evaluator.calculate_czar_improvement(y_true, y_pred)
    assert m["czar_improvement"] == pytest.approx(-1.0)


def test_random_czar_near_zero(evaluator):
    rng = np.random.RandomState(55)
    y_true = rng.randn(1000)
    y_pred = rng.randn(1000)
    m = evaluator.calculate_czar_improvement(y_true, y_pred)
    assert abs(m["czar_improvement"]) < 0.15


# ── Threshold pass/fail ─────────────────────────────────────────────

def test_strong_model_passes_all_primary_metrics(evaluator):
    rng = np.random.RandomState(12)
    y_true = rng.randn(500)
    y_pred = y_true * 0.8 + rng.randn(500) * 0.2
    report = evaluator.evaluate(y_true, y_pred)
    assert all(report["passed"].values()), (
        f"Strong model should pass all metrics: {report['passed']}"
    )


def test_random_model_fails_most_metrics(evaluator):
    rng = np.random.RandomState(42)
    y_true = rng.randn(500)
    y_pred = rng.randn(500)
    report = evaluator.evaluate(y_true, y_pred)
    n_passed = sum(report["passed"].values())
    assert n_passed <= 3, f"Random model passed too many metrics: {report['passed']}"


# ── Full evaluate pipeline ──────────────────────────────────────────

def test_evaluate_returns_expected_structure(evaluator):
    rng = np.random.RandomState(0)
    y_true = rng.randn(100)
    y_pred = y_true + rng.randn(100) * 0.5
    report = evaluator.evaluate(y_true, y_pred)

    assert "metrics" in report
    assert "passed" in report
    assert "score" in report
    assert "grade" in report
    assert "num_passed" in report
    assert len(report["passed"]) == 7

    for key in [
        "directional_accuracy", "da_ci_lower", "da_pvalue",
        "pearson_r", "pearson_pvalue",
        "wrmse_improvement", "czar_improvement",
    ]:
        assert key in report["metrics"]


def test_evaluate_rejects_mismatched_lengths(evaluator):
    with pytest.raises(ValueError, match="same length"):
        evaluator.evaluate(np.array([1, 2, 3]), np.array([1, 2]))


def test_evaluate_rejects_empty_arrays(evaluator):
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate(np.array([]), np.array([]))


def test_grade_scale(evaluator):
    passed_all = {f"metric_{i}": True for i in range(7)}
    score, grade, n = evaluator.calculate_performance_score(passed_all)
    assert grade == "A+"
    assert n == 7

    passed_none = {f"metric_{i}": False for i in range(7)}
    score, grade, n = evaluator.calculate_performance_score(passed_none)
    assert grade == "F"
    assert n == 0
