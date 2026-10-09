from __future__ import annotations
"""
Allora Forge Builder Kit - Performance Metrics Evaluation
==========================================================

Offline evaluation for scalar returns and triple-barrier class probabilities.

Usage:
    from allora_forge_builder_kit import PerformanceEvaluator
    
    evaluator = PerformanceEvaluator()
    report = evaluator.evaluate(
        y_true=actual_log_returns,
        y_pred=predicted_log_returns,
        epoch_length_minutes=60
    )
    
    print(report)
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats
import warnings


class PerformanceEvaluator:
    """
    Comprehensive performance metrics calculator for financial time-series predictions.
    
    Scalar evaluation implements seven criteria with a letter grade.
    Triple-barrier evaluation implements six strict criteria and no grade.
    The volatility mode retains the legacy scalar path; dedicated volatility
    diagnostics live in the volatility example workflows.
    
    Log-return evaluate() uses the worker-metrics RES-1578 promotion gate and
    confidence estimators. Existing scalar helper methods and THRESHOLDS below
    retain legacy behavior for volatility and direct helper callers.
    """

    THRESHOLDS = {
        'directional_accuracy': 0.52,
        'da_ci_lower': 0.50,             # CI lower bound must be ABOVE this
        'da_pvalue': 0.05,               # must be BELOW this
        'pearson_r': 0.05,
        'pearson_pvalue': 0.05,          # must be BELOW this
        'wrmse_improvement': 0.05,
        'czar_improvement': 0.10,
    }

    NUM_PRIMARY_METRICS = 7

    # Performance grades based on composite score (7 primary metrics)
    GRADES = {
        7: 'A+',
        6: 'A',
        5: 'B+',
        4: 'B',
        3: 'C',
        2: 'D',
        1: 'F',
        0: 'F',
    }

    TEMPORAL_COVERAGE_THRESHOLD = 0.50
    
    def __init__(self, target_type="log_return"):
        """Select scalar evaluation (default) or triple-barrier classification."""
        if target_type not in ("log_return", "volatility", "triple_barrier"):
            raise ValueError(f"Unknown target_type: {target_type}")
        self.target_type = target_type
    
    @staticmethod
    def power_tanh(x: np.ndarray, p: float = 3.0) -> np.ndarray:
        """
        Power-tanh transformation for robust loss function.
        
        Args:
            x: Input array
            p: Power parameter (default: 3.0)
            
        Returns:
            Transformed array
        """
        return np.tanh(np.abs(x) ** p) * np.sign(x)
    
    def calculate_directional_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate directional accuracy and related metrics.
        
        Uses a z-test with continuity correction and autocorrelation-aware
        effective sample size.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with DA, confidence interval, and p-value
        """
        # Exclude zero true returns where direction is undefined
        nonzero = y_true != 0
        y_true_nz = y_true[nonzero]
        y_pred_nz = y_pred[nonzero]
        n = len(y_true_nz)

        if n == 0:
            return {
                'directional_accuracy': 0.5,
                'da_ci_lower': 0.0,
                'da_ci_upper': 1.0,
                'da_pvalue': 1.0,
                'da_n_effective': 0.0,
                'da_n_samples': 0,
                'da_n_correct': 0,
                'da_autocorrelation': 0.0,
            }

        correct_direction = np.sign(y_true_nz) == np.sign(y_pred_nz)
        da = np.mean(correct_direction)
        n_correct = int(np.sum(correct_direction))
        
        # Effective sample size accounting for lag-1 autocorrelation.
        correct_float = correct_direction.astype(float)
        if n > 2:
            rho = np.corrcoef(correct_float[:-1], correct_float[1:])[0, 1]
            if np.isnan(rho):
                rho = 0.0
            rho = max(rho, 0.0)
            n_eff = n * (1 - rho) / (1 + rho)
            n_eff = max(n_eff, 2.0)
        else:
            rho = 0.0
            n_eff = float(n)
        
        # Z-test with continuity correction
        # H0: p = 0.5, H1: p > 0.5
        z_stat = (da - 0.5 - 0.5 / n_eff) / np.sqrt(0.25 / n_eff)
        z_stat = max(z_stat, 0.0)
        da_pvalue = 1.0 - stats.norm.cdf(z_stat)
        
        # Wilson score CI using effective sample size
        z_ci = 1.96
        p_hat = da
        ne = n_eff
        denominator = 1 + z_ci**2 / ne
        center = (p_hat + z_ci**2 / (2 * ne)) / denominator
        margin = z_ci * np.sqrt((p_hat * (1 - p_hat) / ne + z_ci**2 / (4 * ne**2))) / denominator
        
        da_ci_lower = center - margin
        da_ci_upper = center + margin
        
        return {
            'directional_accuracy': da,
            'da_ci_lower': da_ci_lower,
            'da_ci_upper': da_ci_upper,
            'da_pvalue': da_pvalue,
            'da_n_samples': n,
            'da_n_effective': n_eff,
            'da_n_correct': n_correct,
            'da_autocorrelation': rho,
        }
    
    def calculate_correlation_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate Pearson correlation and related metrics.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with Pearson r, p-value, and related metrics
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson_r, pearson_pvalue = stats.pearsonr(y_pred, y_true)
        
        if np.isnan(pearson_r):
            pearson_r = 0.0
            pearson_pvalue = 1.0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman_r, spearman_pvalue = stats.spearmanr(y_pred, y_true)
        
        if np.isnan(spearman_r):
            spearman_r = 0.0
            spearman_pvalue = 1.0
        
        return {
            'pearson_r': pearson_r,
            'pearson_pvalue': pearson_pvalue,
            'spearman_r': spearman_r,
            'spearman_pvalue': spearman_pvalue,
        }
    
    def calculate_wrmse_improvement(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate Weighted RMSE improvement vs. zero-returns baseline.
        
        WRMSE weights errors by the magnitude of actual returns,
        giving more importance to larger moves.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with WRMSE metrics and improvement
        """
        weights = np.abs(y_true)
        weights_sum = np.sum(weights)
        
        if weights_sum == 0:
            return {
                'wrmse_model': 0.0,
                'wrmse_baseline': 0.0,
                'wrmse_improvement': 0.0,
            }
        
        squared_errors = (y_true - y_pred) ** 2
        wrmse_model = np.sqrt(np.sum(weights * squared_errors) / weights_sum)
        
        baseline_squared_errors = y_true ** 2
        wrmse_baseline = np.sqrt(np.sum(weights * baseline_squared_errors) / weights_sum)
        
        if wrmse_baseline > 0:
            wrmse_improvement = (wrmse_baseline - wrmse_model) / wrmse_baseline
        else:
            wrmse_improvement = 0.0
        
        return {
            'wrmse_model': wrmse_model,
            'wrmse_baseline': wrmse_baseline,
            'wrmse_improvement': wrmse_improvement,
        }

    def calculate_czar_improvement(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate Cumulative Z-scored Absolute Return (CZAR) improvement.

        Replaces ZPTAE as a primary metric. Measures the fraction of z-scored directional returns captured by
        the model relative to a perfect-direction oracle.

        A value of 0 corresponds to random guessing (50% DA); 1.0 means
        every directional bet was correct, weighted by z-scored magnitude.

        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns

        Returns:
            Dictionary with CZAR model score, oracle score, and improvement
        """
        stdev = np.std(y_true)
        if stdev == 0:
            return {
                'czar_model': 0.0,
                'czar_oracle': 0.0,
                'czar_improvement': 0.0,
            }

        z_true = y_true / stdev
        correct = np.sign(y_true) == np.sign(y_pred)

        czar_model = np.sum(np.where(correct, np.abs(z_true), -np.abs(z_true)))
        czar_oracle = np.sum(np.abs(z_true))
        czar_improvement = czar_model / czar_oracle

        return {
            'czar_model': czar_model,
            'czar_oracle': czar_oracle,
            'czar_improvement': czar_improvement,
        }

    def calculate_zptae_improvement(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate Z-transformed Power-Tanh Absolute Error improvement.
        
        Retained as an additional (non-scored) metric for backward
        compatibility.  Replaced by CZAR in the primary metrics.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with ZPTAE metrics and improvement
        """
        stdev = np.std(y_true)
        if stdev == 0:
            return {
                'zptae_model': 0.0,
                'zptae_baseline': 0.0,
                'zptae_improvement': 0.0,
            }
        
        weights = np.abs(y_true)
        weights_sum = np.sum(weights)
        
        if weights_sum == 0:
            return {
                'zptae_model': 0.0,
                'zptae_baseline': 0.0,
                'zptae_improvement': 0.0,
            }
        
        z_true = y_true / stdev
        z_pred = y_pred / stdev
        z_baseline = np.zeros_like(y_true)
        
        pt_diff_model = np.abs(
            self.power_tanh(z_true) - self.power_tanh(z_pred)
        )
        zptae_model = np.sum(weights * pt_diff_model) / weights_sum
        
        pt_diff_baseline = np.abs(
            self.power_tanh(z_true) - self.power_tanh(z_baseline)
        )
        zptae_baseline = np.sum(weights * pt_diff_baseline) / weights_sum
        
        if zptae_baseline > 0:
            zptae_improvement = (zptae_baseline - zptae_model) / zptae_baseline
        else:
            zptae_improvement = 0.0
        
        return {
            'zptae_model': zptae_model,
            'zptae_baseline': zptae_baseline,
            'zptae_improvement': zptae_improvement,
        }
    
    def calculate_aspect_ratio(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate log aspect ratio: log10(std(predicted) / std(actual)).
        
        Retained as an additional (non-scored) metric.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with aspect ratio metrics
        """
        std_true = np.std(y_true)
        std_pred = np.std(y_pred)
        
        if std_true == 0 or std_pred == 0:
            log_aspect_ratio = 0.0
        else:
            log_aspect_ratio = np.log10(std_pred / std_true)
        
        return {
            'std_true': std_true,
            'std_pred': std_pred,
            'log_aspect_ratio': log_aspect_ratio,
        }
    
    def calculate_naive_annualized_return(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        epoch_length_minutes: int
    ) -> dict[str, float]:
        """
        Calculate naive annualized return from a simple trading strategy.
        
        Strategy: Go long if prediction is positive, short if negative.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            epoch_length_minutes: Length of each prediction period in minutes
            
        Returns:
            Dictionary with return metrics
        """
        same_sign = np.sign(y_true) == np.sign(y_pred)
        
        naive_return = (
            np.sum(same_sign * np.abs(y_true)) - 
            np.sum(~same_sign * np.abs(y_true))
        )
        
        n = len(y_true)
        minutes_per_year = 365.24 * 24 * 60
        annualization_factor = minutes_per_year / epoch_length_minutes / n
        naive_annualized_return = naive_return * annualization_factor
        
        return {
            'naive_return': naive_return,
            'naive_annualized_return': naive_annualized_return,
            'n_samples': n,
            'epoch_length_minutes': epoch_length_minutes,
        }
    
    def calculate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate standard regression metrics.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with MAE, MSE, RMSE, R-squared, MAPE
        """
        errors = y_true - y_pred
        
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100
        else:
            mape = np.inf
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
        }
    
    def calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate classification metrics for directional predictions.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with precision, recall, F1, specificity
        """
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }
    
    def check_primary_metrics_pass(self, metrics: dict[str, float]) -> dict[str, bool]:
        """
        Check which of the 7 primary metrics pass their thresholds.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Dictionary with pass/fail for each primary metric
        """
        return {
            'da_pass': metrics['directional_accuracy'] >= self.THRESHOLDS['directional_accuracy'],
            'da_ci_pass': metrics['da_ci_lower'] >= self.THRESHOLDS['da_ci_lower'],
            'da_pvalue_pass': metrics['da_pvalue'] < self.THRESHOLDS['da_pvalue'],
            'pearson_r_pass': metrics['pearson_r'] >= self.THRESHOLDS['pearson_r'],
            'pearson_pvalue_pass': metrics['pearson_pvalue'] < self.THRESHOLDS['pearson_pvalue'],
            'wrmse_improvement_pass': metrics['wrmse_improvement'] >= self.THRESHOLDS['wrmse_improvement'],
            'czar_improvement_pass': metrics['czar_improvement'] >= self.THRESHOLDS['czar_improvement'],
        }

    @staticmethod
    def check_temporal_coverage(
        n_predictions: int,
        n_expected: int,
        threshold: float = 0.50,
    ) -> bool:
        """
        Check whether predictions cover a sufficient portion of the evaluation window.

        Args:
            n_predictions: Number of actual predictions submitted.
            n_expected: Total number of epochs in the evaluation window.
            threshold: Minimum fraction of epochs that must have predictions.

        Returns:
            ``True`` if coverage >= threshold.
        """
        if n_expected <= 0:
            return False
        return (n_predictions / n_expected) >= threshold

    def calculate_performance_score(
        self,
        passed: dict[str, bool],
    ) -> tuple[float, str, int]:
        """
        Calculate overall performance score and grade.

        The composite score is the count of the 7 primary metrics that pass,
        divided by 7. Temporal coverage is informational only (see evaluate()).

        Args:
            passed: Dictionary of pass/fail for the 7 primary metrics.

        Returns:
            Tuple of ``(score, grade, num_passed)``.
        """
        num_passed = sum(passed.values())
        score = num_passed / 7.0
        grade = self.GRADES.get(num_passed, 'F')
        return score, grade, num_passed
    
    @staticmethod
    def validate_probabilities(values):
        """Validate probabilities in explicit [down, neutral, up] order."""
        p = np.asarray(values, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or not len(p):
            raise ValueError("Expected a non-empty Nx3 probability array [down, neutral, up]")
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Probabilities must be finite and between zero and one")
        if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-8):
            raise ValueError("Probability rows must sum to one")
        return p

    @staticmethod
    def causal_class_baseline(prediction_times, history_times, history_truth):
        """Previous 100 resolved targets; history_times are availability times.

        Call per topic. Include history preceding the evaluation window, not
        merely the subset of opportunities at which the worker submitted.
        """
        import pandas as pd
        truth = np.asarray(history_truth, dtype=float)
        # No resolved history is valid; preserve shape validation for malformed
        # empty inputs instead of treating every zero-sized array as Nx3.
        if truth.shape == (0,):
            truth = np.empty((0, 3))
        if truth.shape != (0, 3):
            truth = PerformanceEvaluator.validate_probabilities(truth)
        if not np.isin(truth, [0, 1]).all():
            raise ValueError("Baseline history must be one-hot")
        times = pd.DatetimeIndex(pd.to_datetime(history_times, utc=True)).as_unit("ns")
        queries = pd.DatetimeIndex(pd.to_datetime(prediction_times, utc=True)).as_unit("ns")
        if len(times) != len(truth) or times.hasnans or queries.hasnans:
            raise ValueError("Aligned, non-null availability and prediction times required")
        order = np.argsort(times.asi8, kind='stable')
        times, truth = times.asi8[order], truth[order]
        sums = np.vstack([np.zeros(3), np.cumsum(truth, axis=0)])
        ends = np.searchsorted(times, queries.asi8, side='right')
        starts = np.maximum(0, ends - 100)
        result = np.full((len(queries), 3), 1 / 3)
        valid = ends > starts
        result[valid] = (sums[ends[valid]] - sums[starts[valid]]) / (ends - starts)[valid, None]
        return result

    def evaluate_classification(
        self, y_true, y_pred, *, baseline_probabilities,
        n_expected_epochs=None, n_submitted=None, participation_kind='observed',
        transaction_cost=None, seed=42,
    ):
        """Six ratified gates; optional payoff is reported separately.

        Bounds are paired circular block-bootstrap percentiles (10 rows,
        1,000 replicates). n_eff=nvalid/10 is a descriptive block count, not
        another eligibility gate. Missing participation fails that gate.
        """
        truth = self.validate_probabilities(y_true)
        pred = self.validate_probabilities(y_pred)
        baseline = self.validate_probabilities(baseline_probabilities)
        if truth.shape != pred.shape or truth.shape != baseline.shape:
            raise ValueError("Truth, predictions, and baseline must have identical shapes")
        if not np.isin(truth, [0, 1]).all():
            raise ValueError("Classification truth must be one-hot")
        if participation_kind not in ('observed', 'offline'):
            raise ValueError("participation_kind must be observed or offline")
        n = len(truth)
        y, hard, base_hard = truth.argmax(1), pred.argmax(1), baseline.argmax(1)
        weights = (np.arange(3)[:, None] - np.arange(3)[None, :]) ** 2 / 4

        def metrics(indices):
            actual, ph, bh = y[indices], hard[indices], base_hard[indices]
            yt, pp, bp = truth[indices], pred[indices], baseline[indices]
            acc, bacc = np.mean(ph == actual), np.mean(bh == actual)
            observed = np.mean(weights[actual, ph])
            expected = (weights * np.outer(np.bincount(actual, minlength=3),
                                           np.bincount(ph, minlength=3))).sum() / len(indices) ** 2
            kappa = 1 - observed / expected if expected > 0 else np.nan
            bl, bbl = np.mean(np.sum((pp - yt) ** 2, axis=1)), np.mean(np.sum((bp - yt) ** 2, axis=1))
            # Clip only the realized class probability, not the whole vector.
            pt = np.clip(pp[np.arange(len(indices)), actual], 1e-15, 1)
            bt = np.clip(bp[np.arange(len(indices)), actual], 1e-15, 1)
            fl, bfl = np.mean(-(1 - pt) ** 2 * np.log(pt)), np.mean(-(1 - bt) ** 2 * np.log(bt))
            return dict(accuracy=acc, baseline_accuracy=bacc, accuracy_improvement=acc - bacc,
                        quadratic_weighted_kappa=kappa, brier_loss=bl, baseline_brier_loss=bbl,
                        brier_skill=1 - bl / bbl if bbl > 0 else np.nan,
                        focal_loss=fl, baseline_focal_loss=bfl,
                        focal_skill=1 - fl / bfl if bfl > 0 else np.nan)

        report = metrics(np.arange(n))
        names = ('accuracy_improvement', 'quadratic_weighted_kappa', 'brier_skill', 'focal_skill')
        boot = {k: [] for k in names}
        # For n <= block length, circular samples are permutations of the same
        # rows and bounds can collapse. Keep the ratified gates unchanged;
        # provisional reports must not be read as independent-sample evidence.
        rng = np.random.default_rng(seed)
        for _ in range(1000):
            starts = rng.integers(0, n, size=(n + 9) // 10)
            idx = ((starts[:, None] + np.arange(10)) % n).ravel()[:n]
            m = metrics(idx)
            for key in names:
                if np.isfinite(m[key]):
                    boot[key].append(m[key])
        for key in names:
            vals = boot[key]
            bounds = np.percentile(vals, [5, 95]) if vals else [None, None]
            report[key + '_ci'] = dict(lower=bounds[0], upper=bounds[1])
            if not vals:
                report[key] = None
        participation = None
        if n_expected_epochs is not None:
            submitted = n if n_submitted is None else n_submitted
            if (int(n_expected_epochs) != n_expected_epochs or n_expected_epochs <= 0
                    or int(submitted) != submitted or not 0 <= submitted <= n_expected_epochs):
                raise ValueError("Participation requires integer counts: 0 <= submitted <= available, available > 0")
            participation = submitted / n_expected_epochs
        report.update(nvalid=n, n_eff=n / 10, n_eff_method='nvalid / bootstrap block length',
                      provisional=n < 100, participation=participation,
                      participation_kind=participation_kind if participation is not None else 'unavailable',
                      bootstrap=dict(block_length=10, replicates=1000, seed=seed,
                                     finite_replicates={k: len(v) for k, v in boot.items()}))
        gates = [('accuracy_improvement', report['accuracy_improvement'], .02)]
        gates += [(k + '_lower', report[k + '_ci']['lower'], 0) for k in names]
        gates += [('participation', participation, .90)]
        report['criteria'] = [dict(key=k, value=v, threshold=t,
                                   passed=bool(v is not None and np.isfinite(v) and v > t)) for k, v, t in gates]
        report['eligible'] = all(c['passed'] for c in report['criteria'])
        if transaction_cost is not None:
            if not np.isfinite(transaction_cost) or transaction_cost < 0:
                raise ValueError("transaction_cost must be finite and non-negative")
            payoff = (hard - 1) * (y - 1) - transaction_cost * (hard != 1)
            report['directional_payoff'] = dict(mean=float(payoff.mean()), sum=float(payoff.sum()),
                                               transaction_cost=float(transaction_cost), units='barrier units')
        def clean(x):
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            if isinstance(x, list):
                return [clean(v) for v in x]
            if isinstance(x, (float, np.floating)):
                return float(x) if np.isfinite(x) else None
            return x
        return clean(report)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epoch_length_minutes: int = 60,
        n_expected_epochs: Optional[int] = None,
        **classification_options,
    ) -> Dict:
        """
        Evaluate log returns with worker-metrics promotion criteria, or probabilities
        for triple_barrier. Volatility retains the legacy scalar evaluator.

        Log returns accept optional lags (epoch gaps), gt_ratio (horizon in epochs),
        horizon_seconds, and n_submitted. Without n_expected_epochs, participation
        is explicitly assumed to be full offline coverage. With counts supplied,
        participation is graded at >90%. A row defaults to one prediction horizon.
        Existing score remains a fraction; eligible requires all seven criteria.

        Classification requires baseline_probabilities in classification_options;
        see evaluate_classification for its full keyword contract and report.
        The scalar arguments and graded report below apply to scalar mode only.
        
        Args:
            y_true: Ground truth log returns (actual)
            y_pred: Predicted log returns
            epoch_length_minutes: Length of each prediction epoch in minutes
            n_expected_epochs: Total number of epochs in the evaluation
                window. When provided, temporal coverage is checked and
                included in the report as ``temporal_coverage_pass`` (informational
                only — does not affect the score or grade).

        Returns:
            Comprehensive dictionary with all metrics, pass/fail, and grade.
            ``report['temporal_coverage_pass']`` is ``None`` when
            ``n_expected_epochs`` is not provided, or a ``bool`` when it is.
            Guard with ``is not None`` before using it.
        """
        if self.target_type == "triple_barrier":
            return self.evaluate_classification(y_true, y_pred, n_expected_epochs=n_expected_epochs, **classification_options)
        if self.target_type == "log_return":
            return _lr_evaluate_log_returns(y_true, y_pred, epoch_length_minutes,
                                        n_expected_epochs, **classification_options)

        if classification_options:
            raise TypeError("Classification options require target_type=triple_barrier")
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred must have same length. Got {len(y_true)} and {len(y_pred)}")
        
        if len(y_true) == 0:
            raise ValueError("y_true and y_pred cannot be empty")
        
        metrics = {}
        
        # 7 Primary Metrics
        metrics.update(self.calculate_directional_metrics(y_true, y_pred))
        metrics.update(self.calculate_correlation_metrics(y_true, y_pred))
        metrics.update(self.calculate_wrmse_improvement(y_true, y_pred))
        metrics.update(self.calculate_czar_improvement(y_true, y_pred))
        
        # Additional (non-scored) metrics
        metrics.update(self.calculate_zptae_improvement(y_true, y_pred))
        metrics.update(self.calculate_aspect_ratio(y_true, y_pred))
        metrics.update(self.calculate_naive_annualized_return(y_true, y_pred, epoch_length_minutes))
        metrics.update(self.calculate_regression_metrics(y_true, y_pred))
        metrics.update(self.calculate_classification_metrics(y_true, y_pred))
        
        # Check pass/fail for the 7 primary metrics
        passed = self.check_primary_metrics_pass(metrics)

        temporal_pass = None
        if n_expected_epochs is not None:
            temporal_pass = self.check_temporal_coverage(len(y_true), n_expected_epochs)
        score, grade, num_passed = self.calculate_performance_score(passed)

        report = {
            'metrics': metrics,
            'passed': passed,
            'score': score,
            'grade': grade,
            'num_passed': num_passed,
            'num_primary_metrics': self.NUM_PRIMARY_METRICS,
            'thresholds': self.THRESHOLDS.copy(),
            'temporal_coverage_pass': temporal_pass,
        }

        return report
    
    def print_report(self, report: Dict, detailed: bool = True):
        """
        Print a formatted performance report.
        
        Args:
            report: Output from evaluate()
            detailed: If True, show all metrics. If False, show only primary metrics.
        """
        if report.get('target_type') == 'log_return':
            print("LOG-RETURN PERFORMANCE — worker-metrics promotion criteria")
            print(f"Passed {report['num_passed']}/7; eligible: {report['eligible']}")
            for criterion in report['criteria']:
                status = 'PASS' if criterion['passed'] else 'FAIL'
                print(f"  {status}: {criterion['label']} (value={criterion['value']})")
            print(f"Participation: {report['participation_basis']}")
            if detailed:
                for key in ['directional_accuracy', 'pearson_r', 'wrmse_improvement',
                            'czar_improvement', 'log_aspect_ratio', 'mse', 'rmse']:
                    print(f"  {key}: {report['metrics'].get(key)}")
            return
        if 'criteria' in report:
            import json
            print(json.dumps(report, indent=2, allow_nan=False))
            return
        m = report['metrics']
        p = report['passed']
        
        print("=" * 80)
        print("PERFORMANCE EVALUATION REPORT")
        print("=" * 80)
        print(f"\nOVERALL PERFORMANCE: {report['grade']} ({report['num_passed']}/7 points)")
        print(f"   Primary metrics passed: {sum(p.values())}/{self.NUM_PRIMARY_METRICS}")
        print(f"   Performance Score: {report['score']:.2%}\n")

        print("=" * 80)
        print("PRIMARY METRICS (7 Core Metrics)")
        print("=" * 80)

        print(f"\n1. Directional Accuracy:")
        print(f"   Value: {m['directional_accuracy']:.4f}  {'PASS' if p['da_pass'] else 'FAIL'}")
        print(f"   Threshold: >= {self.THRESHOLDS['directional_accuracy']}")
        print(f"   Correct: {m['da_n_correct']}/{m['da_n_samples']} predictions")

        print(f"\n2. DA CI Lower Bound:")
        print(f"   Value: {m['da_ci_lower']:.4f}  {'PASS' if p['da_ci_pass'] else 'FAIL'}")
        print(f"   Threshold: >= {self.THRESHOLDS['da_ci_lower']}")
        print(f"   95% CI: [{m['da_ci_lower']:.4f}, {m['da_ci_upper']:.4f}]")
        print(f"   Effective n: {m['da_n_effective']:.1f} (autocorr: {m['da_autocorrelation']:.3f})")

        print(f"\n3. DA Statistical Significance:")
        print(f"   p-value: {m['da_pvalue']:.4f}  {'PASS' if p['da_pvalue_pass'] else 'FAIL'}")
        print(f"   Threshold: < {self.THRESHOLDS['da_pvalue']}")
        print(f"   Method: z-test with continuity correction (n_eff={m['da_n_effective']:.1f})")

        print(f"\n4. Pearson Correlation:")
        print(f"   r: {m['pearson_r']:.4f}  {'PASS' if p['pearson_r_pass'] else 'FAIL'}")
        print(f"   Threshold: >= {self.THRESHOLDS['pearson_r']}")

        print(f"\n5. Pearson Statistical Significance:")
        print(f"   p-value: {m['pearson_pvalue']:.4f}  {'PASS' if p['pearson_pvalue_pass'] else 'FAIL'}")
        print(f"   Threshold: < {self.THRESHOLDS['pearson_pvalue']}")

        print(f"\n6. WRMSE Improvement:")
        print(f"   Improvement: {m['wrmse_improvement']:.4f} ({m['wrmse_improvement']:.2%})  {'PASS' if p['wrmse_improvement_pass'] else 'FAIL'}")
        print(f"   Threshold: >= {self.THRESHOLDS['wrmse_improvement']} (5%)")
        print(f"   Model WRMSE: {m['wrmse_model']:.6f}")
        print(f"   Baseline WRMSE: {m['wrmse_baseline']:.6f}")

        print(f"\n7. CZAR Improvement:")
        print(f"   Improvement: {m['czar_improvement']:.4f} ({m['czar_improvement']:.2%})  {'PASS' if p['czar_improvement_pass'] else 'FAIL'}")
        print(f"   Threshold: >= {self.THRESHOLDS['czar_improvement']} (10%)")
        print(f"   Model CZAR: {m['czar_model']:.6f}")
        print(f"   Oracle CZAR: {m['czar_oracle']:.6f}")
        
        if detailed:
            print("\n" + "=" * 80)
            print("ADDITIONAL METRICS (non-scored)")
            print("=" * 80)

            print(f"\nLog Aspect Ratio:")
            print(f"   Value: {m['log_aspect_ratio']:.4f}")
            print(f"   Std(true): {m['std_true']:.6f}")
            print(f"   Std(pred): {m['std_pred']:.6f}")

            print(f"\nZPTAE (legacy):")
            print(f"   Improvement: {m['zptae_improvement']:.4f} ({m['zptae_improvement']:.2%})")
            print(f"   Model ZPTAE: {m['zptae_model']:.6f}")
            print(f"   Baseline ZPTAE: {m['zptae_baseline']:.6f}")
            
            print(f"\nRegression Metrics:")
            print(f"   MAE:  {m['mae']:.6f}")
            print(f"   MSE:  {m['mse']:.6f}")
            print(f"   RMSE: {m['rmse']:.6f}")
            print(f"   R²:   {m['r2']:.6f}")
            print(f"   MAPE: {m['mape']:.2f}%")
            
            print(f"\nClassification Metrics:")
            print(f"   Precision:   {m['precision']:.4f}")
            print(f"   Recall:      {m['recall']:.4f}")
            print(f"   F1 Score:    {m['f1_score']:.4f}")
            print(f"   Specificity: {m['specificity']:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(f"   True Positives:  {m['true_positives']}")
            print(f"   True Negatives:  {m['true_negatives']}")
            print(f"   False Positives: {m['false_positives']}")
            print(f"   False Negatives: {m['false_negatives']}")
            
            print(f"\nTrading Simulation:")
            print(f"   Naive Return:            {m['naive_return']:.6f}")
            print(f"   Naive Annualized Return: {m['naive_annualized_return']:.6f} ({m['naive_annualized_return']:.2%})")
            
            print(f"\nAdditional Correlation:")
            print(f"   Spearman r: {m['spearman_r']:.4f} (p={m['spearman_pvalue']:.4f})")
        
        print("\n" + "=" * 80)

# Log-return confidence intervals and promotion criteria.
# Numerical reference: worker-metrics bd01f9d2bdb6351e351ecddeb68d0fb101ebc0ee.
# These private helpers preserve the public evaluator/report API above. Legacy
# volatility and classification paths continue to use their existing methods.
# DA and improvement bounds use horizon-adjusted effective sample counts;
# Pearson and log-aspect-ratio bounds do not. The aspect interval is one SE.


_LR_LIM_PVALUE = 0.05
_LR_CONFIDENCE_LEVEL = 1.0 - _LR_LIM_PVALUE  # 0.95
_LR_PI_2 = np.pi / 2
_LR_MIN_EFFECTIVE_SAMPLES = 20.0
_LR_LIM_DA_CI = 0.50
_LR_LIM_CORRELATION_CI = 0.0
_LR_LIM_WRMSE_CI_PCT = 0.0
_LR_LIM_WCZAR_CI_PCT = 0.0
_LR_LIM_LOG_ASPECT_RATIO = 0.5
_LR_LIM_PARTICIPATION = 0.90
_LR_PROMOTION_CL = 0.95
_LR_NEFF_ANCHOR_MINUTES = 20.0
_LR_NEFF_POWER = 0.5
_LR_ONE_SE_CL: float = float(stats.norm.cdf(1.0))
_LR_SECONDS_PER_YEAR: float = 365.25 * 24 * 60 * 60


def _lr_safe_float(value: Any) -> Optional[float]:
    """Convert a value to a JSON-safe float."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _lr_czar_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of arctan: 1 / (1 + x^2)."""
    return 1.0 / (1.0 + x**2)


def _lr_czar_antiderivative(x: np.ndarray) -> np.ndarray:
    """Antiderivative (arctan)."""
    return np.arctan(x)


def _lr_softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus function."""
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _lr_norm_smooth(z_true: np.ndarray, eps: float, tau: float = 0.1) -> np.ndarray:
    """Smooth normalization for CZAR loss softening term."""
    norm_min = 1.0 - np.abs(_lr_czar_antiderivative(z_true)) / eps

    if eps >= _LR_PI_2 or tau == 0:
        return np.maximum(norm_min, 0.0)

    tau_eff = tau * (_LR_PI_2 / eps - 1)
    return _lr_softplus(norm_min / tau_eff) / _lr_softplus(1 / tau_eff)


def _lr_loss_czar(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    std: float,
    mean: float = 0,
    epsilon: float = 1,
    tau: float = 0.1,
) -> np.ndarray:
    """CZAR (Clamped Z-score Aspect Ratio) loss function."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    z_true = (y_true - mean) / std
    z_pred = (y_pred - mean) / std

    s = np.where(z_true == 0, 1, np.sign(z_true))
    a = np.abs(z_true)
    u = s * z_pred

    C = s * _lr_czar_antiderivative(z_true)

    L1 = -s * z_pred + C
    L2 = -s * _lr_czar_antiderivative(z_pred) + C
    k = s * _lr_czar_derivative(z_true)
    L3 = k * (z_pred - z_true)

    if epsilon > 0:
        eps_eff = np.arctan(1) * epsilon
        softening_0 = _lr_loss_czar(np.array([0.0]), np.array([eps_eff]), 1.0, epsilon=0)[0]
        norm = _lr_norm_smooth(z_true, eps_eff, tau)
        Lsoft = norm * softening_0
    else:
        Lsoft = 0

    return np.where(u <= 0, L1, np.where(u <= a, L2, L3)) + Lsoft


def _lr_lag_autocorr(x: np.ndarray, lags: np.ndarray, lag: int = 1) -> Tuple[float, int]:
    """Compute autocorrelation for pairs exactly ``lag`` epochs apart."""
    x = np.asarray(x, dtype=float)
    lags = np.asarray(lags)

    if len(x) < 3 or lag < 1 or len(lags) != len(x) - 1:
        return np.nan, 0
    if np.any(~np.isfinite(lags)) or np.any(lags <= 0):
        return np.nan, 0
    rounded_lags = np.rint(lags)
    if np.any(lags != rounded_lags):
        return np.nan, 0
    lags = rounded_lags.astype(np.int64)

    positions = np.concatenate(([0], np.cumsum(lags, dtype=np.int64)))
    targets = positions + lag
    partners = np.searchsorted(positions, targets, side="left")
    starts = np.flatnonzero(partners < len(positions))
    if starts.size == 0:
        return np.nan, 0

    ends = partners[starts]
    exact = positions[ends] == targets[starts]
    starts = starts[exact]
    ends = ends[exact]
    finite = np.isfinite(x[starts]) & np.isfinite(x[ends])
    x0 = x[starts[finite]]
    x1 = x[ends[finite]]
    n_pairs = len(x0)
    if n_pairs == 0:
        return np.nan, 0

    x0 = x0 - np.mean(x0)
    x1 = x1 - np.mean(x1)
    denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))

    if denom == 0:
        return np.nan, 0

    return np.sum(x0 * x1) / denom, n_pairs


def _lr_sum_autocorr(x: np.ndarray, lags: np.ndarray, gt_ratio: int) -> float:
    """Sum significant autocorrelations up to gt_ratio lags."""
    rho_sum = 0.0

    # The design calls for every epoch lag through gt_ratio - 1.  The number
    # of submitted observations is not an upper bound on epoch distance: 50
    # sparse observations two epochs apart still have pairs at lag 98.
    for lag in range(1, max(1, gt_ratio)):
        rho, n_lag = _lr_lag_autocorr(x, lags, lag)
        if n_lag > 0 and not np.isnan(rho):
            signif = 1.96 / np.sqrt(n_lag)
            if rho > signif:
                rho_sum += rho

    return rho_sum


def _lr_compute_effective_sample_size(
    x: np.ndarray, lags: Optional[np.ndarray], gt_ratio: int
) -> float:
    """Compute effective sample size accounting for autocorrelation and GT lag."""
    x = np.asarray(x)
    n = len(x)

    if n < 3:
        return max(1.0, n)

    if lags is None:
        return max(1.0, n / gt_ratio)

    lags = np.asarray(lags)
    rho_sum = _lr_sum_autocorr(x, lags, gt_ratio)

    if np.isnan(rho_sum) or rho_sum < 0:
        n_eff = n / gt_ratio
    else:
        n_eff = n / (1.0 + 2.0 * rho_sum)

    return max(1.0, n_eff)


def _lr_ztest_pvalue(p_hat: float, n: float, p: float) -> float:
    """One-sided z-test p-value for proportion."""
    if n <= 0:
        return 1.0
    num = n * p_hat - n * p - 0.5
    denom = np.sqrt(n * p * (1 - p))
    if denom == 0:
        return 1.0
    z = num / denom
    return stats.norm.sf(z)


def _lr_ztest_ci(
    p_hat: float, n: float, cl: float = _LR_CONFIDENCE_LEVEL
) -> Tuple[float, float]:
    """Analytic confidence interval for binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    if not (0.0 <= p_hat <= 1.0):
        return (0.0, 1.0)
    if p_hat == 0 or n == 1:
        return (0.0, 1.0)

    z = stats.norm.ppf(cl)
    a2 = (z * z) / n
    c = 0.5 / n
    tp = p_hat - c

    A = 1.0 + a2
    B = -(2.0 * tp + a2)
    C_coeff = tp * tp

    disc = B * B - 4.0 * A * C_coeff
    if disc < 0:
        return (0.0, 1.0)

    sqrt_disc = np.sqrt(disc)
    p2 = (-B - sqrt_disc) / (2.0 * A)

    return (max(p2, 0.0), 1.0)


def _lr_directional_accuracy_test(
    predicted_returns: np.ndarray,
    true_returns: np.ndarray,
    n_eff: float,
    cl: float = _LR_CONFIDENCE_LEVEL,
) -> Tuple[int, float, float, Tuple[float, float]]:
    """Directional accuracy with a one-sided lower bound at *n_eff*."""
    true_returns = np.asarray(true_returns, dtype=float)
    predicted_returns = np.asarray(predicted_returns, dtype=float)

    invalid_mask = (true_returns == 0) | ~np.isfinite(true_returns)
    valid_mask = ~invalid_mask
    n = np.sum(valid_mask)

    if n == 0:
        return (0, np.nan, np.nan, (np.nan, np.nan))

    success_mask = true_returns[valid_mask] * predicted_returns[valid_mask] > 0
    n_successes = np.sum(success_mask)
    directional_accuracy = n_successes / n

    pvalue = _lr_ztest_pvalue(directional_accuracy, n_eff, 0.5)
    confidence_interval = _lr_ztest_ci(directional_accuracy, n_eff, cl)

    return (int(n), directional_accuracy, pvalue, confidence_interval)


def _lr_pearsonr_ci(
    r: float, n: float, alpha: float = _LR_LIM_PVALUE
) -> Tuple[float, float]:
    """Fisher z-transform confidence interval for Pearson correlation."""
    if n <= 3 or np.isnan(r):
        return (np.nan, np.nan)

    r = np.clip(r, -0.9999, 0.9999)

    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)

    z_crit = stats.norm.ppf(1 - alpha / 2)

    z_low = z - z_crit * se
    z_high = z + z_crit * se

    return (np.tanh(z_low), np.tanh(z_high))


def _lr_correlation_test(
    predicted_returns: np.ndarray,
    true_returns: np.ndarray,
    n_eff: float,
    cl: float = _LR_CONFIDENCE_LEVEL,
) -> Tuple[float, Tuple[float, float], float]:
    """Pearson correlation test with effective sample size adjustment."""
    true_returns = np.asarray(true_returns)
    predicted_returns = np.asarray(predicted_returns)

    mask = np.isfinite(true_returns) & np.isfinite(predicted_returns)
    x = predicted_returns[mask]
    y = true_returns[mask]
    N = len(x)

    if N < 3:
        return (np.nan, (np.nan, np.nan), np.nan)

    # scipy.stats.pearsonr emits ConstantInputWarning to stderr when either
    # input has zero variance (correlation is 0/0, undefined). Return nan
    # early — downstream code already handles nan r values correctly.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return (np.nan, (np.nan, np.nan), np.nan)

    pearson = stats.pearsonr(x, y)
    r = getattr(pearson, "statistic", getattr(pearson, "correlation", pearson[0]))

    n_eff = max(3.0, min(float(N), n_eff))

    ci = _lr_pearsonr_ci(r, n_eff, alpha=1.0 - cl)

    df = n_eff - 2.0
    if abs(r) == 1.0:
        p_val_eff = 0.0
    else:
        t_stat = r * np.sqrt(df / (1.0 - r**2))
        p_val_eff = 2.0 * stats.t.sf(np.abs(t_stat), df=df)

    return (r, ci, p_val_eff)


def _lr_finite_pairs(
    pred: np.ndarray, actual: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """The samples where both series are finite, as (pred, actual)."""
    pred = np.asarray(pred, float)
    actual = np.asarray(actual, float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    return pred[mask], actual[mask]


def _lr_se_from_influence(if_t: np.ndarray, n_eff: float) -> float:
    """Compute standard error from influence function series."""
    if_t = np.asarray(if_t, float)
    return np.std(if_t, ddof=1) / np.sqrt(max(1.0, n_eff))


def _lr_wrmse_improvement_and_ci(
    pred: np.ndarray, actual: np.ndarray, n_eff: float, cl: float = _LR_CONFIDENCE_LEVEL
) -> Tuple[float, Tuple[float, float]]:
    """WRMSE improvement with influence-function-based confidence interval."""
    p, r = _lr_finite_pairs(pred, actual)
    if p.size < 3:
        return (np.nan, (np.nan, np.nan))

    w = np.abs(r)
    m1 = w * r**2
    m2 = w * (r - p)**2

    b = m1.mean()
    c = m2.mean()

    if b == 0:
        # The weighted baseline itself is zero (every actual exactly zero):
        # there is no error to improve on, so improvement is undefined-as-zero.
        return (0.0, (0.0, 0.0))
    if c == 0:
        # The MODEL's weighted error is exactly zero — a flawless scored
        # subsample. theta = 1 - sqrt(0/b) = 1.0, the best possible
        # improvement, with a zero-width band (the influence series is
        # identically zero). The research reference has no guard here and
        # returns exactly this; failing it as 0% would be a wrong answer at
        # the ratified gate (SYNTH-008).
        return (1.0, (1.0, 1.0))

    s = np.sqrt(c / b)
    theta = 1.0 - s

    dtheta_db = 0.5 * s / b
    dtheta_dc = -0.5 / np.sqrt(b * c)

    if_t = dtheta_db * (m1 - b) + dtheta_dc * (m2 - c)

    se = _lr_se_from_influence(if_t, n_eff)
    z = stats.norm.ppf(cl)

    return (theta, (theta - z * se, theta + z * se))


def _lr_wczar_improvement_and_ci(
    pred: np.ndarray, actual: np.ndarray, n_eff: float, cl: float = _LR_CONFIDENCE_LEVEL
) -> Tuple[float, Tuple[float, float]]:
    """CZAR improvement with influence-function-based confidence interval."""
    p, r = _lr_finite_pairs(pred, actual)
    if p.size < 3:
        return (np.nan, (np.nan, np.nan))

    r_bar = r.mean()
    v = (r * r).mean() - r_bar * r_bar
    sigma = np.sqrt(max(v, 1e-12))

    w = np.abs(r)

    def g0(sig: float) -> np.ndarray:
        return w * _lr_loss_czar(r, 0.0, sig)

    def g1(sig: float) -> np.ndarray:
        return w * _lr_loss_czar(r, p, sig)

    g0_t = g0(sigma)
    g1_t = g1(sigma)

    B = g0_t.mean()
    C_val = g1_t.mean()

    if B == 0:
        return (0.0, (0.0, 0.0))

    theta = 1.0 - C_val / B

    dtheta_dB = C_val / (B * B)
    dtheta_dC = -1.0 / B

    fd_rel = 1e-4
    h = fd_rel * sigma + 1e-12
    Bp = (g0(sigma + h).mean() - g0(sigma - h).mean()) / (2.0 * h)
    Cp = (g1(sigma + h).mean() - g1(sigma - h).mean()) / (2.0 * h)
    dtheta_dsig = dtheta_dB * Bp + dtheta_dC * Cp

    r2_bar = (r * r).mean()
    IF_v_t = (r * r - r2_bar) - 2.0 * r_bar * (r - r_bar)
    IF_sig_t = (0.5 / sigma) * IF_v_t

    if_t = (
        dtheta_dB * (g0_t - B)
        + dtheta_dC * (g1_t - C_val)
        + dtheta_dsig * IF_sig_t
    )

    se = _lr_se_from_influence(if_t, n_eff)
    z = stats.norm.ppf(cl)

    return (theta, (theta - z * se, theta + z * se))


def _lr_log_aspect_ratio_and_ci(
    pred: np.ndarray, actual: np.ndarray, n_eff: float, cl: float = _LR_CONFIDENCE_LEVEL
) -> Tuple[float, Tuple[float, float]]:
    """Log aspect ratio with influence-function-based confidence interval."""
    x, y = _lr_finite_pairs(pred, actual)
    if x.size < 3:
        return (np.nan, (np.nan, np.nan))

    eps = 1e-12

    Ex = x.mean()
    Ex2 = (x * x).mean()
    Ey = y.mean()
    Ey2 = (y * y).mean()

    vx = max(Ex2 - Ex * Ex, eps)
    vy = max(Ey2 - Ey * Ey, eps)

    sx = np.sqrt(vx)
    sy = np.sqrt(vy)

    if sx == 0 or sy == 0:
        return (np.nan, (np.nan, np.nan))

    theta = np.log10(sx / sy)

    c = 1.0 / np.log(10.0)
    dtheta_dEx = c * 0.5 * (1.0 / vx) * (-2.0 * Ex)
    dtheta_dEx2 = c * 0.5 * (1.0 / vx) * 1.0
    dtheta_dEy = c * (-0.5) * (1.0 / vy) * (-2.0 * Ey)
    dtheta_dEy2 = c * (-0.5) * (1.0 / vy) * 1.0

    m = np.column_stack([x, x * x, y, y * y])
    mu = m.mean(axis=0)

    if_t = (
        dtheta_dEx * (m[:, 0] - mu[0])
        + dtheta_dEx2 * (m[:, 1] - mu[1])
        + dtheta_dEy * (m[:, 2] - mu[2])
        + dtheta_dEy2 * (m[:, 3] - mu[3])
    )

    se = _lr_se_from_influence(if_t, n_eff)
    z = stats.norm.ppf(cl)

    return (theta, (theta - z * se, theta + z * se))


def _lr_normalized_mean_offset(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean signed offset normalised by ground-truth standard deviation."""
    pred = np.asarray(pred)
    actual = np.asarray(actual)
    if pred.size == 0 or actual.size == 0:
        return float("nan")
    std_actual = np.std(actual)
    if std_actual == 0:
        return float("nan")
    return float(np.mean(pred - actual) / std_actual)


def _lr_neff_scale(gt_lag_minutes: Optional[float]) -> float:
    """Horizon adjustment factor for the effective sample size."""
    if gt_lag_minutes is None:
        return 1.0
    if not math.isfinite(gt_lag_minutes) or gt_lag_minutes <= 0:
        return 1.0
    return float(max(1.0, gt_lag_minutes / _LR_NEFF_ANCHOR_MINUTES) ** _LR_NEFF_POWER)


def _lr_adjusted_improvement_lower_bound(
    improvement: float, one_se_lower: float, scale: float
) -> float:
    """Improvement lower bound rescaled for the horizon and widened to 95%."""
    if not (math.isfinite(improvement) and math.isfinite(one_se_lower)):
        return float("nan")
    standard_error = improvement - one_se_lower
    if standard_error < 0:
        return float("nan")
    z = float(stats.norm.ppf(_LR_PROMOTION_CL))
    return float(improvement - z * standard_error / math.sqrt(scale))


def _lr_score_worker(
    true: np.ndarray,
    pred: np.ndarray,
    lags: Optional[np.ndarray],
    gt_ratio: int,
    n_active: int,
    total_nonces: int,
    *,
    horizon_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Score a worker against the seven ratified promotion criteria (RES-1578)."""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)


    keep = np.isfinite(true) & (true != 0)
    n_eff = _lr_compute_effective_sample_size(true[keep], _lr_mask_lags(lags, keep), gt_ratio)

    gt_lag_minutes = horizon_seconds / 60.0
    scale = _lr_neff_scale(gt_lag_minutes)

    nvalid, dir_acc, dir_acc_pval, dir_acc_ci = _lr_directional_accuracy_test(
        pred, true, n_eff * scale, cl=_LR_PROMOTION_CL,
    )
    pearson_r, pearson_ci, pearson_pval = _lr_correlation_test(
        pred, true, n_eff, cl=_LR_PROMOTION_CL,
    )
    wrmse_imp, wrmse_one_se = _lr_wrmse_improvement_and_ci(
        pred, true, n_eff, cl=_LR_ONE_SE_CL,
    )
    wczar_imp, wczar_one_se = _lr_wczar_improvement_and_ci(
        pred, true, n_eff, cl=_LR_ONE_SE_CL,
    )

    log_ar, log_ar_ci = _lr_log_aspect_ratio_and_ci(pred, true, n_eff, cl=_LR_ONE_SE_CL)

    finite = np.isfinite(pred) & np.isfinite(true)
    constant_prediction = bool(
        np.count_nonzero(finite) >= 3 and np.unique(pred[finite]).size <= 1
    )

    participation = min(1.0, n_active / total_nonces) if total_nonces > 0 else np.nan

    # Improvement bounds are tested in percent, with native one-SE bands
    # widened to one-sided 95% after adjusting for the prediction horizon.
    wrmse_lower = _lr_adjusted_improvement_lower_bound(
        _lr_pct(wrmse_imp), _lr_pct(wrmse_one_se[0]), scale,
    )
    wczar_lower = _lr_adjusted_improvement_lower_bound(
        _lr_pct(wczar_imp), _lr_pct(wczar_one_se[0]), scale,
    )
    checks = [
        ("n_eff", "Effective samples >= 20", n_eff,
         _LR_MIN_EFFECTIVE_SAMPLES, n_eff >= _LR_MIN_EFFECTIVE_SAMPLES),
        ("directional_accuracy_ci", "Directional accuracy lower CI > 50%",
         dir_acc_ci[0], _LR_LIM_DA_CI, dir_acc_ci[0] > _LR_LIM_DA_CI),
        ("correlation_ci", "Pearson correlation lower CI > 0%",
         pearson_ci[0], _LR_LIM_CORRELATION_CI, pearson_ci[0] > _LR_LIM_CORRELATION_CI),
        ("wrmse_ci", "WRMSE improvement lower CI > 0%",
         wrmse_lower, _LR_LIM_WRMSE_CI_PCT, wrmse_lower > _LR_LIM_WRMSE_CI_PCT),
        ("wczar_ci", "CZAR improvement lower CI > 0%",
         wczar_lower, _LR_LIM_WCZAR_CI_PCT, wczar_lower > _LR_LIM_WCZAR_CI_PCT),
        ("log_aspect_ratio", "Log aspect ratio CI overlaps +/- 0.5",
         log_ar, _LR_LIM_LOG_ASPECT_RATIO,
         np.isfinite(log_ar_ci).all()
         and log_ar_ci[1] >= -_LR_LIM_LOG_ASPECT_RATIO
         and log_ar_ci[0] <= _LR_LIM_LOG_ASPECT_RATIO),
        ("participation", "Participation > 90%",
         participation, _LR_LIM_PARTICIPATION, participation > _LR_LIM_PARTICIPATION),
    ]
    criteria = [
        dict(key=key, label=label, value=_lr_safe_float(value),
             threshold=threshold, passed=bool(np.isfinite(value) and passed))
        for key, label, value, threshold, passed in checks
    ]
    tested = {c["key"]: c["value"] for c in criteria}

    wrmse_ci = _lr_mirror_band(wrmse_imp, _lr_fraction(tested["wrmse_ci"]))
    wczar_ci = _lr_mirror_band(wczar_imp, _lr_fraction(tested["wczar_ci"]))

    descriptive = _lr_descriptive_stats(pred, true)

    return {
        "score": sum(c["passed"] for c in criteria),
        "max_score": len(criteria),
        "eligible": all(c["passed"] for c in criteria),
        "criteria": criteria,
        "n_eff": n_eff,
        "neff_scale": scale,
        "nvalid": nvalid,
        "dir_acc": dir_acc,
        "dir_acc_pval": dir_acc_pval,
        "dir_acc_ci": (tested["directional_accuracy_ci"], dir_acc_ci[1]),
        "pearson_r": pearson_r,
        "pearson_ci": pearson_ci,
        "pearson_pval": pearson_pval,
        "wrmse_imp": wrmse_imp,
        "wrmse_ci": wrmse_ci,
        "wczar_imp": wczar_imp,
        "wczar_ci": wczar_ci,
        "log_aspect_ratio": log_ar,
        "log_aspect_ratio_ci": log_ar_ci,
        "constant_prediction": constant_prediction,
        "participation": participation,
        "naive_annualized_return": _lr_naive_annualized_return(
            pred, true, horizon_seconds,
        ),
        **descriptive,
    }


def _lr_mask_lags(
    lags: Optional[np.ndarray], keep: np.ndarray
) -> Optional[np.ndarray]:
    """Re-derive inter-sample lags for the subset selected by *keep*."""
    if lags is None:
        return None
    lags = np.asarray(lags)
    if lags.size != keep.size - 1:

        return lags
    kept = np.flatnonzero(keep)
    if kept.size < 2:
        return np.array([1], dtype=int)

    cumulative = np.concatenate([[0], np.cumsum(lags)])
    return np.diff(cumulative[kept]).astype(int)


def _lr_pct(value: float) -> float:
    """Fraction to percent, preserving NaN."""
    return float(value) * 100.0 if value is not None and np.isfinite(value) else float("nan")


def _lr_fraction(percent: Optional[float]) -> Optional[float]:
    """Percent back to fraction, preserving a missing value."""
    return None if percent is None else float(percent) / 100.0


def _lr_mirror_band(point: float, lower: Optional[float]) -> Tuple[float, float]:
    """Symmetric band around *point* given its lower bound."""
    if lower is None or not np.isfinite(point) or not np.isfinite(lower):
        return (float("nan"), float("nan"))
    return (float(lower), float(2.0 * point - lower))


def _lr_descriptive_stats(pred: np.ndarray, true: np.ndarray) -> Dict[str, Any]:
    """Error and distribution summaries over the samples finite in both series."""
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return {
            "mae": None, "rmse": None,
            "mean_prediction": None, "mean_actual": None,
            "std_prediction": None, "std_actual": None,
            "normalized_mean_offset": None,
        }
    p = pred[mask]
    t = true[mask]
    std_t = float(np.std(t))
    return {
        "mae": float(np.mean(np.abs(t - p))),
        "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
        "mean_prediction": float(np.mean(p)),
        "mean_actual": float(np.mean(t)),
        "std_prediction": float(np.std(p)),
        "std_actual": std_t,
        "normalized_mean_offset": _lr_normalized_mean_offset(p, t),
    }


def _lr_naive_annualized_return(
    pred: np.ndarray,
    true: np.ndarray,
    horizon_seconds: Optional[float],
) -> Optional[float]:
    """Annualized return of a naive long/short strategy on the worker's signal."""
    if horizon_seconds is None or horizon_seconds <= 0:
        return None
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return None
    p = pred[mask]
    t = true[mask]
    log_return_for_trade = np.where(p > 0, t, np.where(p < 0, -t, 0))
    annualized = float(np.mean(log_return_for_trade)) / horizon_seconds * _LR_SECONDS_PER_YEAR
    try:
        result = math.exp(annualized) - 1
    except OverflowError:
        return None
    return _lr_safe_float(result)


def _lr_evaluate_log_returns(
    y_true, y_pred, epoch_length_minutes=60, n_expected_epochs=None,
    *, n_submitted=None, lags=None, gt_ratio=1, horizon_seconds=None,
):
    """Evaluate log returns; offline arrays default to full participation.

    Supply submission counts for live participation, and lags/gt_ratio for
    irregular or overlapping observations. The public score stays on 0..1.
    """
    true = np.asarray(y_true, dtype=float).flatten()
    pred = np.asarray(y_pred, dtype=float).flatten()
    if len(true) != len(pred):
        raise ValueError('y_true and y_pred must have same length')
    if not len(true):
        raise ValueError('y_true and y_pred cannot be empty')
    if not np.isfinite(epoch_length_minutes) or epoch_length_minutes <= 0:
        raise ValueError('epoch_length_minutes must be positive')
    if not np.isfinite(gt_ratio) or int(gt_ratio) != gt_ratio or gt_ratio < 1:
        raise ValueError('gt_ratio must be a positive integer')
    if lags is not None:
        lags = np.asarray(lags)
        if (lags.shape != (len(true) - 1,) or not np.isfinite(lags).all()
                or (lags < 1).any() or (lags != np.floor(lags)).any()):
            raise ValueError('lags must contain one positive integer gap per adjacent pair')
    if horizon_seconds is None:
        horizon_seconds = epoch_length_minutes * 60 * gt_ratio
    if not np.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError('horizon_seconds must be positive')

    assumed = n_expected_epochs is None
    if assumed and n_submitted is not None:
        raise ValueError('n_submitted requires n_expected_epochs')
    total = len(true) if assumed else n_expected_epochs
    submitted = len(true) if n_submitted is None else n_submitted
    if not np.isfinite(total) or int(total) != total or total <= 0:
        raise ValueError('n_expected_epochs must be positive integer')
    if (not np.isfinite(submitted) or int(submitted) != submitted
            or not 0 <= submitted <= total):
        raise ValueError('n_submitted must be between zero and n_expected_epochs')

    raw = _lr_score_worker(
        true, pred, lags, int(gt_ratio), int(submitted), int(total),
        horizon_seconds=horizon_seconds,
    )
    criteria = raw['criteria']
    metrics = {
        key: value for key, value in raw.items()
        if key not in ('criteria', 'score', 'max_score', 'eligible')
    }
    # Keep the builder kit's existing metric names alongside reference names.
    metrics.update(
        directional_accuracy=raw['dir_acc'],
        da_ci_lower=raw['dir_acc_ci'][0], da_ci_upper=raw['dir_acc_ci'][1],
        da_pvalue=raw['dir_acc_pval'], da_n_effective=raw['n_eff'],
        da_n_samples=raw['nvalid'], pearson_pvalue=raw['pearson_pval'],
        wrmse_improvement=raw['wrmse_imp'], czar_improvement=raw['wczar_imp'],
        mse=raw['rmse'] ** 2 if raw['rmse'] is not None else None,
    )
    count = raw['score']
    report = dict(
        target_type='log_return', metrics=metrics, criteria=criteria,
        passed={c['key']: c['passed'] for c in criteria},
        score=count / len(criteria), num_passed=count,
        num_primary_metrics=len(criteria), eligible=raw['eligible'],
        grade={7: 'A+', 6: 'A', 5: 'B+', 4: 'B', 3: 'C', 2: 'D', 1: 'F', 0: 'F'}[count],
        thresholds={c['key']: c['threshold'] for c in criteria},
        temporal_coverage_pass=(
            None if assumed else submitted / total > _LR_LIM_PARTICIPATION
        ),
        participation_basis=(
            'assumed full offline coverage' if assumed else 'provided submission counts'
        ),
    )

    def clean(value):
        """Reports must serialize as strict JSON, including undefined metrics."""
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    return clean(report)
