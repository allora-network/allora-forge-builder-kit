from __future__ import annotations
"""
Allora Forge Builder Kit - Performance Metrics Evaluation
==========================================================

Official metrics for log return predictions with comprehensive evaluation framework.

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

import numpy as np
from scipy import stats
import warnings


class PerformanceEvaluator:
    """
    Comprehensive performance metrics calculator for financial time-series predictions.
    
    Implements 7 primary metrics with pass/fail thresholds plus additional metrics
    for evaluating predictive model performance.
    
    v3.0 evaluation framework:
      - DA CI lower bound >= 0.50
      - DA threshold >= 0.52
      - DA p-value via z-test with continuity correction and
        autocorrelation-aware effective sample size
      - WRMSE improvement threshold >= 5%
      - CZAR (Cumulative Z-scored Absolute Return) improvement >= 10%
        replaces ZPTAE as primary metric
      - Log Aspect Ratio moved to additional (non-scored) metrics
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
        truth = PerformanceEvaluator.validate_probabilities(history_truth)
        if not np.isin(truth, [0, 1]).all():
            raise ValueError("Baseline history must be one-hot")
        times = pd.DatetimeIndex(pd.to_datetime(history_times, utc=True))
        queries = pd.DatetimeIndex(pd.to_datetime(prediction_times, utc=True))
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
        Calculate all performance metrics for log return predictions.
        
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
