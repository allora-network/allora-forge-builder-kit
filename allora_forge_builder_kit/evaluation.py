"""
Allora Forge Builder Kit - Performance Metrics Evaluation
==========================================================

Official metrics for log return predictions, aligned with Research team's framework.
Reference: Linear issue ENGN-4244 and research repo RES-1087

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
from typing import Dict, Tuple, Optional
import warnings


class PerformanceEvaluator:
    """
    Comprehensive performance metrics calculator for financial time-series predictions.
    
    Implements 8 primary metrics with pass/fail thresholds plus additional metrics.
    Aligned with Research team's implementation in:
    - research_notebooks/financial_predictions/evaluation/metrics.py
    - PR #178 (RES-1087)
    """
    
    # Primary metric thresholds (8 core metrics)
    THRESHOLDS = {
        'directional_accuracy': 0.55,
        'da_ci_lower': 0.52,
        'da_pvalue': 0.05,  # Must be BELOW this
        'pearson_r': 0.05,
        'pearson_pvalue': 0.05,  # Must be BELOW this
        'wrmse_improvement': 0.10,
        'zptae_improvement': 0.20,
        'log_aspect_ratio_abs': 0.5,  # |value| must be BELOW this
    }
    
    # Performance grades based on how many primary metrics pass
    GRADES = {
        8: 'A+',
        7: 'A',
        6: 'B+',
        5: 'B',
        4: 'C',
        3: 'D',
        2: 'F',
        1: 'F',
        0: 'F',
    }
    
    def __init__(self):
        """Initialize the performance evaluator."""
        pass
    
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
    ) -> Dict[str, float]:
        """
        Calculate directional accuracy and related metrics.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with DA, confidence interval, and p-value
        """
        n = len(y_true)
        
        # Directional accuracy: % of predictions with correct sign
        correct_direction = np.sign(y_true) == np.sign(y_pred)
        da = np.mean(correct_direction)
        
        # Binomial test: is DA significantly better than random (0.5)?
        # H0: DA = 0.5 (random guessing)
        # H1: DA > 0.5 (better than random)
        n_correct = np.sum(correct_direction)
        binomial_test = stats.binomtest(n_correct, n, 0.5, alternative='greater')
        da_pvalue = binomial_test.pvalue
        
        # Confidence interval for DA (95% confidence level)
        # Using Wilson score interval (better than normal approximation)
        z = 1.96  # 95% confidence
        p_hat = da
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) / n + z**2 / (4*n**2))) / denominator
        
        da_ci_lower = center - margin
        da_ci_upper = center + margin
        
        return {
            'directional_accuracy': da,
            'da_ci_lower': da_ci_lower,
            'da_ci_upper': da_ci_upper,
            'da_pvalue': da_pvalue,
            'da_n_samples': n,
            'da_n_correct': int(n_correct),
        }
    
    def calculate_correlation_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Pearson correlation and related metrics.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with Pearson r, p-value, and related metrics
        """
        # Pearson correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson_r, pearson_pvalue = stats.pearsonr(y_pred, y_true)
        
        # Handle NaN (can happen if all predictions are identical)
        if np.isnan(pearson_r):
            pearson_r = 0.0
            pearson_pvalue = 1.0
        
        # Spearman correlation (rank-based, more robust to outliers)
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
    ) -> Dict[str, float]:
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
        
        # Avoid division by zero
        if weights_sum == 0:
            return {
                'wrmse_model': 0.0,
                'wrmse_baseline': 0.0,
                'wrmse_improvement': 0.0,
            }
        
        # WRMSE for model
        squared_errors = (y_true - y_pred) ** 2
        wrmse_model = np.sqrt(np.sum(weights * squared_errors) / weights_sum)
        
        # WRMSE for baseline (always predict 0)
        baseline_squared_errors = y_true ** 2
        wrmse_baseline = np.sqrt(np.sum(weights * baseline_squared_errors) / weights_sum)
        
        # Relative improvement
        if wrmse_baseline > 0:
            wrmse_improvement = (wrmse_baseline - wrmse_model) / wrmse_baseline
        else:
            wrmse_improvement = 0.0
        
        return {
            'wrmse_model': wrmse_model,
            'wrmse_baseline': wrmse_baseline,
            'wrmse_improvement': wrmse_improvement,
        }
    
    def calculate_zptae_improvement(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Z-transformed Power-Tanh Absolute Error improvement.
        
        ZPTAE is a custom loss function that:
        1. Z-scores the data (normalize by std)
        2. Applies power-tanh transformation (robust to outliers)
        3. Weights by absolute true value
        
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
        
        # Weights by absolute true value
        weights = np.abs(y_true)
        weights_sum = np.sum(weights)
        
        if weights_sum == 0:
            return {
                'zptae_model': 0.0,
                'zptae_baseline': 0.0,
                'zptae_improvement': 0.0,
            }
        
        # Z-transform and apply power-tanh
        z_true = y_true / stdev
        z_pred = y_pred / stdev
        z_baseline = np.zeros_like(y_true)
        
        # ZPTAE for model
        pt_diff_model = np.abs(
            self.power_tanh(z_true) - self.power_tanh(z_pred)
        )
        zptae_model = np.sum(weights * pt_diff_model) / weights_sum
        
        # ZPTAE for baseline (always predict 0)
        pt_diff_baseline = np.abs(
            self.power_tanh(z_true) - self.power_tanh(z_baseline)
        )
        zptae_baseline = np.sum(weights * pt_diff_baseline) / weights_sum
        
        # Relative improvement
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
    ) -> Dict[str, float]:
        """
        Calculate log aspect ratio: log10(std(predicted) / std(actual)).
        
        Measures if the model's predictions have appropriate variance.
        - log_aspect_ratio = 0: perfect variance matching
        - log_aspect_ratio > 0: over-confident (too much variance)
        - log_aspect_ratio < 0: under-confident (too little variance)
        
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
    ) -> Dict[str, float]:
        """
        Calculate naive annualized return from a simple trading strategy.
        
        Strategy: Go long if prediction is positive, short if negative.
        Return = sum(|true_return| where direction is correct) - 
                 sum(|true_return| where direction is wrong)
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            epoch_length_minutes: Length of each prediction period in minutes
            
        Returns:
            Dictionary with return metrics
        """
        # Same sign = correct direction
        same_sign = np.sign(y_true) == np.sign(y_pred)
        
        # Calculate cumulative return
        naive_return = (
            np.sum(same_sign * np.abs(y_true)) - 
            np.sum(~same_sign * np.abs(y_true))
        )
        
        # Annualize
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
    ) -> Dict[str, float]:
        """
        Calculate standard regression metrics.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with MAE, MSE, RMSE, R², MAPE
        """
        errors = y_true - y_pred
        
        # Basic metrics
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        
        # R² (coefficient of determination)
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # MAPE (Mean Absolute Percentage Error)
        # Only calculate for non-zero true values
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
    ) -> Dict[str, float]:
        """
        Calculate classification metrics for directional predictions.
        
        Args:
            y_true: Ground truth log returns
            y_pred: Predicted log returns
            
        Returns:
            Dictionary with precision, recall, F1, specificity
        """
        # Convert to binary: 1 = up (positive), 0 = down (negative or zero)
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        
        # Confusion matrix
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall (Sensitivity): TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Specificity: TN / (TN + FP)
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
    
    def check_primary_metrics_pass(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Check which of the 8 primary metrics pass their thresholds.
        
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
            'zptae_improvement_pass': metrics['zptae_improvement'] >= self.THRESHOLDS['zptae_improvement'],
            'log_aspect_ratio_pass': np.abs(metrics['log_aspect_ratio']) < self.THRESHOLDS['log_aspect_ratio_abs'],
        }
    
    def calculate_performance_score(self, passed: Dict[str, bool]) -> Tuple[float, str, int]:
        """
        Calculate overall performance score and grade.
        
        Args:
            passed: Dictionary of pass/fail for primary metrics
            
        Returns:
            Tuple of (score, grade, num_passed)
        """
        num_passed = sum(passed.values())
        score = num_passed / 8.0
        grade = self.GRADES.get(num_passed, 'F')
        
        return score, grade, num_passed
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epoch_length_minutes: int = 60
    ) -> Dict:
        """
        Calculate all performance metrics for log return predictions.
        
        Args:
            y_true: Ground truth log returns (actual)
            y_pred: Predicted log returns
            epoch_length_minutes: Length of each prediction epoch in minutes
            
        Returns:
            Comprehensive dictionary with all metrics, pass/fail, and grade
        """
        # Convert to numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred must have same length. Got {len(y_true)} and {len(y_pred)}")
        
        if len(y_true) == 0:
            raise ValueError("y_true and y_pred cannot be empty")
        
        # Calculate all metrics
        metrics = {}
        
        # 8 Primary Metrics
        metrics.update(self.calculate_directional_metrics(y_true, y_pred))
        metrics.update(self.calculate_correlation_metrics(y_true, y_pred))
        metrics.update(self.calculate_wrmse_improvement(y_true, y_pred))
        metrics.update(self.calculate_zptae_improvement(y_true, y_pred))
        metrics.update(self.calculate_aspect_ratio(y_true, y_pred))
        
        # Additional Metrics
        metrics.update(self.calculate_naive_annualized_return(y_true, y_pred, epoch_length_minutes))
        metrics.update(self.calculate_regression_metrics(y_true, y_pred))
        metrics.update(self.calculate_classification_metrics(y_true, y_pred))
        
        # Check pass/fail
        passed = self.check_primary_metrics_pass(metrics)
        score, grade, num_passed = self.calculate_performance_score(passed)
        
        # Compile final report
        report = {
            'metrics': metrics,
            'passed': passed,
            'score': score,
            'grade': grade,
            'num_passed': num_passed,
            'thresholds': self.THRESHOLDS.copy(),
        }
        
        return report
    
    def print_report(self, report: Dict, detailed: bool = True):
        """
        Print a formatted performance report.
        
        Args:
            report: Output from evaluate()
            detailed: If True, show all metrics. If False, show only primary metrics.
        """
        m = report['metrics']
        p = report['passed']
        
        print("=" * 80)
        print(f"PERFORMANCE EVALUATION REPORT")
        print("=" * 80)
        print(f"\n📊 OVERALL PERFORMANCE: {report['grade']} ({report['num_passed']}/8 metrics passed)")
        print(f"   Performance Score: {report['score']:.2%}\n")
        
        print("=" * 80)
        print("🎯 PRIMARY METRICS (8 Core Metrics)")
        print("=" * 80)
        
        # Directional Accuracy
        print(f"\n1. Directional Accuracy:")
        print(f"   Value: {m['directional_accuracy']:.4f}  {'✅ PASS' if p['da_pass'] else '❌ FAIL'}")
        print(f"   Threshold: ≥ {self.THRESHOLDS['directional_accuracy']}")
        print(f"   Correct: {m['da_n_correct']}/{m['da_n_samples']} predictions")
        
        print(f"\n2. DA Confidence Interval (95%):")
        print(f"   Lower: {m['da_ci_lower']:.4f}  {'✅ PASS' if p['da_ci_pass'] else '❌ FAIL'}")
        print(f"   Upper: {m['da_ci_upper']:.4f}")
        print(f"   Threshold: Lower ≥ {self.THRESHOLDS['da_ci_lower']}")
        
        print(f"\n3. DA Statistical Significance:")
        print(f"   p-value: {m['da_pvalue']:.4f}  {'✅ PASS' if p['da_pvalue_pass'] else '❌ FAIL'}")
        print(f"   Threshold: < {self.THRESHOLDS['da_pvalue']}")
        
        print(f"\n4. Pearson Correlation:")
        print(f"   r: {m['pearson_r']:.4f}  {'✅ PASS' if p['pearson_r_pass'] else '❌ FAIL'}")
        print(f"   Threshold: ≥ {self.THRESHOLDS['pearson_r']}")
        
        print(f"\n5. Pearson Statistical Significance:")
        print(f"   p-value: {m['pearson_pvalue']:.4f}  {'✅ PASS' if p['pearson_pvalue_pass'] else '❌ FAIL'}")
        print(f"   Threshold: < {self.THRESHOLDS['pearson_pvalue']}")
        
        print(f"\n6. WRMSE Improvement:")
        print(f"   Improvement: {m['wrmse_improvement']:.4f} ({m['wrmse_improvement']:.2%})  {'✅ PASS' if p['wrmse_improvement_pass'] else '❌ FAIL'}")
        print(f"   Threshold: ≥ {self.THRESHOLDS['wrmse_improvement']} (10%)")
        print(f"   Model WRMSE: {m['wrmse_model']:.6f}")
        print(f"   Baseline WRMSE: {m['wrmse_baseline']:.6f}")
        
        print(f"\n7. ZPTAE Improvement:")
        print(f"   Improvement: {m['zptae_improvement']:.4f} ({m['zptae_improvement']:.2%})  {'✅ PASS' if p['zptae_improvement_pass'] else '❌ FAIL'}")
        print(f"   Threshold: ≥ {self.THRESHOLDS['zptae_improvement']} (20%)")
        print(f"   Model ZPTAE: {m['zptae_model']:.6f}")
        print(f"   Baseline ZPTAE: {m['zptae_baseline']:.6f}")
        
        print(f"\n8. Log Aspect Ratio:")
        print(f"   Value: {m['log_aspect_ratio']:.4f}  {'✅ PASS' if p['log_aspect_ratio_pass'] else '❌ FAIL'}")
        print(f"   Threshold: |value| < {self.THRESHOLDS['log_aspect_ratio_abs']}")
        print(f"   Std(true): {m['std_true']:.6f}")
        print(f"   Std(pred): {m['std_pred']:.6f}")
        
        if detailed:
            print("\n" + "=" * 80)
            print("📈 ADDITIONAL METRICS")
            print("=" * 80)
            
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



