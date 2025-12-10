"""
Statistical Analysis Module

Provides statistical tests and effect size calculations:
- Descriptive statistics with confidence intervals
- ANOVA for comparing groups
- Cohen's d for effect sizes
- Statistical significance testing
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Statistical analysis for experiment results

    Setup Data:
    - confidence_level: float - confidence level (default 0.95)
    - significance_alpha: float - significance threshold (default 0.05)

    Input Data:
    - values: List[float] - numerical values to analyze
    - groups: Dict[str, List[float]] - grouped data for comparison

    Output Data:
    - Dict with statistics: mean, std, ci_lower, ci_upper, etc.
    """

    def __init__(self, confidence_level: float = 0.95, significance_alpha: float = 0.05):
        """
        Initialize statistical analyzer

        Args:
            confidence_level: Confidence level for intervals (0.95 = 95%)
            significance_alpha: Significance threshold (0.05 = 5%)
        """
        self.confidence_level = confidence_level
        self.significance_alpha = significance_alpha

        logger.info(
            f"Initialized StatisticalAnalyzer "
            f"(confidence={confidence_level}, alpha={significance_alpha})"
        )

    def descriptive_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate descriptive statistics

        Args:
            values: List of numerical values

        Returns:
            Dictionary with:
            - mean: float
            - median: float
            - std: float
            - min: float
            - max: float
            - count: int

        Raises:
            ValueError: If values is empty
        """
        if not values:
            raise ValueError("Values list cannot be empty")

        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }

    def confidence_interval(
        self,
        values: List[float],
        confidence_level: float = None
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for mean

        Formula: CI = x̄ ± t_(α/2,n-1) · σ/√n

        Args:
            values: List of numerical values
            confidence_level: Override default confidence level

        Returns:
            Tuple of (mean, ci_lower, ci_upper)

        Raises:
            ValueError: If values is empty or has only one element
        """
        if not values:
            raise ValueError("Values list cannot be empty")

        if len(values) == 1:
            raise ValueError("Need at least 2 values for confidence interval")

        conf_level = confidence_level if confidence_level is not None else self.confidence_level

        n = len(values)
        mean = np.mean(values)
        std_err = stats.sem(values)  # Standard error of the mean
        df = n - 1  # Degrees of freedom

        # t-value for confidence level
        t_value = stats.t.ppf((1 + conf_level) / 2, df)

        # Confidence interval
        margin = t_value * std_err
        ci_lower = mean - margin
        ci_upper = mean + margin

        logger.debug(
            f"CI_{int(conf_level*100)}%: "
            f"{mean:.4f} ± {margin:.4f} = [{ci_lower:.4f}, {ci_upper:.4f}]"
        )

        return float(mean), float(ci_lower), float(ci_upper)

    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size

        Formula: d = (μ₁ - μ₂) / σ_pooled
        where σ_pooled = √[(σ₁² + σ₂²) / 2]

        Args:
            group1: First group values
            group2: Second group values

        Returns:
            Cohen's d effect size

        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large

        Raises:
            ValueError: If either group is empty
        """
        if not group1 or not group2:
            raise ValueError("Both groups must have values")

        mean1 = np.mean(group1)
        mean2 = np.mean(group2)

        std1 = np.std(group1, ddof=1) if len(group1) > 1 else 0.0
        std2 = np.std(group2, ddof=1) if len(group2) > 1 else 0.0

        # Pooled standard deviation
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)

        if pooled_std == 0:
            logger.warning("Pooled std is 0, returning 0 for Cohen's d")
            return 0.0

        d = (mean1 - mean2) / pooled_std

        logger.debug(f"Cohen's d = {d:.4f}")

        return float(d)

    def interpret_cohens_d(self, d: float) -> str:
        """
        Interpret Cohen's d effect size

        Args:
            d: Cohen's d value

        Returns:
            Interpretation string
        """
        abs_d = abs(d)

        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def one_way_anova(self, groups: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform one-way ANOVA

        Tests null hypothesis that all groups have the same mean.

        Args:
            groups: Dictionary mapping group names to value lists

        Returns:
            Dictionary with:
            - f_statistic: float
            - p_value: float
            - significant: bool
            - df_between: int
            - df_within: int

        Raises:
            ValueError: If groups has fewer than 2 groups
        """
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for ANOVA")

        group_values = list(groups.values())

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*group_values)

        # Degrees of freedom
        k = len(groups)  # Number of groups
        n = sum(len(g) for g in group_values)  # Total sample size
        df_between = k - 1
        df_within = n - k

        significant = p_value < self.significance_alpha

        logger.info(
            f"ANOVA: F({df_between},{df_within}) = {f_stat:.4f}, "
            f"p = {p_value:.4f}, significant = {significant}"
        )

        return {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': bool(significant),
            'df_between': df_between,
            'df_within': df_within,
            'interpretation': 'significant' if significant else 'not significant'
        }

    def t_test_independent(
        self,
        group1: List[float],
        group2: List[float]
    ) -> Dict[str, Any]:
        """
        Perform independent samples t-test

        Args:
            group1: First group values
            group2: Second group values

        Returns:
            Dictionary with:
            - t_statistic: float
            - p_value: float
            - significant: bool
            - df: int
        """
        if not group1 or not group2:
            raise ValueError("Both groups must have values")

        t_stat, p_value = stats.ttest_ind(group1, group2)

        significant = p_value < self.significance_alpha

        logger.info(
            f"t-test: t = {t_stat:.4f}, p = {p_value:.4f}, "
            f"significant = {significant}"
        )

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(significant),
            'df': len(group1) + len(group2) - 2
        }

    def correlation(
        self,
        x: List[float],
        y: List[float],
        method: str = 'pearson'
    ) -> Dict[str, float]:
        """
        Calculate correlation coefficient

        Args:
            x: First variable
            y: Second variable
            method: 'pearson' or 'spearman'

        Returns:
            Dictionary with:
            - correlation: float
            - p_value: float
            - significant: bool

        Raises:
            ValueError: If x and y have different lengths
        """
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        if method == 'pearson':
            corr, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(x, y)
        else:
            raise ValueError(f"Unknown method: {method}")

        significant = p_value < self.significance_alpha

        logger.debug(
            f"{method.capitalize()} correlation: r = {corr:.4f}, "
            f"p = {p_value:.4f}"
        )

        return {
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant': bool(significant)
        }

    def summary_statistics(
        self,
        values: List[float],
        label: str = "Sample"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics

        Args:
            values: List of numerical values
            label: Label for this sample

        Returns:
            Dictionary with all statistics
        """
        if not values:
            raise ValueError("Values list cannot be empty")

        stats_dict = {
            'label': label,
            **self.descriptive_stats(values)
        }

        # Add confidence interval if possible
        if len(values) > 1:
            mean, ci_lower, ci_upper = self.confidence_interval(values)
            stats_dict['ci_lower'] = ci_lower
            stats_dict['ci_upper'] = ci_upper
            stats_dict['ci_margin'] = (ci_upper - ci_lower) / 2

        return stats_dict
