"""
Research-grade statistical analysis for RAG experiments with multiple runs.

This module provides comprehensive statistical analysis including:
- Multiple run aggregation
- Effect size calculations (Cohen's d, η²)
- Statistical significance testing (ANOVA, t-tests)
- Confidence intervals
- Power analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
import pandas as pd

from src.analysis.statistics import StatisticalAnalyzer


@dataclass
class ResearchResult:
    """Aggregated research result from multiple runs"""
    experiment_name: str
    condition: str  # e.g., "position=middle" or "noise=0.8"

    # Aggregated metrics
    mean_accuracy: float
    std_accuracy: float
    ci_lower: float
    ci_upper: float

    # Sample statistics
    n_runs: int
    n_samples_per_run: int
    total_samples: int

    # Raw data
    accuracies_per_run: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Result of statistical comparison between conditions"""
    condition_a: str
    condition_b: str

    # Means
    mean_a: float
    mean_b: float
    difference: float

    # Effect size
    cohens_d: float
    effect_size_interpretation: str

    # Statistical test
    test_statistic: float
    p_value: float
    significant: bool

    # Confidence interval of difference
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ResearchAnalyzer:
    """
    Analyzer for research-grade statistical analysis across multiple experimental runs.

    Building Blocks Design:

    Setup Data:
        - confidence_level: float (default 0.95)
        - significance_alpha: float (default 0.05)
        - analyzer: StatisticalAnalyzer instance

    Input Data:
        - experiment_results: Dict with runs and measurements
        - condition_column: str (grouping variable)
        - metric_column: str (dependent variable)

    Output Data:
        - ResearchResult objects with aggregated statistics
        - ComparisonResult objects with effect sizes and tests
        - Summary DataFrames for tables and plots
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        significance_alpha: float = 0.05
    ):
        """Initialize research analyzer"""
        self.confidence_level = confidence_level
        self.significance_alpha = significance_alpha
        self.analyzer = StatisticalAnalyzer(confidence_level, significance_alpha)

    def aggregate_multiple_runs(
        self,
        results: List[Dict[str, Any]],
        condition_key: str,
        metric_key: str = 'correct'
    ) -> Dict[str, ResearchResult]:
        """
        Aggregate results from multiple runs by condition.

        Input Data:
            - results: List of result dicts from multiple runs
            - condition_key: Key to group by (e.g., 'position', 'noise_ratio')
            - metric_key: Key for the metric (e.g., 'correct')

        Output Data:
            - Dict mapping condition -> ResearchResult with statistics
        """
        # Group results by condition
        condition_data: Dict[str, List[bool]] = {}

        for result in results:
            condition = str(result.get(condition_key, 'unknown'))
            metric_value = result.get(metric_key, False)

            if condition not in condition_data:
                condition_data[condition] = []

            condition_data[condition].append(bool(metric_value))

        # Calculate statistics for each condition
        research_results = {}

        for condition, values in condition_data.items():
            # Convert to accuracy values (0 or 1)
            accuracy_values = [1.0 if v else 0.0 for v in values]

            # Calculate mean and CI
            mean_acc = np.mean(accuracy_values)
            std_acc = np.std(accuracy_values, ddof=1) if len(accuracy_values) > 1 else 0.0

            if len(accuracy_values) >= 2:
                mean_calc, ci_lower, ci_upper = self.analyzer.confidence_interval(accuracy_values)
            else:
                mean_calc = mean_acc
                ci_lower = mean_acc
                ci_upper = mean_acc

            research_results[condition] = ResearchResult(
                experiment_name="",
                condition=condition,
                mean_accuracy=mean_acc,
                std_accuracy=std_acc,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_runs=1,  # Will be updated if runs are tracked
                n_samples_per_run=len(accuracy_values),
                total_samples=len(accuracy_values),
                accuracies_per_run=[mean_acc]  # Single aggregated value
            )

        return research_results

    def aggregate_by_runs(
        self,
        all_runs_results: List[List[Dict[str, Any]]],
        condition_key: str,
        metric_key: str = 'correct'
    ) -> Dict[str, ResearchResult]:
        """
        Aggregate results from multiple independent runs.

        Each run should produce accuracy per condition.
        This computes mean accuracy across runs with proper CI.

        Input Data:
            - all_runs_results: List of runs, each run is list of result dicts
            - condition_key: Grouping variable
            - metric_key: Metric to measure

        Output Data:
            - Dict mapping condition -> ResearchResult with cross-run statistics
        """
        # First, compute accuracy per condition per run
        condition_accuracies: Dict[str, List[float]] = {}

        for run_results in all_runs_results:
            # Compute accuracy for each condition in this run
            condition_correct: Dict[str, List[bool]] = {}

            for result in run_results:
                condition = str(result.get(condition_key, 'unknown'))
                metric_value = result.get(metric_key, False)

                if condition not in condition_correct:
                    condition_correct[condition] = []

                condition_correct[condition].append(bool(metric_value))

            # Calculate accuracy for this run
            for condition, correct_list in condition_correct.items():
                accuracy = np.mean([1.0 if c else 0.0 for c in correct_list])

                if condition not in condition_accuracies:
                    condition_accuracies[condition] = []

                condition_accuracies[condition].append(accuracy)

        # Now aggregate across runs
        research_results = {}

        for condition, accuracies in condition_accuracies.items():
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0

            if len(accuracies) >= 2:
                mean_calc, ci_lower, ci_upper = self.analyzer.confidence_interval(accuracies)
            else:
                mean_calc = mean_acc
                ci_lower = mean_acc
                ci_upper = mean_acc

            research_results[condition] = ResearchResult(
                experiment_name="",
                condition=condition,
                mean_accuracy=mean_acc,
                std_accuracy=std_acc,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_runs=len(accuracies),
                n_samples_per_run=len(all_runs_results[0]) // len(condition_accuracies) if all_runs_results else 0,
                total_samples=sum(len(run) for run in all_runs_results),
                accuracies_per_run=accuracies
            )

        return research_results

    def compare_conditions(
        self,
        result_a: ResearchResult,
        result_b: ResearchResult
    ) -> ComparisonResult:
        """
        Statistical comparison between two conditions.

        Input Data:
            - result_a: ResearchResult for condition A
            - result_b: ResearchResult for condition B

        Output Data:
            - ComparisonResult with effect size and significance test
        """
        # Extract accuracy values
        accuracies_a = result_a.accuracies_per_run
        accuracies_b = result_b.accuracies_per_run

        # Calculate difference
        mean_diff = result_a.mean_accuracy - result_b.mean_accuracy

        # Calculate Cohen's d
        cohens_d = self.analyzer.cohens_d(accuracies_a, accuracies_b)
        effect_interpretation = self.analyzer.interpret_cohens_d(abs(cohens_d))

        # Perform t-test
        t_result = self.analyzer.t_test_independent(accuracies_a, accuracies_b)

        # Confidence interval of difference
        # Using pooled standard error
        se_a = result_a.std_accuracy / np.sqrt(result_a.n_runs) if result_a.n_runs > 0 else 0
        se_b = result_b.std_accuracy / np.sqrt(result_b.n_runs) if result_b.n_runs > 0 else 0
        se_diff = np.sqrt(se_a**2 + se_b**2)

        # t-critical value
        df = result_a.n_runs + result_b.n_runs - 2
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, df) if df > 0 else 1.96

        ci_lower_diff = mean_diff - t_crit * se_diff
        ci_upper_diff = mean_diff + t_crit * se_diff

        return ComparisonResult(
            condition_a=result_a.condition,
            condition_b=result_b.condition,
            mean_a=result_a.mean_accuracy,
            mean_b=result_b.mean_accuracy,
            difference=mean_diff,
            cohens_d=cohens_d,
            effect_size_interpretation=effect_interpretation,
            test_statistic=t_result['t_statistic'],
            p_value=t_result['p_value'],
            significant=t_result['significant'],
            ci_lower=ci_lower_diff,
            ci_upper=ci_upper_diff
        )

    def perform_anova(
        self,
        research_results: Dict[str, ResearchResult]
    ) -> Dict[str, Any]:
        """
        Perform one-way ANOVA across conditions.

        Input Data:
            - research_results: Dict mapping condition -> ResearchResult

        Output Data:
            - ANOVA result with F-statistic, p-value, eta-squared
        """
        # Prepare groups
        groups = {
            condition: result.accuracies_per_run
            for condition, result in research_results.items()
        }

        # Perform ANOVA
        anova_result = self.analyzer.one_way_anova(groups)

        # Calculate eta-squared (effect size for ANOVA)
        # η² = SS_between / SS_total
        all_values = []
        for values in groups.values():
            all_values.extend(values)

        grand_mean = np.mean(all_values)

        ss_between = sum(
            len(values) * (np.mean(values) - grand_mean)**2
            for values in groups.values()
        )

        ss_total = sum((x - grand_mean)**2 for x in all_values)

        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Interpret eta-squared
        # Small: 0.01, Medium: 0.06, Large: 0.14
        if eta_squared < 0.01:
            eta_interpretation = "negligible"
        elif eta_squared < 0.06:
            eta_interpretation = "small"
        elif eta_squared < 0.14:
            eta_interpretation = "medium"
        else:
            eta_interpretation = "large"

        return {
            **anova_result,
            'eta_squared': eta_squared,
            'eta_squared_interpretation': eta_interpretation,
            'n_groups': len(groups),
            'total_n': len(all_values)
        }

    def create_summary_table(
        self,
        research_results: Dict[str, ResearchResult],
        sort_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create summary table for publication.

        Input Data:
            - research_results: Dict of research results
            - sort_by: Optional column to sort by

        Output Data:
            - pandas DataFrame ready for export/display
        """
        rows = []

        for condition, result in research_results.items():
            rows.append({
                'Condition': condition,
                'Mean Accuracy': f"{result.mean_accuracy:.3f}",
                'Std Dev': f"{result.std_accuracy:.3f}",
                '95% CI': f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}]",
                'N Runs': result.n_runs,
                'Total Samples': result.total_samples
            })

        df = pd.DataFrame(rows)

        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by)

        return df

    def save_results(
        self,
        research_results: Dict[str, ResearchResult],
        output_path: Path,
        include_raw_data: bool = True
    ):
        """
        Save research results to JSON.

        Input Data:
            - research_results: Dict of results
            - output_path: Where to save
            - include_raw_data: Whether to include raw run data

        Output Data:
            - JSON file with structured results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metadata': {
                'confidence_level': self.confidence_level,
                'significance_alpha': self.significance_alpha
            },
            'results': {}
        }

        for condition, result in research_results.items():
            result_dict = result.to_dict()

            if not include_raw_data:
                result_dict.pop('accuracies_per_run', None)

            data['results'][condition] = result_dict

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def calculate_power_analysis(
        self,
        effect_size: float,
        n_per_group: int,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate statistical power for given effect size and sample size.

        Input Data:
            - effect_size: Expected Cohen's d
            - n_per_group: Sample size per group
            - alpha: Significance level

        Output Data:
            - Dict with power and required N for 0.80 power
        """
        from scipy.stats import norm

        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n_per_group / 2)

        # Critical value for two-tailed test
        z_alpha = norm.ppf(1 - alpha / 2)

        # Power calculation
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)

        # Required N for 80% power
        target_power = 0.80
        z_beta = norm.ppf(target_power)
        required_n = ((z_alpha + z_beta) / effect_size) ** 2 * 2

        return {
            'effect_size': effect_size,
            'n_per_group': n_per_group,
            'alpha': alpha,
            'power': power,
            'required_n_for_80_power': int(np.ceil(required_n))
        }
