"""
Publication-quality visualizations for RAG research with statistical annotations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd

from src.analysis.research_analysis import ResearchResult, ComparisonResult


class ResearchVisualizer:
    """
    Create publication-quality visualizations with statistical annotations.

    Building Blocks Design:

    Setup Data:
        - style: str (seaborn style)
        - dpi: int (resolution)
        - figure_size: Tuple[int, int]
        - palette: str (color palette)

    Input Data:
        - research_results: Dict[str, ResearchResult]
        - comparison_results: List[ComparisonResult]
        - anova_results: Dict with F-statistic and p-value

    Output Data:
        - Saved figure files (PNG, PDF)
        - matplotlib Figure objects for notebooks
    """

    def __init__(
        self,
        style: str = 'whitegrid',
        dpi: int = 300,
        figure_size: Tuple[int, int] = (10, 6),
        palette: str = 'Set2'
    ):
        """Initialize visualizer"""
        self.style = style
        self.dpi = dpi
        self.figure_size = figure_size
        self.palette = palette

        # Set style
        sns.set_style(style)
        sns.set_palette(palette)

        # Configure matplotlib for publication quality
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16

    def plot_accuracy_with_ci(
        self,
        research_results: Dict[str, ResearchResult],
        title: str,
        xlabel: str,
        ylabel: str = "Accuracy",
        output_path: Optional[Path] = None,
        show_values: bool = True,
        sort_conditions: bool = False
    ) -> plt.Figure:
        """
        Plot accuracy with confidence intervals.

        Input Data:
            - research_results: Results to plot
            - title, xlabel, ylabel: Labels
            - output_path: Where to save
            - show_values: Annotate with values
            - sort_conditions: Sort by condition name

        Output Data:
            - matplotlib Figure
            - Saved PNG/PDF if output_path provided
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Prepare data
        conditions = list(research_results.keys())
        if sort_conditions:
            conditions = sorted(conditions)

        means = [research_results[c].mean_accuracy for c in conditions]
        ci_lowers = [research_results[c].ci_lower for c in conditions]
        ci_uppers = [research_results[c].ci_upper for c in conditions]

        # Calculate error bars
        errors_lower = [means[i] - ci_lowers[i] for i in range(len(means))]
        errors_upper = [ci_uppers[i] - means[i] for i in range(len(means))]

        # Create bar plot
        x_pos = np.arange(len(conditions))
        bars = ax.bar(x_pos, means, yerr=[errors_lower, errors_upper],
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Customize
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Chance Level')

        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Annotate values
        if show_values:
            for i, (bar, mean, ci_lower, ci_upper) in enumerate(zip(bars, means, ci_lowers, ci_uppers)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + errors_upper[i] + 0.02,
                       f'{mean:.3f}\n[{ci_lower:.2f}, {ci_upper:.2f}]',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.legend()
        plt.tight_layout()

        # Save
        if output_path:
            self._save_figure(fig, output_path)

        return fig

    def plot_line_with_ci(
        self,
        research_results: Dict[str, ResearchResult],
        title: str,
        xlabel: str,
        ylabel: str = "Accuracy",
        output_path: Optional[Path] = None,
        x_numeric: bool = True,
        reference_line: Optional[float] = None
    ) -> plt.Figure:
        """
        Plot line chart with confidence interval shading.

        Input Data:
            - research_results: Results with ordered conditions
            - title, labels: Plot labels
            - x_numeric: Whether x-axis is numeric (vs categorical)
            - reference_line: Optional horizontal reference line

        Output Data:
            - matplotlib Figure with shaded CI region
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Prepare data
        conditions = list(research_results.keys())

        # Sort by condition if numeric
        if x_numeric:
            try:
                conditions = sorted(conditions, key=lambda x: float(x))
            except (ValueError, TypeError):
                pass

        means = [research_results[c].mean_accuracy for c in conditions]
        ci_lowers = [research_results[c].ci_lower for c in conditions]
        ci_uppers = [research_results[c].ci_upper for c in conditions]

        if x_numeric:
            try:
                x_values = [float(c) for c in conditions]
            except (ValueError, TypeError):
                x_values = list(range(len(conditions)))
        else:
            x_values = list(range(len(conditions)))

        # Plot line with markers
        ax.plot(x_values, means, marker='o', linewidth=2, markersize=8,
               label='Mean Accuracy', color='steelblue')

        # Shade confidence interval
        ax.fill_between(x_values, ci_lowers, ci_uppers, alpha=0.2,
                        color='steelblue', label='95% CI')

        # Reference line
        if reference_line is not None:
            ax.axhline(y=reference_line, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.7, label=f'Reference ({reference_line:.2f})')

        # Customize
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)

        if not x_numeric:
            ax.set_xticks(x_values)
            ax.set_xticklabels(conditions, rotation=45, ha='right')

        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()

        if output_path:
            self._save_figure(fig, output_path)

        return fig

    def plot_comparison_heatmap(
        self,
        comparison_results: List[ComparisonResult],
        title: str,
        output_path: Optional[Path] = None,
        metric: str = 'cohens_d'
    ) -> plt.Figure:
        """
        Create heatmap of pairwise comparisons.

        Input Data:
            - comparison_results: List of pairwise comparisons
            - metric: 'cohens_d' or 'p_value'

        Output Data:
            - Heatmap figure
        """
        # Extract unique conditions
        conditions = set()
        for comp in comparison_results:
            conditions.add(comp.condition_a)
            conditions.add(comp.condition_b)

        conditions = sorted(list(conditions))
        n = len(conditions)

        # Create matrix
        matrix = np.zeros((n, n))
        annot_matrix = np.empty((n, n), dtype=object)

        for comp in comparison_results:
            i = conditions.index(comp.condition_a)
            j = conditions.index(comp.condition_b)

            if metric == 'cohens_d':
                value = comp.cohens_d
                annot = f"{value:.2f}\n{comp.effect_size_interpretation}"
            elif metric == 'p_value':
                value = comp.p_value
                annot = f"p={value:.3f}\n{'*' if comp.significant else 'ns'}"
            else:
                value = comp.difference
                annot = f"{value:.3f}"

            matrix[i, j] = value
            matrix[j, i] = -value if metric != 'p_value' else value

            annot_matrix[i, j] = annot
            if metric != 'p_value':
                annot_matrix[j, i] = f"{-value:.2f}\n{comp.effect_size_interpretation}"
            else:
                annot_matrix[j, i] = annot

        # Create heatmap
        fig, ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[0]))

        if metric == 'cohens_d':
            cmap = 'RdBu_r'
            vmin, vmax = -2, 2
        elif metric == 'p_value':
            cmap = 'RdYlGn_r'
            vmin, vmax = 0, 0.1
        else:
            cmap = 'coolwarm'
            vmin, vmax = None, None

        sns.heatmap(matrix, annot=annot_matrix, fmt='', cmap=cmap,
                   center=0 if metric != 'p_value' else None,
                   vmin=vmin, vmax=vmax,
                   xticklabels=conditions, yticklabels=conditions,
                   cbar_kws={'label': metric.replace('_', ' ').title()},
                   linewidths=0.5, ax=ax)

        ax.set_title(title, fontweight='bold', pad=20)
        plt.tight_layout()

        if output_path:
            self._save_figure(fig, output_path)

        return fig

    def plot_effect_sizes(
        self,
        comparison_results: List[ComparisonResult],
        title: str,
        output_path: Optional[Path] = None,
        reference_lines: bool = True
    ) -> plt.Figure:
        """
        Forest plot of effect sizes with confidence intervals.

        Input Data:
            - comparison_results: List of comparisons

        Output Data:
            - Forest plot figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Sort by effect size
        comparisons = sorted(comparison_results, key=lambda x: abs(x.cohens_d), reverse=True)

        labels = [f"{c.condition_a} vs {c.condition_b}" for c in comparisons]
        effect_sizes = [c.cohens_d for c in comparisons]

        # For CI, use difference CI as proxy (not exact for Cohen's d)
        # In practice, you'd calculate proper CI for Cohen's d
        ci_errors = [(abs(c.ci_upper - c.ci_lower) / 2) / (abs(c.mean_a - c.mean_b) / abs(c.cohens_d) if c.cohens_d != 0 else 1)
                    for c in comparisons]

        # Plot
        y_pos = np.arange(len(labels))
        colors = ['red' if c.significant else 'gray' for c in comparisons]

        ax.errorbar(effect_sizes, y_pos, xerr=ci_errors,
                   fmt='o', markersize=8, capsize=5, elinewidth=2,
                   color='black', ecolor='black', alpha=0.7)

        for i, (es, color, comp) in enumerate(zip(effect_sizes, colors, comparisons)):
            ax.plot(es, i, 'o', markersize=10, color=color, alpha=0.8)
            # Add significance annotation
            ax.text(es + 0.1, i, f"p={comp.p_value:.3f}", va='center', fontsize=8)

        # Reference lines
        if reference_lines:
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            ax.axvline(x=0.2, color='green', linestyle='--', linewidth=1, alpha=0.3, label='Small (0.2)')
            ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.3, label='Medium (0.5)')
            ax.axvline(x=0.8, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Large (0.8)')
            ax.axvline(x=-0.2, color='green', linestyle='--', linewidth=1, alpha=0.3)
            ax.axvline(x=-0.5, color='orange', linestyle='--', linewidth=1, alpha=0.3)
            ax.axvline(x=-0.8, color='red', linestyle='--', linewidth=1, alpha=0.3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Cohen's d (Effect Size)", fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()

        if output_path:
            self._save_figure(fig, output_path)

        return fig

    def plot_multi_experiment_comparison(
        self,
        experiment_data: Dict[str, Dict[str, ResearchResult]],
        title: str,
        xlabel: str,
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Compare results across multiple experiments.

        Input Data:
            - experiment_data: Dict mapping experiment_name -> research_results
            - Labels for plot

        Output Data:
            - Multi-line comparison figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        colors = sns.color_palette(self.palette, len(experiment_data))

        for (exp_name, results), color in zip(experiment_data.items(), colors):
            conditions = sorted(list(results.keys()))

            try:
                x_values = [float(c) for c in conditions]
            except (ValueError, TypeError):
                x_values = list(range(len(conditions)))

            means = [results[c].mean_accuracy for c in conditions]
            ci_lowers = [results[c].ci_lower for c in conditions]
            ci_uppers = [results[c].ci_upper for c in conditions]

            ax.plot(x_values, means, marker='o', linewidth=2, markersize=8,
                   label=exp_name, color=color)
            ax.fill_between(x_values, ci_lowers, ci_uppers, alpha=0.15, color=color)

        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()

        if output_path:
            self._save_figure(fig, output_path)

        return fig

    def create_statistical_summary_figure(
        self,
        anova_result: Dict[str, Any],
        comparisons: List[ComparisonResult],
        title: str,
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive statistical summary figure.

        Input Data:
            - anova_result: ANOVA results dict
            - comparisons: List of pairwise comparisons

        Output Data:
            - Multi-panel figure with statistical tests
        """
        fig = plt.figure(figsize=(14, 8))

        # Create grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Panel 1: ANOVA summary (text)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')

        anova_text = f"""
ANOVA Results
{'='*40}

F-statistic: {anova_result['f_statistic']:.3f}
p-value: {anova_result['p_value']:.4f}
Significant: {'Yes' if anova_result['significant'] else 'No'}

Effect Size (η²): {anova_result['eta_squared']:.3f}
Interpretation: {anova_result['eta_squared_interpretation'].title()}

Degrees of Freedom:
  Between groups: {anova_result['df_between']}
  Within groups: {anova_result['df_within']}

Number of groups: {anova_result['n_groups']}
Total N: {anova_result['total_n']}
        """

        ax1.text(0.1, 0.9, anova_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Panel 2: Effect size distribution
        ax2 = fig.add_subplot(gs[0, 1])
        effect_sizes = [abs(c.cohens_d) for c in comparisons]
        ax2.hist(effect_sizes, bins=10, edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(effect_sizes), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(effect_sizes):.2f}')
        ax2.set_xlabel("Absolute Effect Size (|Cohen's d|)", fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Distribution of Effect Sizes', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: P-value distribution
        ax3 = fig.add_subplot(gs[1, 0])
        p_values = [c.p_value for c in comparisons]
        ax3.hist(p_values, bins=10, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0.05, color='red', linestyle='--',
                   linewidth=2, label='α = 0.05')
        ax3.set_xlabel('P-value', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Distribution of P-values', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Significance summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        n_significant = sum(1 for c in comparisons if c.significant)
        n_total = len(comparisons)
        pct_significant = (n_significant / n_total * 100) if n_total > 0 else 0

        # Effect size breakdown
        negligible = sum(1 for c in comparisons if c.effect_size_interpretation == 'negligible')
        small = sum(1 for c in comparisons if c.effect_size_interpretation == 'small')
        medium = sum(1 for c in comparisons if c.effect_size_interpretation == 'medium')
        large = sum(1 for c in comparisons if c.effect_size_interpretation == 'large')

        summary_text = f"""
Pairwise Comparisons Summary
{'='*40}

Total comparisons: {n_total}
Significant (α=0.05): {n_significant} ({pct_significant:.1f}%)
Non-significant: {n_total - n_significant}

Effect Size Breakdown:
  Negligible: {negligible}
  Small: {small}
  Medium: {medium}
  Large: {large}
        """

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        if output_path:
            self._save_figure(fig, output_path)

        return fig

    def _save_figure(self, fig: plt.Figure, output_path: Path):
        """Save figure in multiple formats"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save PNG
        fig.savefig(output_path.with_suffix('.png'),
                   dpi=self.dpi, bbox_inches='tight', facecolor='white')

        # Save PDF
        fig.savefig(output_path.with_suffix('.pdf'),
                   bbox_inches='tight', facecolor='white')
