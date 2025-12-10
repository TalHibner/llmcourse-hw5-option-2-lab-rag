"""
Visualization Module

Creates publication-quality plots for experiments:
- Line plots with confidence intervals
- Bar charts with error bars
- Heatmaps for position analysis
- 300 DPI export with LaTeX equations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ExperimentVisualizer:
    """
    Visualization tools for experiment results

    Setup Data:
    - style: str - matplotlib style (default 'seaborn-v0_8-paper')
    - palette: str - color palette (default 'Set2')
    - dpi: int - resolution for saved figures (default 300)
    - figure_size: Tuple[int, int] - figure size in inches

    Input Data:
    - data: Dict or List - experiment data to visualize
    - labels: List[str] - labels for data series

    Output Data:
    - Matplotlib figure and axes objects
    - Saved PNG/PDF files
    """

    def __init__(
        self,
        style: str = 'seaborn-v0_8-paper',
        palette: str = 'Set2',
        dpi: int = 300,
        figure_size: Tuple[int, int] = (10, 6)
    ):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style
            palette: Seaborn color palette
            dpi: DPI for saved figures
            figure_size: Figure size (width, height) in inches
        """
        self.dpi = dpi
        self.figure_size = figure_size

        # Set style
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not found, using default")
            plt.style.use('default')

        # Set color palette
        sns.set_palette(palette)

        # Enable LaTeX rendering if available
        try:
            plt.rcParams['text.usetex'] = False  # Use matplotlib's mathtext instead
            plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern font
        except:
            logger.warning("LaTeX rendering not available")

        logger.info(f"Initialized ExperimentVisualizer (dpi={dpi}, size={figure_size})")

    def plot_position_accuracy(
        self,
        position_data: Dict[str, List[float]],
        title: str = "Lost in the Middle: Position vs Accuracy",
        output_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot accuracy vs position (U-shaped curve)

        Args:
            position_data: Dict mapping positions to accuracy values
            title: Plot title
            output_path: Optional path to save figure

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        positions = list(position_data.keys())
        accuracies = [np.mean(vals) * 100 for vals in position_data.values()]
        stds = [np.std(vals) * 100 for vals in position_data.values()]

        # Plot bars with error bars
        x_pos = np.arange(len(positions))
        ax.bar(x_pos, accuracies, yerr=stds, capsize=5, alpha=0.7)

        # Styling
        ax.set_xlabel('Position in Context', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(positions)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        # Add horizontal line at random chance (if applicable)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random chance')
        ax.legend()

        plt.tight_layout()

        if output_path:
            self._save_figure(fig, output_path)

        logger.info(f"Created position accuracy plot with {len(positions)} positions")

        return fig, ax

    def plot_noise_impact(
        self,
        noise_data: Dict[float, List[float]],
        title: str = "Performance Degradation with Noise",
        output_path: Optional[str] = None,
        show_ci: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot accuracy vs noise ratio

        Args:
            noise_data: Dict mapping noise ratios to accuracy values
            title: Plot title
            output_path: Optional path to save figure
            show_ci: Show confidence intervals

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        noise_ratios = sorted(noise_data.keys())
        means = [np.mean(noise_data[nr]) * 100 for nr in noise_ratios]
        stds = [np.std(noise_data[nr]) * 100 for nr in noise_ratios]

        # Convert to percentages for x-axis
        noise_pct = [nr * 100 for nr in noise_ratios]

        # Plot line with markers
        ax.plot(noise_pct, means, marker='o', linewidth=2, markersize=8, label='Accuracy')

        # Add confidence intervals
        if show_ci:
            ci_margin = [1.96 * std / np.sqrt(len(noise_data[nr])) for nr, std in zip(noise_ratios, stds)]
            ax.fill_between(
                noise_pct,
                [m - ci for m, ci in zip(means, ci_margin)],
                [m + ci for m, ci in zip(means, ci_margin)],
                alpha=0.2,
                label='95% CI'
            )

        # Styling
        ax.set_xlabel('Noise Ratio (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if output_path:
            self._save_figure(fig, output_path)

        logger.info(f"Created noise impact plot with {len(noise_ratios)} noise levels")

        return fig, ax

    def plot_rag_comparison(
        self,
        baseline_data: Dict[float, List[float]],
        rag_data: Dict[float, List[float]],
        title: str = "RAG vs Baseline: Accuracy Comparison",
        output_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot RAG vs baseline accuracy

        Args:
            baseline_data: Dict mapping noise ratios to baseline accuracy
            rag_data: Dict mapping noise ratios to RAG accuracy
            title: Plot title
            output_path: Optional path to save figure

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        noise_ratios = sorted(baseline_data.keys())
        noise_pct = [nr * 100 for nr in noise_ratios]

        baseline_means = [np.mean(baseline_data[nr]) * 100 for nr in noise_ratios]
        rag_means = [np.mean(rag_data[nr]) * 100 for nr in noise_ratios]

        # Plot both lines
        ax.plot(noise_pct, baseline_means, marker='o', linewidth=2, markersize=8,
                label='Baseline (No RAG)', linestyle='--')
        ax.plot(noise_pct, rag_means, marker='s', linewidth=2, markersize=8,
                label='RAG', linestyle='-')

        # Add horizontal line at 90% (target)
        ax.axhline(y=90, color='g', linestyle=':', alpha=0.5,
                   label='Target (90%)')

        # Styling
        ax.set_xlabel('Noise Ratio (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()

        if output_path:
            self._save_figure(fig, output_path)

        logger.info("Created RAG comparison plot")

        return fig, ax

    def plot_metrics_heatmap(
        self,
        data: np.ndarray,
        x_labels: List[str],
        y_labels: List[str],
        title: str = "Metrics Heatmap",
        output_path: Optional[str] = None,
        fmt: str = '.2f',
        cmap: str = 'YlOrRd'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create heatmap visualization

        Args:
            data: 2D numpy array of values
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            title: Plot title
            output_path: Optional path to save figure
            fmt: Format string for annotations
            cmap: Colormap name

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Create heatmap
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=x_labels,
            yticklabels=y_labels,
            ax=ax,
            cbar_kws={'label': 'Accuracy (%)'}
        )

        # Styling
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            self._save_figure(fig, output_path)

        logger.info(f"Created heatmap ({data.shape[0]}x{data.shape[1]})")

        return fig, ax

    def plot_retrieval_precision(
        self,
        top_k_data: Dict[int, List[float]],
        title: str = "Retrieval Precision vs top-k",
        output_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot retrieval precision for different top-k values

        Args:
            top_k_data: Dict mapping top-k values to precision scores
            title: Plot title
            output_path: Optional path to save figure

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        top_k_values = sorted(top_k_data.keys())
        means = [np.mean(top_k_data[k]) * 100 for k in top_k_values]
        stds = [np.std(top_k_data[k]) * 100 for k in top_k_values]

        # Plot bars
        ax.bar(range(len(top_k_values)), means, yerr=stds, capsize=5, alpha=0.7)

        # Styling
        ax.set_xlabel('top-k', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(top_k_values)))
        ax.set_xticklabels([f'k={k}' for k in top_k_values])
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            self._save_figure(fig, output_path)

        logger.info(f"Created retrieval precision plot with {len(top_k_values)} k values")

        return fig, ax

    def create_summary_figure(
        self,
        exp1_data: Dict[str, List[float]],
        exp2_data: Dict[float, List[float]],
        exp3_baseline: Dict[float, List[float]],
        exp3_rag: Dict[float, List[float]],
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive summary figure with all experiments

        Args:
            exp1_data: Position data from Experiment 1
            exp2_data: Noise data from Experiment 2
            exp3_baseline: Baseline data from Experiment 3
            exp3_rag: RAG data from Experiment 3
            output_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(15, 5))

        # Experiment 1: Position
        ax1 = plt.subplot(1, 3, 1)
        positions = list(exp1_data.keys())
        accuracies = [np.mean(vals) * 100 for vals in exp1_data.values()]
        ax1.bar(range(len(positions)), accuracies, alpha=0.7)
        ax1.set_xlabel('Position', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('Exp 1: Lost in the Middle', fontweight='bold')
        ax1.set_xticks(range(len(positions)))
        ax1.set_xticklabels(positions, rotation=45)
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)

        # Experiment 2: Noise
        ax2 = plt.subplot(1, 3, 2)
        noise_ratios = sorted(exp2_data.keys())
        noise_pct = [nr * 100 for nr in noise_ratios]
        means = [np.mean(exp2_data[nr]) * 100 for nr in noise_ratios]
        ax2.plot(noise_pct, means, marker='o', linewidth=2)
        ax2.set_xlabel('Noise Ratio (%)', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Exp 2: Noise Impact', fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(alpha=0.3)

        # Experiment 3: RAG vs Baseline
        ax3 = plt.subplot(1, 3, 3)
        baseline_means = [np.mean(exp3_baseline[nr]) * 100 for nr in noise_ratios]
        rag_means = [np.mean(exp3_rag[nr]) * 100 for nr in noise_ratios]
        ax3.plot(noise_pct, baseline_means, marker='o', label='Baseline', linestyle='--')
        ax3.plot(noise_pct, rag_means, marker='s', label='RAG', linestyle='-')
        ax3.axhline(y=90, color='g', linestyle=':', alpha=0.5, label='Target')
        ax3.set_xlabel('Noise Ratio (%)', fontweight='bold')
        ax3.set_ylabel('Accuracy (%)', fontweight='bold')
        ax3.set_title('Exp 3: RAG Solution', fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.grid(alpha=0.3)
        ax3.legend()

        plt.tight_layout()

        if output_path:
            self._save_figure(fig, output_path)

        logger.info("Created comprehensive summary figure")

        return fig

    def _save_figure(self, fig: plt.Figure, output_path: str) -> None:
        """
        Save figure to file

        Args:
            fig: Figure to save
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save as PNG
        fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")

        # Also save as PDF if PNG
        if output_file.suffix == '.png':
            pdf_path = output_file.with_suffix('.pdf')
            fig.savefig(pdf_path, bbox_inches='tight')
            logger.info(f"Saved PDF to {pdf_path}")
