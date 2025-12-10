"""
Generate sample visualizations with synthetic data to demonstrate plot capabilities.

This script creates example visualizations showing what the actual experiment
results will look like, without requiring Ollama to be running.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.visualization import ExperimentVisualizer
from src.analysis.research_visualization import ResearchVisualizer
from src.analysis.research_analysis import ResearchResult, ComparisonResult
from src.utils.logging import setup_logger

logger = setup_logger("sample_viz")


def generate_sample_position_data():
    """Generate sample data showing U-shaped curve for position experiment"""
    positions = ['beginning', 'early', 'middle-early', 'middle', 'middle-late', 'late', 'end']

    # Create U-shaped accuracy pattern
    accuracy_pattern = [0.87, 0.82, 0.68, 0.52, 0.64, 0.81, 0.86]

    data = {}
    for pos, acc in zip(positions, accuracy_pattern):
        # Generate 10 samples around the mean with some variance
        samples = np.random.normal(acc, 0.08, 10)
        samples = np.clip(samples, 0, 1)  # Keep in [0, 1]
        data[pos] = samples.tolist()

    return data


def generate_sample_noise_data():
    """Generate sample data showing linear degradation with noise"""
    noise_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

    # Linear degradation: accuracy = 0.94 - 0.64 * noise_ratio
    data = {}
    for nr in noise_ratios:
        mean_acc = 0.94 - 0.64 * nr
        # Add some variance
        samples = np.random.normal(mean_acc, 0.06, 10)
        samples = np.clip(samples, 0, 1)
        data[nr] = samples.tolist()

    return data


def generate_sample_rag_data():
    """Generate sample data showing RAG maintaining high accuracy"""
    noise_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

    # Baseline: degrades significantly
    baseline_data = {}
    for nr in noise_ratios:
        mean_acc = 0.94 - 0.64 * nr
        samples = np.random.normal(mean_acc, 0.06, 10)
        samples = np.clip(samples, 0, 1)
        baseline_data[nr] = samples.tolist()

    # RAG: maintains high accuracy
    rag_data = {}
    for nr in noise_ratios:
        mean_acc = 0.93 - 0.03 * nr  # Only slight degradation
        samples = np.random.normal(mean_acc, 0.03, 10)
        samples = np.clip(samples, 0, 1)
        rag_data[nr] = samples.tolist()

    return baseline_data, rag_data


def generate_sample_research_results():
    """Generate ResearchResult objects for research visualizations"""
    positions = ['0.0', '0.2', '0.4', '0.5', '0.6', '0.8', '1.0']
    accuracy_pattern = [0.87, 0.68, 0.58, 0.52, 0.62, 0.81, 0.86]

    results = {}
    for pos, acc in zip(positions, accuracy_pattern):
        accuracies = np.random.normal(acc, 0.08, 10)
        accuracies = np.clip(accuracies, 0, 1)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)

        # Calculate 95% CI
        from scipy import stats
        n = len(accuracies)
        se = std_acc / np.sqrt(n)
        ci_margin = stats.t.ppf(0.975, n-1) * se

        results[pos] = ResearchResult(
            experiment_name="Experiment 1",
            condition=pos,
            mean_accuracy=mean_acc,
            std_accuracy=std_acc,
            ci_lower=mean_acc - ci_margin,
            ci_upper=mean_acc + ci_margin,
            n_runs=10,
            n_samples_per_run=25,
            total_samples=250,
            accuracies_per_run=accuracies.tolist()
        )

    return results


def generate_sample_comparisons():
    """Generate sample comparison results"""
    comparisons = []

    # Beginning vs Middle
    comparisons.append(ComparisonResult(
        condition_a="0.0",
        condition_b="0.5",
        mean_a=0.87,
        mean_b=0.52,
        difference=0.35,
        cohens_d=3.11,
        effect_size_interpretation="large",
        test_statistic=9.83,
        p_value=0.0001,
        significant=True,
        ci_lower=0.29,
        ci_upper=0.41
    ))

    # Middle vs End
    comparisons.append(ComparisonResult(
        condition_a="0.5",
        condition_b="1.0",
        mean_a=0.52,
        mean_b=0.86,
        difference=-0.34,
        cohens_d=-2.88,
        effect_size_interpretation="large",
        test_statistic=-9.12,
        p_value=0.0001,
        significant=True,
        ci_lower=-0.40,
        ci_upper=-0.28
    ))

    return comparisons


def generate_sample_anova():
    """Generate sample ANOVA results"""
    return {
        'f_statistic': 14.73,
        'p_value': 0.0001,
        'significant': True,
        'df_between': 6,
        'df_within': 2743,
        'eta_squared': 0.182,
        'eta_squared_interpretation': 'large',
        'n_groups': 7,
        'total_n': 2750
    }


def main():
    """Generate all sample visualizations"""
    logger.info("Starting sample visualization generation")

    # Create output directory
    output_dir = Path('results/figures/samples')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizers
    viz = ExperimentVisualizer(dpi=300, figure_size=(10, 6))
    research_viz = ResearchVisualizer(dpi=300, figure_size=(10, 6))

    logger.info("Generating Experiment 1 visualization (Position Effects)")
    # Experiment 1: Position Effects
    position_data = generate_sample_position_data()
    fig1, _ = viz.plot_position_accuracy(
        position_data,
        title="Experiment 1: Lost in the Middle (Sample Data)",
        output_path=output_dir / "experiment1_position_accuracy.png"
    )
    logger.info("✓ Created experiment1_position_accuracy.png")

    logger.info("Generating Experiment 2 visualization (Noise Impact)")
    # Experiment 2: Noise Impact
    noise_data = generate_sample_noise_data()
    fig2, _ = viz.plot_noise_impact(
        noise_data,
        title="Experiment 2: Noise Impact on Accuracy (Sample Data)",
        output_path=output_dir / "experiment2_noise_impact.png",
        show_ci=True
    )
    logger.info("✓ Created experiment2_noise_impact.png")

    logger.info("Generating Experiment 3 visualization (RAG Comparison)")
    # Experiment 3: RAG vs Baseline
    baseline_data, rag_data = generate_sample_rag_data()
    fig3, _ = viz.plot_rag_comparison(
        baseline_data,
        rag_data,
        title="Experiment 3: RAG vs Baseline (Sample Data)",
        output_path=output_dir / "experiment3_rag_comparison.png"
    )
    logger.info("✓ Created experiment3_rag_comparison.png")

    logger.info("Generating comprehensive summary figure")
    # Summary figure with all three experiments
    fig4 = viz.create_summary_figure(
        position_data,
        noise_data,
        baseline_data,
        rag_data,
        output_path=output_dir / "summary_all_experiments.png"
    )
    logger.info("✓ Created summary_all_experiments.png")

    logger.info("Generating research-grade visualizations")
    # Research-grade visualizations
    research_results = generate_sample_research_results()

    # Position accuracy with CI
    fig5 = research_viz.plot_line_with_ci(
        research_results,
        title="Context Position Effects with 95% Confidence Intervals (Sample)",
        xlabel="Normalized Position (0=Beginning, 1=End)",
        ylabel="Accuracy",
        output_path=output_dir / "research_position_with_ci.png",
        x_numeric=True,
        reference_line=0.5
    )
    logger.info("✓ Created research_position_with_ci.png")

    # Effect sizes forest plot
    comparisons = generate_sample_comparisons()
    fig6 = research_viz.plot_effect_sizes(
        comparisons,
        title="Effect Sizes for Key Comparisons (Sample)",
        output_path=output_dir / "research_effect_sizes.png",
        reference_lines=True
    )
    logger.info("✓ Created research_effect_sizes.png")

    # Statistical summary figure
    anova_results = generate_sample_anova()
    fig7 = research_viz.create_statistical_summary_figure(
        anova_results,
        comparisons,
        title="Statistical Analysis Summary (Sample Data)",
        output_path=output_dir / "research_statistical_summary.png"
    )
    logger.info("✓ Created research_statistical_summary.png")

    # Retrieval precision
    top_k_data = {
        3: np.random.normal(0.89, 0.04, 10),
        5: np.random.normal(0.87, 0.04, 10),
        10: np.random.normal(0.84, 0.05, 10)
    }
    fig8, _ = viz.plot_retrieval_precision(
        top_k_data,
        title="RAG Retrieval Precision by Top-K (Sample Data)",
        output_path=output_dir / "rag_retrieval_precision.png"
    )
    logger.info("✓ Created rag_retrieval_precision.png")

    # Create README for samples
    readme_content = """# Sample Visualizations

This directory contains **sample visualizations** generated with synthetic data to demonstrate
what the actual experiment results will look like.

## Generated Figures

### Basic Experiment Plots

1. **experiment1_position_accuracy.png** / **.pdf**
   - Shows U-shaped accuracy curve (Lost in the Middle phenomenon)
   - X-axis: Position in context (beginning, middle, end)
   - Y-axis: Accuracy percentage
   - Demonstrates that middle-positioned information has lowest accuracy

2. **experiment2_noise_impact.png** / **.pdf**
   - Shows linear degradation with increasing noise
   - X-axis: Noise ratio (0% to 90%)
   - Y-axis: Accuracy percentage
   - Includes 95% confidence interval shading
   - Demonstrates systematic performance degradation

3. **experiment3_rag_comparison.png** / **.pdf**
   - Compares RAG vs Baseline across noise levels
   - Shows RAG maintains high accuracy (>90%) even at high noise
   - Baseline degrades significantly with noise
   - Green dashed line shows 90% target accuracy

4. **summary_all_experiments.png** / **.pdf**
   - Three-panel summary showing all experiments
   - Side-by-side comparison for easy interpretation
   - Publication-ready format

### Research-Grade Statistical Plots

5. **research_position_with_ci.png** / **.pdf**
   - Position effects with 95% confidence interval shading
   - More detailed x-axis (11 positions: 0.0 to 1.0)
   - Reference line at 50% (chance level)
   - Publication-quality formatting (300 DPI)

6. **research_effect_sizes.png** / **.pdf**
   - Forest plot of Cohen's d effect sizes
   - Shows magnitude of differences between conditions
   - Color-coded by statistical significance (red=significant, gray=not significant)
   - Reference lines for small (0.2), medium (0.5), large (0.8) effects
   - Includes p-values for each comparison

7. **research_statistical_summary.png** / **.pdf**
   - Four-panel comprehensive statistical summary:
     - Panel 1: ANOVA results (F-statistic, p-value, η²)
     - Panel 2: Distribution of effect sizes
     - Panel 3: Distribution of p-values (with α=0.05 threshold)
     - Panel 4: Summary of significant comparisons
   - Suitable for supplementary materials in papers

8. **rag_retrieval_precision.png** / **.pdf**
   - Retrieval precision for different top-k values (3, 5, 10)
   - Shows that precision remains high (>80%) across k values
   - Error bars show standard deviation

## Data Used

All visualizations use **synthetic data** that mimics expected experimental results:

- **Position effects**: U-shaped curve with ~35% accuracy drop in middle
- **Noise impact**: Linear degradation with slope ≈ -0.64
- **RAG performance**: Maintains 90-93% accuracy across noise levels
- **Effect sizes**: Cohen's d > 0.8 (large effects) for critical comparisons
- **Statistical significance**: p < 0.001 for main effects

## Actual Results

Once experiments are run with Ollama, actual results will be placed in:
- `results/figures/experiment1_*.png`
- `results/figures/experiment2_*.png`
- `results/figures/experiment3_*.png`

The actual figures will have the same format but with real experimental data.

## Usage in Papers

These visualizations are designed for academic publications:
- **300 DPI** resolution (journal-quality)
- **PDF format** for vector graphics (scalable)
- **Clear labels** with bold, readable fonts
- **Statistical annotations** (CI, p-values, effect sizes)
- **Publication-standard** color schemes and styling

## Regenerating Samples

To regenerate these samples:

```bash
python3 scripts/generate_sample_visualizations.py
```

## Dependencies

- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- numpy >= 1.21.0
- scipy >= 1.7.0
"""

    with open(output_dir / "README.md", 'w') as f:
        f.write(readme_content)
    logger.info("✓ Created README.md")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          Sample Visualizations Generated Successfully         ║
╚══════════════════════════════════════════════════════════════╝

Generated 8 sample visualizations in: {output_dir}

📊 Basic Experiment Plots:
   1. experiment1_position_accuracy.png/pdf
   2. experiment2_noise_impact.png/pdf
   3. experiment3_rag_comparison.png/pdf
   4. summary_all_experiments.png/pdf

📈 Research-Grade Statistical Plots:
   5. research_position_with_ci.png/pdf
   6. research_effect_sizes.png/pdf
   7. research_statistical_summary.png/pdf
   8. rag_retrieval_precision.png/pdf

✓ All figures saved as both PNG (300 DPI) and PDF (vector)
✓ README.md created with documentation

View the samples to see what actual experiment results will look like!
""")
    logger.info("Sample visualization generation completed successfully")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Failed to generate sample visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
