# Sample Visualizations

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
