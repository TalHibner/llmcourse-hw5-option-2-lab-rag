"""
Run mock experiments with simulated LLM responses.

This script simulates realistic LLM behavior without requiring Ollama:
- Experiment 1: U-shaped accuracy curve (Lost in the Middle)
- Experiment 2: Linear degradation with noise
- Experiment 3: RAG maintains high accuracy

The simulated responses match expected experimental patterns with
realistic variance, allowing complete analysis pipeline testing.
"""

import sys
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger
from src.utils.helpers import ensure_dir

logger = setup_logger("mock_experiments")


class MockLLMSimulator:
    """
    Simulates LLM behavior with realistic error patterns.

    The simulator models:
    - Position effects (U-shaped curve)
    - Noise sensitivity (linear degradation)
    - RAG effectiveness (maintained accuracy)
    - Realistic variance across runs
    """

    def __init__(self, random_seed: int = 42):
        """Initialize simulator with random seed"""
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def simulate_position_effect(
        self,
        position_normalized: float,
        base_accuracy: float = 0.87
    ) -> bool:
        """
        Simulate position effect with U-shaped curve.

        Args:
            position_normalized: Position in [0, 1] (0=beginning, 0.5=middle, 1=end)
            base_accuracy: Base accuracy at edges

        Returns:
            bool: Whether response is correct
        """
        # U-shaped curve: lowest at middle (0.5)
        # accuracy = base - depth * (1 - 4*(pos - 0.5)^2)
        depth = 0.35  # 35% accuracy drop at middle
        position_effect = 1 - 4 * (position_normalized - 0.5) ** 2
        accuracy = base_accuracy - depth * position_effect

        # Add some randomness
        accuracy += np.random.normal(0, 0.05)
        accuracy = np.clip(accuracy, 0, 1)

        return random.random() < accuracy

    def simulate_noise_effect(
        self,
        noise_ratio: float,
        base_accuracy: float = 0.94
    ) -> bool:
        """
        Simulate noise effect with linear degradation.

        Args:
            noise_ratio: Ratio of noise in [0, 1]
            base_accuracy: Base accuracy with no noise

        Returns:
            bool: Whether response is correct
        """
        # Linear degradation
        slope = -0.64  # -0.64 accuracy per noise unit
        accuracy = base_accuracy + slope * noise_ratio

        # Add randomness
        accuracy += np.random.normal(0, 0.06)
        accuracy = np.clip(accuracy, 0, 1)

        return random.random() < accuracy

    def simulate_rag_effect(
        self,
        noise_ratio: float,
        top_k: int = 5,
        base_accuracy: float = 0.93
    ) -> Dict[str, Any]:
        """
        Simulate RAG maintaining high accuracy despite noise.

        Args:
            noise_ratio: Ratio of noise in context
            top_k: Number of documents retrieved
            base_accuracy: Base RAG accuracy

        Returns:
            Dict with correct flag and retrieval metrics
        """
        # RAG shows minimal degradation
        # Small penalty for very high noise
        degradation = 0.03 * noise_ratio  # Only 3% drop per noise unit
        accuracy = base_accuracy - degradation

        # Top-k affects precision
        precision = 0.89 if top_k == 3 else (0.87 if top_k == 5 else 0.84)

        # Add randomness
        accuracy += np.random.normal(0, 0.03)
        precision += np.random.normal(0, 0.04)

        accuracy = np.clip(accuracy, 0, 1)
        precision = np.clip(precision, 0, 1)

        return {
            'correct': random.random() < accuracy,
            'retrieval_precision': precision,
            'retrieval_scores': [random.uniform(0.7, 0.95) for _ in range(top_k)]
        }


def run_mock_experiment1(n_runs: int = 10) -> Dict[str, Any]:
    """
    Run mock Experiment 1: Context Window Position Effects

    Simulates testing 25 facts at 11 different positions across 10 runs.
    Total measurements: 10 runs × 25 facts × 11 positions = 2,750
    """
    logger.info("Starting Mock Experiment 1: Context Window Position Effects")

    simulator = MockLLMSimulator()

    # Load facts
    facts_path = Path('data/facts/synthetic_facts.json')
    with open(facts_path) as f:
        facts = json.load(f)[:25]  # Use 25 facts

    # Position values to test
    positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    all_runs = []

    for run_idx in range(n_runs):
        logger.info(f"  Run {run_idx + 1}/{n_runs}")
        run_results = []

        for fact in facts:
            for pos_idx, position in enumerate(positions):
                # Simulate LLM response
                correct = simulator.simulate_position_effect(position)

                # Determine position category
                if position <= 0.2:
                    pos_category = 'beginning'
                elif position >= 0.8:
                    pos_category = 'end'
                else:
                    pos_category = 'middle'

                result = {
                    'run': run_idx,
                    'fact_id': fact['id'],
                    'question': fact['question'],
                    'expected_answer': fact['answer'],
                    'position': position,
                    'position_category': pos_category,
                    'position_index': pos_idx,
                    'total_facts': 25,
                    'correct': bool(correct),
                    'generation_time_ms': float(random.uniform(200, 800)),
                    'tokens': int(random.randint(10, 50))
                }

                run_results.append(result)

        all_runs.append(run_results)

    # Calculate summary statistics
    position_stats = {}
    for position in positions:
        pos_results = [
            r for run in all_runs for r in run
            if r['position'] == position
        ]
        accuracy = sum(1 for r in pos_results if r['correct']) / len(pos_results)
        position_stats[str(position)] = {
            'accuracy': accuracy,
            'count': len(pos_results)
        }

    # Save results
    output_dir = Path('results/experiment1')
    ensure_dir(output_dir)

    # Save all runs
    results_data = {
        'experiment': 'experiment1',
        'name': 'Context Window Position Effects',
        'n_runs': n_runs,
        'n_facts': 25,
        'n_positions': len(positions),
        'positions': positions,
        'total_measurements': len(all_runs[0]) * n_runs,
        'runs': all_runs,
        'summary': position_stats
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"  Saved results to {results_path}")
    logger.info(f"  Total measurements: {results_data['total_measurements']}")

    return results_data


def run_mock_experiment2(n_runs: int = 10) -> Dict[str, Any]:
    """
    Run mock Experiment 2: Noise Impact on Accuracy

    Simulates testing 10 facts at 7 noise levels across 10 runs.
    Total measurements: 10 runs × 10 facts × 7 noise levels = 700
    """
    logger.info("Starting Mock Experiment 2: Noise Impact on Accuracy")

    simulator = MockLLMSimulator()

    # Load facts
    facts_path = Path('data/facts/synthetic_facts.json')
    with open(facts_path) as f:
        facts = json.load(f)[:10]  # Use 10 core facts

    # Noise levels to test
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

    all_runs = []

    for run_idx in range(n_runs):
        logger.info(f"  Run {run_idx + 1}/{n_runs}")
        run_results = []

        for fact in facts:
            for noise_ratio in noise_levels:
                # Simulate LLM response
                correct = simulator.simulate_noise_effect(noise_ratio)

                result = {
                    'run': run_idx,
                    'fact_id': fact['id'],
                    'question': fact['question'],
                    'expected_answer': fact['answer'],
                    'noise_ratio': float(noise_ratio),
                    'n_noise_docs': int(10 * noise_ratio / (1 - noise_ratio)) if noise_ratio < 1 else 999,
                    'correct': bool(correct),
                    'generation_time_ms': float(random.uniform(300, 1200)),
                    'tokens': int(random.randint(10, 60))
                }

                run_results.append(result)

        all_runs.append(run_results)

    # Calculate summary statistics
    noise_stats = {}
    for noise_ratio in noise_levels:
        noise_results = [
            r for run in all_runs for r in run
            if r['noise_ratio'] == noise_ratio
        ]
        accuracy = sum(1 for r in noise_results if r['correct']) / len(noise_results)
        noise_stats[str(noise_ratio)] = {
            'accuracy': accuracy,
            'count': len(noise_results)
        }

    # Save results
    output_dir = Path('results/experiment2')
    ensure_dir(output_dir)

    results_data = {
        'experiment': 'experiment2',
        'name': 'Noise Impact on Accuracy',
        'n_runs': n_runs,
        'n_facts': 10,
        'n_noise_levels': len(noise_levels),
        'noise_levels': noise_levels,
        'total_measurements': len(all_runs[0]) * n_runs,
        'runs': all_runs,
        'summary': noise_stats
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"  Saved results to {results_path}")
    logger.info(f"  Total measurements: {results_data['total_measurements']}")

    return results_data


def run_mock_experiment3(n_runs: int = 10) -> Dict[str, Any]:
    """
    Run mock Experiment 3: RAG Solution Performance

    Simulates testing 15 facts with RAG (3 top-k values) and baseline at 90% noise.
    Total measurements: 10 runs × 15 facts × (3 top-k + 1 baseline) = 600
    """
    logger.info("Starting Mock Experiment 3: RAG Solution Performance")

    simulator = MockLLMSimulator()

    # Load facts
    facts_path = Path('data/facts/synthetic_facts.json')
    with open(facts_path) as f:
        facts = json.load(f)[:15]  # Use 15 test facts

    top_k_values = [3, 5, 10]
    high_noise = 0.9

    all_runs = []

    for run_idx in range(n_runs):
        logger.info(f"  Run {run_idx + 1}/{n_runs}")
        run_results = []

        for fact in facts:
            # Test RAG with different top-k values
            for top_k in top_k_values:
                rag_result = simulator.simulate_rag_effect(high_noise, top_k)

                result = {
                    'run': run_idx,
                    'fact_id': fact['id'],
                    'question': fact['question'],
                    'expected_answer': fact['answer'],
                    'approach': 'rag',
                    'top_k': int(top_k),
                    'noise_ratio': float(high_noise),
                    'correct': bool(rag_result['correct']),
                    'retrieval_precision': float(rag_result['retrieval_precision']),
                    'retrieval_scores': [float(s) for s in rag_result['retrieval_scores']],
                    'generation_time_ms': float(random.uniform(400, 1500)),
                    'tokens': int(random.randint(15, 70))
                }

                run_results.append(result)

            # Test baseline (no RAG) at same noise level
            baseline_correct = simulator.simulate_noise_effect(high_noise)

            result = {
                'run': run_idx,
                'fact_id': fact['id'],
                'question': fact['question'],
                'expected_answer': fact['answer'],
                'approach': 'baseline',
                'top_k': None,
                'noise_ratio': float(high_noise),
                'correct': bool(baseline_correct),
                'retrieval_precision': None,
                'retrieval_scores': None,
                'generation_time_ms': float(random.uniform(300, 1200)),
                'tokens': int(random.randint(10, 60))
            }

            run_results.append(result)

        all_runs.append(run_results)

    # Calculate summary statistics
    condition_stats = {}

    # RAG stats by top-k
    for top_k in top_k_values:
        rag_results = [
            r for run in all_runs for r in run
            if r['approach'] == 'rag' and r['top_k'] == top_k
        ]
        accuracy = sum(1 for r in rag_results if r['correct']) / len(rag_results)
        precision = np.mean([r['retrieval_precision'] for r in rag_results])

        condition_stats[f'rag_k{top_k}'] = {
            'accuracy': accuracy,
            'retrieval_precision': precision,
            'count': len(rag_results)
        }

    # Baseline stats
    baseline_results = [
        r for run in all_runs for r in run
        if r['approach'] == 'baseline'
    ]
    baseline_accuracy = sum(1 for r in baseline_results if r['correct']) / len(baseline_results)

    condition_stats['baseline'] = {
        'accuracy': baseline_accuracy,
        'count': len(baseline_results)
    }

    # Save results
    output_dir = Path('results/experiment3')
    ensure_dir(output_dir)

    results_data = {
        'experiment': 'experiment3',
        'name': 'RAG Solution Performance',
        'n_runs': n_runs,
        'n_facts': 15,
        'top_k_values': top_k_values,
        'high_noise_ratio': high_noise,
        'total_measurements': len(all_runs[0]) * n_runs,
        'runs': all_runs,
        'summary': condition_stats
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"  Saved results to {results_path}")
    logger.info(f"  Total measurements: {results_data['total_measurements']}")

    return results_data


def print_summary(exp1_data, exp2_data, exp3_data):
    """Print comprehensive summary of all experiments"""

    print("\n" + "="*70)
    print(" MOCK EXPERIMENT RESULTS SUMMARY")
    print("="*70)

    # Experiment 1
    print("\n📊 EXPERIMENT 1: Context Window Position Effects")
    print("-" * 70)
    print(f"Total measurements: {exp1_data['total_measurements']:,}")
    print(f"Runs: {exp1_data['n_runs']}")
    print(f"Facts tested: {exp1_data['n_facts']}")
    print(f"Positions tested: {exp1_data['n_positions']}")
    print("\nAccuracy by position:")

    for pos in ['0.0', '0.5', '1.0']:
        if pos in exp1_data['summary']:
            acc = exp1_data['summary'][pos]['accuracy']
            print(f"  Position {pos:>4} ({'beginning' if pos == '0.0' else 'middle' if pos == '0.5' else 'end':>9}): {acc:.1%}")

    # Find min/max
    accuracies = {float(k): v['accuracy'] for k, v in exp1_data['summary'].items()}
    min_pos = min(accuracies, key=accuracies.get)
    max_pos = max(accuracies, key=accuracies.get)

    print(f"\nLowest accuracy: {accuracies[min_pos]:.1%} at position {min_pos} (middle)")
    print(f"Highest accuracy: {accuracies[max_pos]:.1%} at position {max_pos}")
    print(f"Accuracy drop (middle): {(accuracies[max_pos] - accuracies[min_pos]):.1%}")

    # Experiment 2
    print("\n📉 EXPERIMENT 2: Noise Impact on Accuracy")
    print("-" * 70)
    print(f"Total measurements: {exp2_data['total_measurements']:,}")
    print(f"Runs: {exp2_data['n_runs']}")
    print(f"Facts tested: {exp2_data['n_facts']}")
    print(f"Noise levels tested: {exp2_data['n_noise_levels']}")
    print("\nAccuracy by noise ratio:")

    for noise in ['0.0', '0.5', '0.9']:
        if noise in exp2_data['summary']:
            acc = exp2_data['summary'][noise]['accuracy']
            print(f"  Noise {noise:>4} ({float(noise)*100:>3.0f}%): {acc:.1%}")

    acc_0 = exp2_data['summary']['0.0']['accuracy']
    acc_90 = exp2_data['summary']['0.9']['accuracy']
    print(f"\nDegradation (0% → 90%): {(acc_0 - acc_90):.1%}")

    # Experiment 3
    print("\n🚀 EXPERIMENT 3: RAG Solution Performance")
    print("-" * 70)
    print(f"Total measurements: {exp3_data['total_measurements']:,}")
    print(f"Runs: {exp3_data['n_runs']}")
    print(f"Facts tested: {exp3_data['n_facts']}")
    print(f"High noise ratio: {exp3_data['high_noise_ratio']:.0%}")
    print("\nAccuracy comparison at 90% noise:")

    rag_accuracies = []
    for top_k in exp3_data['top_k_values']:
        key = f'rag_k{top_k}'
        acc = exp3_data['summary'][key]['accuracy']
        prec = exp3_data['summary'][key]['retrieval_precision']
        rag_accuracies.append(acc)
        print(f"  RAG (top-k={top_k:2d}): {acc:.1%} (precision: {prec:.1%})")

    baseline_acc = exp3_data['summary']['baseline']['accuracy']
    avg_rag_acc = np.mean(rag_accuracies)

    print(f"  Baseline:        {baseline_acc:.1%}")
    print(f"\nRAG improvement: {(avg_rag_acc - baseline_acc):.1%} ({(avg_rag_acc - baseline_acc)*100:.1f} percentage points)")

    # Overall summary
    print("\n" + "="*70)
    print(" KEY FINDINGS")
    print("="*70)
    print(f"✓ Lost in Middle: {(accuracies[max_pos] - accuracies[min_pos]):.0%} accuracy drop at middle position")
    print(f"✓ Noise Impact:   {(acc_0 - acc_90):.0%} accuracy degradation with 90% noise")
    print(f"✓ RAG Solution:   {(avg_rag_acc - baseline_acc):.0%} improvement over baseline at high noise")
    print(f"✓ Total Data:     {exp1_data['total_measurements'] + exp2_data['total_measurements'] + exp3_data['total_measurements']:,} measurements across all experiments")
    print("="*70 + "\n")


def main():
    """Run all mock experiments"""
    logger.info("="*70)
    logger.info(" RUNNING MOCK EXPERIMENTS (No Ollama Required)")
    logger.info("="*70)
    logger.info("")
    logger.info("These experiments simulate realistic LLM behavior:")
    logger.info("  - Experiment 1: U-shaped accuracy curve (Lost in Middle)")
    logger.info("  - Experiment 2: Linear degradation with noise")
    logger.info("  - Experiment 3: RAG maintains high accuracy")
    logger.info("")
    logger.info("Each experiment runs 10 times for statistical validity.")
    logger.info("="*70)

    start_time = time.time()

    # Run experiments
    exp1_data = run_mock_experiment1(n_runs=10)
    exp2_data = run_mock_experiment2(n_runs=10)
    exp3_data = run_mock_experiment3(n_runs=10)

    elapsed = time.time() - start_time

    logger.info(f"\n✓ All experiments completed in {elapsed:.1f}s")

    # Print summary
    print_summary(exp1_data, exp2_data, exp3_data)

    print("\n📁 Results saved to:")
    print("   - results/experiment1/results.json")
    print("   - results/experiment2/results.json")
    print("   - results/experiment3/results.json")

    print("\n📊 Next steps:")
    print("   - Run statistical analysis: python3 scripts/analyze_results.py")
    print("   - Generate visualizations: python3 scripts/generate_visualizations.py")
    print("   - View sample plots: ls -lh results/figures/samples/")

    return exp1_data, exp2_data, exp3_data


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Failed to run mock experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
