#!/usr/bin/env python3
"""
Run Experiment 2: Noise and Irrelevance

Measures how noise (irrelevant documents) impacts LLM accuracy.
Demonstrates degradation as noise ratio increases.

Usage:
    python scripts/run_experiment2.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.experiment2_noise_impact import NoiseImpactExperiment
from src.llm.ollama_client import OllamaClient
from src.data_generation.fact_generator import FactGenerator
from src.data_generation.noise_generator import NoiseGenerator
from src.config.settings import Settings
import json


def main():
    """Run Experiment 2"""
    print("=" * 80)
    print("EXPERIMENT 2: Noise and Irrelevance")
    print("=" * 80)

    # Load configuration
    try:
        settings = Settings.load()
        random_seed = settings.experiments.random_seed
        n_runs = settings.experiments.n_runs
        n_core_facts = settings.experiments.experiment2.n_core_facts
        noise_levels = settings.experiments.experiment2.noise_levels
        llm_config = settings.llm
    except Exception as e:
        print(f"Warning: Could not load config, using defaults: {e}")
        random_seed = 42
        n_runs = 5
        n_core_facts = 10
        noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        # Default LLM config
        class LLMConfig:
            model_name = "llama2"
            base_url = "http://localhost:11434"
            temperature = 0.0
            max_tokens = 512
            timeout = 30
        llm_config = LLMConfig()

    print(f"\nConfiguration:")
    print(f"  Model: {llm_config.model_name}")
    print(f"  Core facts: {n_core_facts}")
    print(f"  Noise levels: {[f'{n:.0%}' for n in noise_levels]}")
    print(f"  Runs per level: {n_runs}")
    print(f"  Random seed: {random_seed}")

    # Load facts
    facts_file = Path("data/facts/synthetic_facts.json")
    if not facts_file.exists():
        print(f"\nError: Facts file not found at {facts_file}")
        print("Run: python scripts/generate_data.py")
        return 1

    print(f"\nLoading facts from {facts_file}...")
    fact_gen = FactGenerator()
    facts = fact_gen.load_from_json(str(facts_file))
    print(f"Loaded {len(facts)} facts")

    # Load noise documents
    noise_file = Path("data/noise/noise_documents.json")
    if not noise_file.exists():
        print(f"\nError: Noise file not found at {noise_file}")
        print("Run: python scripts/generate_data.py")
        return 1

    print(f"Loading noise from {noise_file}...")
    noise_gen = NoiseGenerator()
    noise_docs = noise_gen.load_from_json(str(noise_file))
    print(f"Loaded {len(noise_docs)} noise documents")

    # Initialize Ollama client
    print(f"\nInitializing Ollama client...")
    llm_client = OllamaClient(
        model_name=llm_config.model_name,
        base_url=llm_config.base_url,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        timeout=llm_config.timeout
    )

    # Check Ollama availability
    if not llm_client.check_availability():
        print("\nError: Ollama service is not available")
        print("Please ensure Ollama is running: ollama serve")
        return 1

    print("✓ Ollama is available")

    # Initialize experiment
    print(f"\nInitializing Experiment 2...")
    experiment = NoiseImpactExperiment(
        llm_client=llm_client,
        facts=facts,
        noise_docs=noise_docs,
        n_core_facts=n_core_facts,
        noise_levels=noise_levels,
        random_seed=random_seed
    )

    # Run experiment
    print(f"\nRunning experiment...")
    print(f"Total trials: {len(facts)} facts × {len(noise_levels)} levels × {n_runs} runs = {len(facts) * len(noise_levels) * n_runs}")
    print("This may take a while...\n")

    results = experiment.run(n_runs=n_runs)

    # Save results
    output_file = Path("results/experiment2_results.json")
    print(f"\nSaving results to {output_file}...")
    experiment.save_results(results, str(output_file))

    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    hallucinated = sum(1 for r in results if r.hallucinated)
    accuracy = (correct / total) * 100
    hallucination_rate = (hallucinated / total) * 100

    print(f"\nOverall:")
    print(f"  Total trials: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Hallucination rate: {hallucination_rate:.1f}%")

    print(f"\nBy Noise Level:")
    for noise_ratio in noise_levels:
        noise_results = [r for r in results if r.noise_ratio == noise_ratio]
        noise_correct = sum(1 for r in noise_results if r.correct)
        noise_accuracy = (noise_correct / len(noise_results)) * 100
        print(f"  {noise_ratio:5.0%} noise: {noise_correct:3d}/{len(noise_results):3d} = {noise_accuracy:5.1f}%")

    # Calculate avg generation time
    avg_time = sum(r.generation_time_ms for r in results) / len(results)
    print(f"\nAverage generation time: {avg_time:.0f} ms")

    print(f"\n{'='*80}")
    print("✓ Experiment 2 complete!")
    print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
