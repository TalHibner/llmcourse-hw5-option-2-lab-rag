#!/usr/bin/env python3
"""
Run Experiment 1: Lost in the Middle

Demonstrates U-shaped performance curve where LLMs struggle with facts
in the middle of long contexts.

Usage:
    python scripts/run_experiment1.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.experiment1_context_window import ContextWindowExperiment
from src.llm.ollama_client import OllamaClient
from src.data_generation.fact_generator import FactGenerator
from src.config.settings import Settings
import json


def main():
    """Run Experiment 1"""
    print("=" * 80)
    print("EXPERIMENT 1: Lost in the Middle")
    print("=" * 80)

    # Load configuration
    try:
        settings = Settings.load()
        random_seed = settings.experiments.random_seed
        n_runs = settings.experiments.n_runs
        n_facts = settings.experiments.experiment1.n_facts
        position_categories = settings.experiments.experiment1.position_categories
        llm_config = settings.llm
    except Exception as e:
        print(f"Warning: Could not load config, using defaults: {e}")
        random_seed = 42
        n_runs = 5
        n_facts = 25
        position_categories = ["beginning", "middle", "end"]
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
    print(f"  Facts in context: {n_facts}")
    print(f"  Positions: {position_categories}")
    print(f"  Runs per position: {n_runs}")
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

    # List available models
    models = llm_client.list_models()
    print(f"Available models: {', '.join(models) if models else 'None'}")

    # Initialize experiment
    print(f"\nInitializing Experiment 1...")
    experiment = ContextWindowExperiment(
        llm_client=llm_client,
        facts=facts,
        n_facts=n_facts,
        position_categories=position_categories,
        random_seed=random_seed
    )

    # Run experiment
    print(f"\nRunning experiment...")
    print(f"Total trials: {len(facts)} facts × {len(position_categories)} positions × {n_runs} runs = {len(facts) * len(position_categories) * n_runs}")
    print("This may take a while...\n")

    results = experiment.run(n_runs=n_runs)

    # Save results
    output_file = Path("results/experiment1_results.json")
    print(f"\nSaving results to {output_file}...")
    experiment.save_results(results, str(output_file))

    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = (correct / total) * 100

    print(f"\nOverall:")
    print(f"  Total trials: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.1f}%")

    print(f"\nBy Position:")
    for position in position_categories:
        pos_results = [r for r in results if r.position == position]
        pos_correct = sum(1 for r in pos_results if r.correct)
        pos_accuracy = (pos_correct / len(pos_results)) * 100
        print(f"  {position:10s}: {pos_correct:3d}/{len(pos_results):3d} = {pos_accuracy:5.1f}%")

    # Calculate avg generation time
    avg_time = sum(r.generation_time_ms for r in results) / len(results)
    print(f"\nAverage generation time: {avg_time:.0f} ms")

    print(f"\n{'='*80}")
    print("✓ Experiment 1 complete!")
    print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
