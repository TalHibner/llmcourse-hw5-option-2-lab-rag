#!/usr/bin/env python3
"""
Generate all synthetic data for RAG experiments

Creates:
1. 25 synthetic fact documents (data/facts/synthetic_facts.json)
2. 100 noise documents (data/noise/noise_documents.json)

Usage:
    python scripts/generate_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.fact_generator import FactGenerator
from src.data_generation.noise_generator import NoiseGenerator
from src.config.settings import Settings


def main():
    """Generate all synthetic datasets"""
    print("=" * 60)
    print("RAG Context Window Research - Data Generation")
    print("=" * 60)

    # Load configuration
    try:
        settings = Settings.load()
        random_seed = settings.experiments.random_seed
        n_facts = settings.experiments.experiment1.n_facts
        n_noise_pool = settings.experiments.experiment2.n_noise_pool
    except Exception as e:
        print(f"Warning: Could not load config, using defaults: {e}")
        random_seed = 42
        n_facts = 25
        n_noise_pool = 100

    print(f"\nConfiguration:")
    print(f"  Random seed: {random_seed}")
    print(f"  Number of facts: {n_facts}")
    print(f"  Noise pool size: {n_noise_pool}")

    # Create output directories if they don't exist
    facts_dir = Path("data/facts")
    noise_dir = Path("data/noise")
    facts_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    # Generate facts
    print(f"\n{'='*60}")
    print("Generating synthetic facts...")
    print(f"{'='*60}")

    fact_gen = FactGenerator(seed=random_seed)
    facts = fact_gen.generate_facts(n=n_facts)

    # Validate diversity
    try:
        fact_gen.ensure_diversity(facts)
        print(f"✓ Generated {len(facts)} diverse facts")
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    # Print sample facts
    print("\nSample facts:")
    for i, fact in enumerate(facts[:3], 1):
        print(f"\n{i}. [{fact.category}] {fact.fact_text}")
        print(f"   Q: {fact.question}")
        print(f"   A: {fact.answer}")

    # Save facts
    facts_file = facts_dir / "synthetic_facts.json"
    fact_gen.save_to_json(facts, str(facts_file))
    print(f"\n✓ Saved facts to: {facts_file}")

    # Print category distribution
    categories = {}
    for fact in facts:
        categories[fact.category] = categories.get(fact.category, 0) + 1

    print("\nCategory distribution:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count} facts")

    # Generate noise documents
    print(f"\n{'='*60}")
    print("Generating noise documents...")
    print(f"{'='*60}")

    noise_gen = NoiseGenerator(seed=random_seed)
    noise_docs = noise_gen.generate_noise(n=n_noise_pool)
    print(f"✓ Generated {len(noise_docs)} noise documents")

    # Print sample noise
    print("\nSample noise documents:")
    for i, doc in enumerate(noise_docs[:3], 1):
        print(f"{i}. {doc[:80]}...")

    # Save noise
    noise_file = noise_dir / "noise_documents.json"
    noise_gen.save_to_json(noise_docs, str(noise_file))
    print(f"\n✓ Saved noise documents to: {noise_file}")

    # Statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")

    avg_fact_length = sum(len(f.fact_text.split()) for f in facts) / len(facts)
    avg_noise_length = sum(len(doc.split()) for doc in noise_docs) / len(noise_docs)

    print(f"Facts:")
    print(f"  Total: {len(facts)}")
    print(f"  Categories: {len(categories)}")
    print(f"  Avg length: {avg_fact_length:.1f} words")

    print(f"\nNoise documents:")
    print(f"  Total: {len(noise_docs)}")
    print(f"  Avg length: {avg_noise_length:.1f} words")

    # Calculate noise ratios
    print(f"\nNoise ratio examples (for {len(facts)} core facts):")
    for ratio in [0.2, 0.4, 0.6, 0.8, 0.9]:
        n_noise_needed = noise_gen.sample_for_noise_ratio(len(facts), ratio)
        total = len(facts) + n_noise_needed
        print(f"  {ratio*100:.0f}% noise: {n_noise_needed} noise docs + {len(facts)} facts = {total} total")

    print(f"\n{'='*60}")
    print("✓ Data generation complete!")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
