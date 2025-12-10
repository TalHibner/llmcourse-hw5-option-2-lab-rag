"""
Tests for data generation modules
"""

import pytest
from src.data_generation.fact_generator import Fact, FactGenerator
from src.data_generation.noise_generator import NoiseGenerator


class TestFact:
    """Test Fact dataclass"""

    def test_fact_creation(self):
        """Test basic fact creation"""
        fact = Fact(
            id="test_001",
            category="test",
            fact_text="Paris is the capital of France.",
            question="What is the capital of France?",
            answer="Paris",
            metadata={"complexity": "simple"}
        )

        assert fact.id == "test_001"
        assert fact.category == "test"
        assert "Paris" in fact.fact_text
        assert fact.answer in fact.fact_text

    def test_fact_validation(self):
        """Test fact validation"""
        fact = Fact(
            id="test_001",
            category="test",
            fact_text="Paris is the capital of France.",
            question="What is the capital of France?",
            answer="Paris",
            metadata={}
        )

        assert fact.validate() is True

    def test_fact_validation_fails(self):
        """Test fact validation failure when answer not in fact"""
        fact = Fact(
            id="test_001",
            category="test",
            fact_text="Paris is the capital of France.",
            question="What is the capital of Germany?",
            answer="Berlin",
            metadata={}
        )

        with pytest.raises(ValueError, match="Answer.*not found"):
            fact.validate()


class TestFactGenerator:
    """Test FactGenerator"""

    def test_generator_initialization(self):
        """Test generator initialization"""
        gen = FactGenerator(seed=42)
        assert gen.seed == 42
        assert gen.random is not None

    def test_generate_facts(self):
        """Test fact generation"""
        gen = FactGenerator(seed=42)
        facts = gen.generate_facts(n=10)

        assert len(facts) == 10
        assert all(isinstance(f, Fact) for f in facts)
        assert all(f.validate() for f in facts)

    def test_fact_diversity(self):
        """Test fact diversity"""
        gen = FactGenerator(seed=42)
        facts = gen.generate_facts(n=25)

        # Check category diversity
        categories = set(f.category for f in facts)
        assert len(categories) >= 3

        # Ensure diversity validation passes
        gen.ensure_diversity(facts)

    def test_save_and_load(self, tmp_path):
        """Test saving and loading facts"""
        gen = FactGenerator(seed=42)
        facts = gen.generate_facts(n=5)

        # Save
        output_file = tmp_path / "test_facts.json"
        gen.save_to_json(facts, str(output_file))
        assert output_file.exists()

        # Load
        loaded_facts = gen.load_from_json(str(output_file))
        assert len(loaded_facts) == len(facts)
        assert loaded_facts[0].id == facts[0].id


class TestNoiseGenerator:
    """Test NoiseGenerator"""

    def test_generator_initialization(self):
        """Test generator initialization"""
        gen = NoiseGenerator(seed=42)
        assert gen.seed == 42

    def test_generate_noise(self):
        """Test noise generation"""
        gen = NoiseGenerator(seed=42)
        noise_docs = gen.generate_noise(n=50)

        assert len(noise_docs) == 50
        assert all(isinstance(doc, str) for doc in noise_docs)
        assert all(len(doc) > 0 for doc in noise_docs)

    def test_noise_ratio_calculation(self):
        """Test noise ratio calculation"""
        gen = NoiseGenerator()

        # Test various ratios
        assert gen.sample_for_noise_ratio(10, 0.0) == 0
        assert gen.sample_for_noise_ratio(10, 0.5) == 10  # 10 * 0.5 / 0.5 = 10
        assert gen.sample_for_noise_ratio(10, 0.8) == 40  # 10 * 0.8 / 0.2 = 40

    def test_save_and_load(self, tmp_path):
        """Test saving and loading noise"""
        gen = NoiseGenerator(seed=42)
        noise = gen.generate_noise(n=10)

        # Save
        output_file = tmp_path / "test_noise.json"
        gen.save_to_json(noise, str(output_file))
        assert output_file.exists()

        # Load
        loaded_noise = gen.load_from_json(str(output_file))
        assert len(loaded_noise) == len(noise)
        assert loaded_noise[0] == noise[0]
