"""
Noise Document Generator

Generates irrelevant "filler" documents that don't overlap with fact domains.
Used to test LLM robustness and RAG retrieval precision.
"""

import random
import json
from typing import List


class NoiseGenerator:
    """
    Generates noise documents for experiments

    Input Data:
    - n: int - number of noise documents to generate

    Setup Data:
    - seed: int - random seed for reproducibility

    Output Data:
    - List[str] - noise document texts
    """

    # Noise templates from unrelated domains
    NOISE_TEMPLATES = [
        # Astronomy
        "The Andromeda Galaxy is approximately 2.5 million light-years from Earth.",
        "Jupiter is the largest planet in our solar system with a mass of 1.898 × 10^27 kg.",
        "A black hole's event horizon marks the boundary beyond which nothing can escape.",
        "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        "Neutron stars are the densest objects in the universe apart from black holes.",

        # Cuisine
        "Sushi originated in Southeast Asia as a method of preserving fish in fermented rice.",
        "Molecular gastronomy combines physics and chemistry to transform food textures.",
        "The Maillard reaction creates complex flavors when proteins and sugars are heated.",
        "Umami was identified as the fifth basic taste in addition to sweet, sour, bitter, and salty.",
        "Sourdough bread uses wild yeast and lactic acid bacteria for natural fermentation.",

        # Sports
        "The modern Olympic Games were revived in Athens in 1896 by Baron Pierre de Coubertin.",
        "Basketball was invented by James Naismith in 1891 using a soccer ball and peach baskets.",
        "The Tour de France covers approximately 3,500 kilometers over 21 stages.",
        "In cricket, a googly is a deceptive delivery that spins opposite to the expected direction.",
        "Formula 1 racing cars can accelerate from 0 to 100 km/h in approximately 2 seconds.",

        # Technology
        "Quantum computing utilizes quantum mechanical phenomena like superposition and entanglement.",
        "The first computer mouse was invented by Douglas Engelbart in 1964 using a wooden shell.",
        "Blockchain technology creates immutable distributed ledgers through cryptographic hashing.",
        "Machine learning algorithms improve their performance through iterative training on data.",
        "The TCP/IP protocol suite forms the foundation of internet communication.",

        # Art
        "Impressionism emerged in France during the 1860s emphasizing light and color over detail.",
        "The Renaissance period marked a revival of classical learning and artistic innovation.",
        "Abstract expressionism developed in New York in the 1940s focusing on emotional expression.",
        "Leonardo da Vinci's Last Supper uses one-point linear perspective from the vanishing point.",
        "The Baroque period is characterized by dramatic use of light, shadow, and ornamentation.",

        # Music
        "The Stradivarius violin is renowned for its superior sound quality and craftsmanship.",
        "A symphony typically consists of four movements with varying tempos and moods.",
        "Jazz originated in African American communities of New Orleans in the late 19th century.",
        "The equal temperament tuning system divides the octave into 12 equal semitones.",
        "Beethoven's Ninth Symphony was the first major symphony to include vocal soloists and chorus.",

        # Architecture
        "The Gothic style features pointed arches, ribbed vaults, and flying buttresses.",
        "Frank Lloyd Wright pioneered organic architecture integrating buildings with nature.",
        "The Pantheon's dome remains the world's largest unreinforced concrete dome.",
        "Brutalist architecture emphasizes raw concrete and geometric forms.",
        "The International Style emerged in the 1920s with clean lines and glass curtain walls.",

        # Biology
        "Mitochondria are known as the powerhouse of the cell producing ATP through cellular respiration.",
        "DNA replication is semiconservative with each strand serving as a template.",
        "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
        "The human genome contains approximately 3 billion base pairs of DNA.",
        "Evolution occurs through natural selection acting on heritable genetic variation.",

        # Economics
        "Supply and demand curves intersect at the equilibrium price and quantity.",
        "Keynesian economics advocates government intervention during economic downturns.",
        "The law of diminishing marginal returns states productivity gains eventually decline.",
        "Inflation measures the rate of increase in the general price level over time.",
        "Comparative advantage explains why countries benefit from international trade.",

        # Philosophy
        "René Descartes' cogito ergo sum establishes existence through the act of thinking.",
        "Plato's Theory of Forms posits that abstract forms are more real than material objects.",
        "Utilitarianism seeks the greatest happiness for the greatest number of people.",
        "Existentialism emphasizes individual freedom, choice, and personal responsibility.",
        "Kant's categorical imperative requires acting only on universalizable maxims.",

        # Fictional entities (clearly not real)
        "Zorgonia is a fictional country located in the eastern region of Fantasyland.",
        "The Quibble bird is native to imaginary rainforests and feeds on crystallized moonbeams.",
        "Chronosteel is a theoretical material that can manipulate localized time flow.",
        "The Nebulous Theorem proposes that abstract concepts have measurable quantum properties.",
        "Mythical creatures like dragons are said to have evolved from ancient mega-reptiles.",
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize noise generator

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def generate_noise(self, n: int = 100) -> List[str]:
        """
        Generate n noise documents

        Args:
            n: Number of noise documents (recommended 100+)

        Returns:
            List of noise document texts

        Raises:
            ValueError: If n is too small
        """
        if n < 50:
            raise ValueError(f"Should generate at least 50 noise documents, got {n}")

        noise_docs = []

        # Use all templates
        noise_docs.extend(self.NOISE_TEMPLATES)

        # Generate additional synthetic noise by combining and modifying templates
        while len(noise_docs) < n:
            # Strategy 1: Random selection with slight modification
            template = random.choice(self.NOISE_TEMPLATES)
            noise_docs.append(template)

            # Strategy 2: Combine two unrelated facts
            if len(noise_docs) < n:
                t1 = random.choice(self.NOISE_TEMPLATES)
                t2 = random.choice(self.NOISE_TEMPLATES)
                combined = f"{t1.split('.')[0]}, while {t2[0].lower()}{t2[1:]}"
                noise_docs.append(combined)

            # Strategy 3: Create variations
            if len(noise_docs) < n:
                template = random.choice(self.NOISE_TEMPLATES)
                variation = self._create_variation(template)
                noise_docs.append(variation)

        # Return exactly n documents
        return noise_docs[:n]

    def _create_variation(self, text: str) -> str:
        """
        Create a variation of a noise text

        Args:
            text: Original text

        Returns:
            Modified text
        """
        # Simple variations: add prefixes or suffixes
        prefixes = [
            "Research shows that ",
            "Studies indicate that ",
            "It is well known that ",
            "Experts agree that ",
            "Historical records show that ",
        ]

        suffixes = [
            " This has been verified through multiple studies.",
            " Further research continues in this area.",
            " This principle applies across various contexts.",
            " These findings have broad implications.",
        ]

        variation = text
        if random.random() < 0.5:
            variation = random.choice(prefixes) + variation[0].lower() + variation[1:]
        if random.random() < 0.5:
            variation = variation.rstrip('.') + "." + random.choice(suffixes)

        return variation

    def validate_no_overlap(self, noise_docs: List[str], fact_keywords: List[str]) -> bool:
        """
        Validate that noise documents don't overlap with fact domains

        Args:
            noise_docs: Generated noise documents
            fact_keywords: Keywords from fact documents to avoid

        Returns:
            True if no significant overlap

        Raises:
            ValueError: If overlap detected
        """
        for doc in noise_docs:
            doc_lower = doc.lower()
            for keyword in fact_keywords:
                if keyword.lower() in doc_lower:
                    raise ValueError(f"Noise document contains fact keyword: {keyword}")

        return True

    def save_to_json(self, noise_docs: List[str], filepath: str) -> None:
        """
        Save noise documents to JSON file

        Args:
            noise_docs: List of noise documents
            filepath: Output file path
        """
        data = [
            {"id": f"noise_{i+1:03d}", "text": doc}
            for i, doc in enumerate(noise_docs)
        ]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_from_json(filepath: str) -> List[str]:
        """
        Load noise documents from JSON file

        Args:
            filepath: Input file path

        Returns:
            List of noise document texts
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return [item['text'] for item in data]

    def sample_for_noise_ratio(self, n_facts: int, noise_ratio: float) -> int:
        """
        Calculate number of noise documents needed for target ratio

        noise_ratio = n_noise / (n_facts + n_noise)
        n_noise = n_facts × noise_ratio / (1 - noise_ratio)

        Args:
            n_facts: Number of core facts
            noise_ratio: Target noise ratio (0.0 to 0.99)

        Returns:
            Number of noise documents needed

        Example:
            >>> gen = NoiseGenerator()
            >>> gen.sample_for_noise_ratio(10, 0.8)
            40  # 40 noise + 10 facts = 50 total, 40/50 = 0.8
        """
        if not 0 <= noise_ratio < 1.0:
            raise ValueError(f"Noise ratio must be in [0, 1), got {noise_ratio}")

        if noise_ratio == 0:
            return 0

        n_noise = int(n_facts * noise_ratio / (1 - noise_ratio))
        return n_noise
