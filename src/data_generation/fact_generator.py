"""
Synthetic Fact Generator

Generates diverse, verifiable fact documents for RAG experiments.
Each fact has a single correct answer to avoid ambiguity.
"""

import random
from dataclasses import dataclass, asdict
from typing import List, Dict
import json


@dataclass
class Fact:
    """
    Building block for fact representation

    Input Data:
    - All fields below

    Output Data:
    - JSON-serializable fact object

    Validation:
    - All fields must be non-empty strings
    - ID must be unique
    - Answer must appear in fact_text
    """
    id: str
    category: str
    fact_text: str
    question: str
    answer: str
    metadata: Dict[str, str]

    def validate(self) -> bool:
        """
        Validate fact structure

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if not self.id:
            raise ValueError("Fact ID cannot be empty")
        if not self.category:
            raise ValueError("Category cannot be empty")
        if not self.fact_text or len(self.fact_text) < 10:
            raise ValueError("Fact text too short or empty")
        if not self.question.strip():
            raise ValueError("Question cannot be empty")
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")
        # Verify answer appears in fact
        if self.answer.lower() not in self.fact_text.lower():
            raise ValueError(f"Answer '{self.answer}' not found in fact text")

        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class FactGenerator:
    """
    Generates synthetic fact documents

    Input Data:
    - n: int - number of facts to generate (default 25)

    Setup Data:
    - seed: int - random seed for reproducibility
    - categories: List[str] - fact categories to use

    Output Data:
    - List[Fact] - generated facts
    """

    # Fact templates organized by category
    GEOGRAPHY_FACTS = [
        ("Paris", "France", "What is the capital of France?"),
        ("Tokyo", "Japan", "What is the capital of Japan?"),
        ("London", "United Kingdom", "What is the capital of the United Kingdom?"),
        ("Berlin", "Germany", "What is the capital of Germany?"),
        ("Rome", "Italy", "What is the capital of Italy?"),
        ("Madrid", "Spain", "What is the capital of Spain?"),
    ]

    SCIENCE_FACTS = [
        ("Gold", "Au", "What is the chemical symbol for Gold?"),
        ("Iron", "Fe", "What is the chemical symbol for Iron?"),
        ("Silver", "Ag", "What is the chemical symbol for Silver?"),
        ("Copper", "Cu", "What is the chemical symbol for Copper?"),
        ("Oxygen", "O", "What is the chemical symbol for Oxygen?"),
        ("Helium", "He", "What is the chemical symbol for Helium?"),
    ]

    HISTORY_FACTS = [
        ("World War II", "1945", "In what year did World War II end?"),
        ("Declaration of Independence", "1776", "In what year was the Declaration of Independence signed?"),
        ("First Moon Landing", "1969", "In what year was the first moon landing?"),
        ("Fall of Berlin Wall", "1989", "In what year did the Berlin Wall fall?"),
        ("French Revolution", "1789", "In what year did the French Revolution begin?"),
    ]

    MATH_FACTS = [
        ("Pi", "3.14159", "What is the approximate value of Pi?"),
        ("Euler's number", "2.71828", "What is the approximate value of Euler's number (e)?"),
        ("Golden ratio", "1.618", "What is the approximate value of the golden ratio (phi)?"),
        ("Square root of 2", "1.414", "What is the approximate value of the square root of 2?"),
        ("Planck's constant", "6.626 × 10^-34", "What is Planck's constant?"),
    ]

    LITERATURE_FACTS = [
        ("1984", "George Orwell", "Who wrote the novel 1984?"),
        ("Pride and Prejudice", "Jane Austen", "Who wrote Pride and Prejudice?"),
        ("To Kill a Mockingbird", "Harper Lee", "Who wrote To Kill a Mockingbird?"),
        ("The Great Gatsby", "F. Scott Fitzgerald", "Who wrote The Great Gatsby?"),
        ("Moby Dick", "Herman Melville", "Who wrote Moby Dick?"),
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize fact generator

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

        self.categories = {
            'geography': self.GEOGRAPHY_FACTS,
            'science': self.SCIENCE_FACTS,
            'history': self.HISTORY_FACTS,
            'mathematics': self.MATH_FACTS,
            'literature': self.LITERATURE_FACTS,
        }

    def generate_facts(self, n: int = 25) -> List[Fact]:
        """
        Generate n diverse facts across all categories

        Args:
            n: Number of facts to generate (20-30 recommended)

        Returns:
            List of Fact objects

        Raises:
            ValueError: If n is out of valid range
        """
        if not 20 <= n <= 30:
            raise ValueError(f"Number of facts must be between 20 and 30, got {n}")

        facts = []
        facts_per_category = n // len(self.categories)
        extra_facts = n % len(self.categories)

        fact_id_counter = 1

        for category_name, templates in self.categories.items():
            # Determine how many facts from this category
            n_from_category = facts_per_category
            if extra_facts > 0:
                n_from_category += 1
                extra_facts -= 1

            # Sample templates
            sampled = random.sample(templates, min(n_from_category, len(templates)))

            for entity1, entity2, question in sampled:
                # Create fact text based on category
                if category_name == 'geography':
                    fact_text = f"{entity1} is the capital of {entity2}."
                elif category_name == 'science':
                    fact_text = f"The chemical symbol for {entity1} is {entity2}."
                elif category_name == 'history':
                    fact_text = f"{entity1} ended in {entity2}." if "World War" in entity1 else f"{entity1} was in {entity2}."
                elif category_name == 'mathematics':
                    fact_text = f"{entity1} is approximately {entity2}."
                elif category_name == 'literature':
                    fact_text = f"The novel {entity1} was written by {entity2}."

                fact = Fact(
                    id=f"fact_{fact_id_counter:03d}",
                    category=category_name,
                    fact_text=fact_text,
                    question=question,
                    answer=entity2 if category_name != 'literature' else entity2,
                    metadata={
                        'complexity': 'simple',
                        'entity_primary': entity1,
                        'entity_secondary': entity2,
                    }
                )

                # Validate before adding
                fact.validate()
                facts.append(fact)
                fact_id_counter += 1

        # Shuffle to mix categories
        random.shuffle(facts)

        return facts

    def ensure_diversity(self, facts: List[Fact]) -> bool:
        """
        Validate that facts are semantically diverse (no duplicates)

        Args:
            facts: List of facts to check

        Returns:
            True if diverse

        Raises:
            ValueError: If duplicates found
        """
        fact_texts = [f.fact_text for f in facts]
        questions = [f.question for f in facts]

        # Check for duplicate fact texts
        if len(fact_texts) != len(set(fact_texts)):
            raise ValueError("Duplicate fact texts found")

        # Check for duplicate questions
        if len(questions) != len(set(questions)):
            raise ValueError("Duplicate questions found")

        # Check for duplicate IDs
        ids = [f.id for f in facts]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs found")

        return True

    def save_to_json(self, facts: List[Fact], filepath: str) -> None:
        """
        Save facts to JSON file

        Args:
            facts: List of facts
            filepath: Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([f.to_dict() for f in facts], f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_from_json(filepath: str) -> List[Fact]:
        """
        Load facts from JSON file

        Args:
            filepath: Input file path

        Returns:
            List of Fact objects
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        facts = []
        for item in data:
            fact = Fact(**item)
            fact.validate()
            facts.append(fact)

        return facts
