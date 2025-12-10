"""
Experiment 1: Lost in the Middle

Demonstrates that LLMs struggle to retrieve information from the middle
of long contexts (U-shaped performance curve).

Independent Variables:
- Position of target fact in context (beginning, middle, end)
- Number of total facts in context

Dependent Variables:
- Accuracy (% correct answers)
- Response time
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..data_generation.fact_generator import Fact
from ..llm.ollama_client import OllamaClient
from ..utils.logging import setup_logger

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Single experiment result"""
    fact_id: str
    position: str
    position_index: int
    total_facts: int
    query: str
    expected_answer: str
    generated_response: str
    correct: bool
    generation_time_ms: float
    tokens: int


class ContextWindowExperiment:
    """
    Experiment 1: Context Window (Lost in the Middle)

    Setup Data:
    - llm_client: OllamaClient - LLM for generation
    - facts: List[Fact] - fact dataset
    - n_facts: int - number of facts in context
    - position_categories: List[str] - positions to test
    - random_seed: int - for reproducibility

    Input Data:
    - (automatically loads facts from dataset)

    Output Data:
    - List[ExperimentResult] - results for each trial
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        facts: List[Fact],
        n_facts: int = 25,
        position_categories: Optional[List[str]] = None,
        random_seed: int = 42
    ):
        """
        Initialize experiment

        Args:
            llm_client: Ollama client
            facts: List of facts to test
            n_facts: Number of facts in context
            position_categories: Positions to test (beginning, middle, end)
            random_seed: Random seed
        """
        self.llm_client = llm_client
        self.facts = facts
        self.n_facts = n_facts
        self.position_categories = position_categories or ["beginning", "middle", "end"]
        self.random_seed = random_seed
        self.random = random.Random(random_seed)

        self.exp_logger = setup_logger(
            "experiment1",
            log_file="logs/experiment1_context_window.log"
        )

        logger.info(f"Initialized ContextWindowExperiment with {len(facts)} facts")

    def _get_position_index(self, position: str, total: int) -> int:
        """
        Get index for position category

        Args:
            position: Position category (beginning, middle, end)
            total: Total number of facts

        Returns:
            Index for target fact
        """
        if position == "beginning":
            return 0
        elif position == "end":
            return total - 1
        elif position == "middle":
            return total // 2
        else:
            raise ValueError(f"Invalid position: {position}")

    def _build_context(self, facts: List[Fact]) -> str:
        """
        Build context from facts

        Args:
            facts: List of facts

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, fact in enumerate(facts, 1):
            context_parts.append(f"{i}. {fact.fact_text}")

        return "\n".join(context_parts)

    def _check_correctness(self, response: str, expected_answer: str) -> bool:
        """
        Check if response contains expected answer

        Args:
            response: Generated response
            expected_answer: Expected answer

        Returns:
            True if correct, False otherwise
        """
        response_lower = response.lower()
        expected_lower = expected_answer.lower()

        # Simple substring match
        return expected_lower in response_lower

    def run_single_trial(
        self,
        target_fact: Fact,
        position: str
    ) -> ExperimentResult:
        """
        Run single trial

        Args:
            target_fact: Fact to query
            position: Position category

        Returns:
            Experiment result
        """
        # Select random facts (excluding target)
        other_facts = [f for f in self.facts if f.id != target_fact.id]
        selected_facts = self.random.sample(other_facts, self.n_facts - 1)

        # Insert target fact at specified position
        position_index = self._get_position_index(position, self.n_facts)
        selected_facts.insert(position_index, target_fact)

        # Build context
        context = self._build_context(selected_facts)

        # Generate query
        query = target_fact.question

        # Call LLM
        self.exp_logger.debug(
            f"Querying with fact at position {position} (index {position_index})"
        )

        result = self.llm_client.generate(
            prompt=query,
            context=context,
            system_prompt="Answer the question based on the provided context. Be concise."
        )

        # Check correctness
        correct = self._check_correctness(result['response'], target_fact.answer)

        self.exp_logger.debug(
            f"Expected: {target_fact.answer}, "
            f"Got: {result['response'][:50]}..., "
            f"Correct: {correct}"
        )

        return ExperimentResult(
            fact_id=target_fact.id,
            position=position,
            position_index=position_index,
            total_facts=self.n_facts,
            query=query,
            expected_answer=target_fact.answer,
            generated_response=result['response'],
            correct=correct,
            generation_time_ms=result['generation_time_ms'],
            tokens=result['tokens']
        )

    def run(self, n_runs: int = 5) -> List[ExperimentResult]:
        """
        Run complete experiment

        Args:
            n_runs: Number of runs per position per fact

        Returns:
            List of all experiment results
        """
        self.exp_logger.experiment_start(
            "Context Window (Lost in the Middle)",
            {
                "n_facts": self.n_facts,
                "positions": self.position_categories,
                "n_runs": n_runs,
                "total_trials": len(self.facts) * len(self.position_categories) * n_runs
            }
        )

        results = []
        total_trials = len(self.facts) * len(self.position_categories) * n_runs

        trial_num = 0
        for run in range(n_runs):
            self.exp_logger.info(f"\nRun {run + 1}/{n_runs}")

            for fact in self.facts:
                for position in self.position_categories:
                    trial_num += 1

                    self.exp_logger.progress(
                        trial_num,
                        total_trials,
                        f"Fact {fact.id}, Position {position}"
                    )

                    result = self.run_single_trial(fact, position)
                    results.append(result)

        # Calculate summary statistics
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        accuracy = (correct / total) * 100 if total > 0 else 0

        # Per-position accuracy
        position_stats = {}
        for position in self.position_categories:
            pos_results = [r for r in results if r.position == position]
            pos_correct = sum(1 for r in pos_results if r.correct)
            pos_accuracy = (pos_correct / len(pos_results)) * 100 if pos_results else 0
            position_stats[f"{position}_accuracy"] = f"{pos_accuracy:.1f}%"

        self.exp_logger.experiment_end(
            "Context Window (Lost in the Middle)",
            {
                "total_trials": total,
                "correct": correct,
                "overall_accuracy": f"{accuracy:.1f}%",
                **position_stats
            }
        )

        return results

    def save_results(self, results: List[ExperimentResult], output_path: str) -> None:
        """
        Save results to JSON

        Args:
            results: Experiment results
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results_data = []
        for result in results:
            results_data.append({
                'fact_id': result.fact_id,
                'position': result.position,
                'position_index': result.position_index,
                'total_facts': result.total_facts,
                'query': result.query,
                'expected_answer': result.expected_answer,
                'generated_response': result.generated_response,
                'correct': result.correct,
                'generation_time_ms': result.generation_time_ms,
                'tokens': result.tokens
            })

        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        self.exp_logger.info(f"Saved {len(results)} results to {output_path}")

    @staticmethod
    def load_results(input_path: str) -> List[ExperimentResult]:
        """
        Load results from JSON

        Args:
            input_path: Input file path

        Returns:
            List of experiment results
        """
        with open(input_path) as f:
            data = json.load(f)

        results = []
        for item in data:
            results.append(ExperimentResult(**item))

        return results
