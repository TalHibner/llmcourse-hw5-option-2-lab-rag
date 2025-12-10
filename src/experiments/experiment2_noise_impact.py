"""
Experiment 2: Noise and Irrelevance

Measures how noise (irrelevant documents) impacts LLM accuracy.
Demonstrates degradation as noise ratio increases.

Independent Variables:
- Noise ratio (0%, 20%, 40%, 60%, 80%, 90%)
- Number of core facts (constant)

Dependent Variables:
- Accuracy (% correct answers)
- Response time
- Hallucination rate
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
class NoiseExperimentResult:
    """Single noise experiment result"""
    fact_id: str
    noise_ratio: float
    n_core_facts: int
    n_noise_docs: int
    query: str
    expected_answer: str
    generated_response: str
    correct: bool
    hallucinated: bool
    generation_time_ms: float
    tokens: int


class NoiseImpactExperiment:
    """
    Experiment 2: Noise and Irrelevance

    Setup Data:
    - llm_client: OllamaClient - LLM for generation
    - facts: List[Fact] - core facts dataset
    - noise_docs: List[str] - noise document pool
    - n_core_facts: int - number of core facts
    - noise_levels: List[float] - noise ratios to test
    - random_seed: int - for reproducibility

    Input Data:
    - (automatically loads facts and noise from datasets)

    Output Data:
    - List[NoiseExperimentResult] - results for each trial
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        facts: List[Fact],
        noise_docs: List[str],
        n_core_facts: int = 10,
        noise_levels: Optional[List[float]] = None,
        random_seed: int = 42
    ):
        """
        Initialize experiment

        Args:
            llm_client: Ollama client
            facts: List of core facts
            noise_docs: Pool of noise documents
            n_core_facts: Number of core facts in context
            noise_levels: Noise ratios to test
            random_seed: Random seed
        """
        self.llm_client = llm_client
        self.facts = facts
        self.noise_docs = noise_docs
        self.n_core_facts = n_core_facts
        self.noise_levels = noise_levels or [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        self.random_seed = random_seed
        self.random = random.Random(random_seed)

        self.exp_logger = setup_logger(
            "experiment2",
            log_file="logs/experiment2_noise_impact.log"
        )

        logger.info(
            f"Initialized NoiseImpactExperiment with {len(facts)} facts, "
            f"{len(noise_docs)} noise docs"
        )

    def _calculate_noise_count(self, noise_ratio: float) -> int:
        """
        Calculate number of noise documents needed

        Args:
            noise_ratio: Target noise ratio (0.0 to 1.0)

        Returns:
            Number of noise documents
        """
        if noise_ratio == 0.0:
            return 0

        # n_noise = n_facts * ratio / (1 - ratio)
        n_noise = int(self.n_core_facts * noise_ratio / (1 - noise_ratio))
        return n_noise

    def _build_context(
        self,
        core_facts: List[Fact],
        noise_docs: List[str]
    ) -> str:
        """
        Build context from facts and noise

        Args:
            core_facts: Core fact documents
            noise_docs: Noise documents

        Returns:
            Formatted context string
        """
        # Combine all documents
        all_docs = []

        for fact in core_facts:
            all_docs.append(fact.fact_text)

        all_docs.extend(noise_docs)

        # Shuffle to mix facts and noise
        self.random.shuffle(all_docs)

        # Format as numbered list
        context_parts = []
        for i, doc in enumerate(all_docs, 1):
            context_parts.append(f"{i}. {doc}")

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

        return expected_lower in response_lower

    def _check_hallucination(
        self,
        response: str,
        expected_answer: str,
        available_facts: List[Fact]
    ) -> bool:
        """
        Check if response contains hallucinated information

        Args:
            response: Generated response
            expected_answer: Expected answer
            available_facts: Facts available in context

        Returns:
            True if hallucinated, False otherwise
        """
        # If answer is correct, no hallucination
        if self._check_correctness(response, expected_answer):
            return False

        # If response contains answer from different fact, it's hallucination
        response_lower = response.lower()
        for fact in available_facts:
            if fact.answer.lower() in response_lower and fact.answer.lower() != expected_answer.lower():
                return True

        return False

    def run_single_trial(
        self,
        target_fact: Fact,
        noise_ratio: float
    ) -> NoiseExperimentResult:
        """
        Run single trial

        Args:
            target_fact: Fact to query
            noise_ratio: Noise ratio for this trial

        Returns:
            Experiment result
        """
        # Select random core facts (including target)
        selected_facts = self.random.sample(self.facts, self.n_core_facts)
        if target_fact not in selected_facts:
            selected_facts[0] = target_fact

        # Calculate and sample noise documents
        n_noise = self._calculate_noise_count(noise_ratio)
        selected_noise = self.random.sample(self.noise_docs, min(n_noise, len(self.noise_docs)))

        # Build context
        context = self._build_context(selected_facts, selected_noise)

        # Generate query
        query = target_fact.question

        # Call LLM
        self.exp_logger.debug(
            f"Querying with noise_ratio={noise_ratio:.1f} "
            f"({n_noise} noise docs + {self.n_core_facts} facts)"
        )

        result = self.llm_client.generate(
            prompt=query,
            context=context,
            system_prompt="Answer the question based on the provided context. Be concise."
        )

        # Check correctness and hallucination
        correct = self._check_correctness(result['response'], target_fact.answer)
        hallucinated = self._check_hallucination(
            result['response'],
            target_fact.answer,
            selected_facts
        )

        self.exp_logger.debug(
            f"Expected: {target_fact.answer}, "
            f"Got: {result['response'][:50]}..., "
            f"Correct: {correct}, Hallucinated: {hallucinated}"
        )

        return NoiseExperimentResult(
            fact_id=target_fact.id,
            noise_ratio=noise_ratio,
            n_core_facts=self.n_core_facts,
            n_noise_docs=n_noise,
            query=query,
            expected_answer=target_fact.answer,
            generated_response=result['response'],
            correct=correct,
            hallucinated=hallucinated,
            generation_time_ms=result['generation_time_ms'],
            tokens=result['tokens']
        )

    def run(self, n_runs: int = 5) -> List[NoiseExperimentResult]:
        """
        Run complete experiment

        Args:
            n_runs: Number of runs per noise level per fact

        Returns:
            List of all experiment results
        """
        self.exp_logger.experiment_start(
            "Noise and Irrelevance",
            {
                "n_core_facts": self.n_core_facts,
                "noise_levels": self.noise_levels,
                "n_runs": n_runs,
                "total_trials": len(self.facts) * len(self.noise_levels) * n_runs
            }
        )

        results = []
        total_trials = len(self.facts) * len(self.noise_levels) * n_runs

        trial_num = 0
        for run in range(n_runs):
            self.exp_logger.info(f"\nRun {run + 1}/{n_runs}")

            for fact in self.facts:
                for noise_ratio in self.noise_levels:
                    trial_num += 1

                    self.exp_logger.progress(
                        trial_num,
                        total_trials,
                        f"Fact {fact.id}, Noise {noise_ratio:.0%}"
                    )

                    result = self.run_single_trial(fact, noise_ratio)
                    results.append(result)

        # Calculate summary statistics
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        hallucinated = sum(1 for r in results if r.hallucinated)
        accuracy = (correct / total) * 100 if total > 0 else 0
        hallucination_rate = (hallucinated / total) * 100 if total > 0 else 0

        # Per-noise-level accuracy
        noise_stats = {}
        for noise_ratio in self.noise_levels:
            noise_results = [r for r in results if r.noise_ratio == noise_ratio]
            noise_correct = sum(1 for r in noise_results if r.correct)
            noise_accuracy = (noise_correct / len(noise_results)) * 100 if noise_results else 0
            noise_stats[f"noise_{noise_ratio:.0%}_accuracy"] = f"{noise_accuracy:.1f}%"

        self.exp_logger.experiment_end(
            "Noise and Irrelevance",
            {
                "total_trials": total,
                "correct": correct,
                "overall_accuracy": f"{accuracy:.1f}%",
                "hallucination_rate": f"{hallucination_rate:.1f}%",
                **noise_stats
            }
        )

        return results

    def save_results(self, results: List[NoiseExperimentResult], output_path: str) -> None:
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
                'noise_ratio': result.noise_ratio,
                'n_core_facts': result.n_core_facts,
                'n_noise_docs': result.n_noise_docs,
                'query': result.query,
                'expected_answer': result.expected_answer,
                'generated_response': result.generated_response,
                'correct': result.correct,
                'hallucinated': result.hallucinated,
                'generation_time_ms': result.generation_time_ms,
                'tokens': result.tokens
            })

        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        self.exp_logger.info(f"Saved {len(results)} results to {output_path}")

    @staticmethod
    def load_results(input_path: str) -> List[NoiseExperimentResult]:
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
            results.append(NoiseExperimentResult(**item))

        return results
