"""
Experiment 3: RAG Solution

Demonstrates that RAG maintains >90% accuracy even with high noise levels
by retrieving only relevant documents before generation.

Independent Variables:
- Noise ratio (same levels as Experiment 2)
- top_k retrieval parameter
- Reranking enabled/disabled

Dependent Variables:
- Accuracy (% correct answers)
- Retrieval precision (% relevant in top-k)
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
from ..rag.vector_store import VectorStore, Document
from ..rag.retriever import RAGRetriever
from ..utils.logging import setup_logger

logger = logging.getLogger(__name__)


@dataclass
class RAGExperimentResult:
    """Single RAG experiment result"""
    fact_id: str
    noise_ratio: float
    top_k: int
    reranking_enabled: bool
    n_core_facts: int
    n_noise_docs: int
    query: str
    expected_answer: str
    generated_response: str
    correct: bool
    relevant_retrieved: bool
    retrieval_precision: float
    generation_time_ms: float
    tokens: int
    retrieved_doc_ids: List[str]


class RAGSolutionExperiment:
    """
    Experiment 3: RAG Solution

    Setup Data:
    - llm_client: OllamaClient - LLM for generation
    - vector_store: VectorStore - vector store for retrieval
    - facts: List[Fact] - core facts dataset
    - noise_docs: List[str] - noise document pool
    - n_core_facts: int - number of core facts
    - noise_levels: List[float] - noise ratios to test
    - top_k_values: List[int] - top-k values to test
    - reranking_enabled: bool - whether to use reranking
    - random_seed: int - for reproducibility

    Input Data:
    - (automatically loads facts and noise from datasets)

    Output Data:
    - List[RAGExperimentResult] - results for each trial
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        vector_store: VectorStore,
        facts: List[Fact],
        noise_docs: List[str],
        n_core_facts: int = 10,
        noise_levels: Optional[List[float]] = None,
        top_k_values: Optional[List[int]] = None,
        reranking_enabled: bool = False,
        random_seed: int = 42
    ):
        """
        Initialize experiment

        Args:
            llm_client: Ollama client
            vector_store: Vector store for retrieval
            facts: List of core facts
            noise_docs: Pool of noise documents
            n_core_facts: Number of core facts
            noise_levels: Noise ratios to test
            top_k_values: Top-k values to test
            reranking_enabled: Enable reranking
            random_seed: Random seed
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.facts = facts
        self.noise_docs = noise_docs
        self.n_core_facts = n_core_facts
        self.noise_levels = noise_levels or [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        self.top_k_values = top_k_values or [3, 5]
        self.reranking_enabled = reranking_enabled
        self.random_seed = random_seed
        self.random = random.Random(random_seed)

        self.exp_logger = setup_logger(
            "experiment3",
            log_file="logs/experiment3_rag_solution.log"
        )

        logger.info(
            f"Initialized RAGSolutionExperiment with {len(facts)} facts, "
            f"{len(noise_docs)} noise docs"
        )

    def _calculate_noise_count(self, noise_ratio: float) -> int:
        """
        Calculate number of noise documents needed

        Args:
            noise_ratio: Target noise ratio

        Returns:
            Number of noise documents
        """
        if noise_ratio == 0.0:
            return 0

        n_noise = int(self.n_core_facts * noise_ratio / (1 - noise_ratio))
        return n_noise

    def _populate_vector_store(
        self,
        core_facts: List[Fact],
        noise_docs: List[str]
    ) -> None:
        """
        Populate vector store with facts and noise

        Args:
            core_facts: Core fact documents
            noise_docs: Noise documents
        """
        # Clear existing documents
        self.vector_store.clear()

        # Create documents
        documents = []

        # Add facts
        for fact in core_facts:
            documents.append(Document(
                id=fact.id,
                text=fact.fact_text,
                metadata={
                    'type': 'fact',
                    'category': fact.category,
                    'answer': fact.answer
                }
            ))

        # Add noise
        for i, noise_text in enumerate(noise_docs):
            documents.append(Document(
                id=f"noise_{i}",
                text=noise_text,
                metadata={'type': 'noise'}
            ))

        # Add to vector store
        self.vector_store.add_documents(documents)

        self.exp_logger.debug(
            f"Populated vector store with {len(core_facts)} facts + "
            f"{len(noise_docs)} noise docs"
        )

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

    def _calculate_retrieval_precision(
        self,
        retrieved_docs: List[Dict[str, Any]],
        target_fact_id: str
    ) -> tuple[bool, float]:
        """
        Calculate retrieval precision

        Args:
            retrieved_docs: Retrieved documents
            target_fact_id: ID of target fact

        Returns:
            Tuple of (relevant_retrieved, precision)
        """
        if not retrieved_docs:
            return False, 0.0

        # Check if relevant document was retrieved
        relevant_retrieved = any(
            doc['id'] == target_fact_id for doc in retrieved_docs
        )

        # Calculate precision (% of retrieved docs that are facts, not noise)
        fact_count = sum(
            1 for doc in retrieved_docs
            if doc['metadata'].get('type') == 'fact'
        )
        precision = fact_count / len(retrieved_docs)

        return relevant_retrieved, precision

    def run_single_trial(
        self,
        target_fact: Fact,
        noise_ratio: float,
        top_k: int
    ) -> RAGExperimentResult:
        """
        Run single trial

        Args:
            target_fact: Fact to query
            noise_ratio: Noise ratio for this trial
            top_k: Number of documents to retrieve

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

        # Populate vector store
        self._populate_vector_store(selected_facts, selected_noise)

        # Create RAG retriever
        rag_retriever = RAGRetriever(
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            top_k=top_k,
            reranking_enabled=self.reranking_enabled
        )

        # Generate query
        query = target_fact.question

        # Call RAG pipeline
        self.exp_logger.debug(
            f"Querying RAG with noise_ratio={noise_ratio:.1f}, top_k={top_k} "
            f"({n_noise} noise docs + {self.n_core_facts} facts)"
        )

        result = rag_retriever.generate(
            query=query,
            system_prompt="Answer the question based on the provided context. Be concise."
        )

        # Check correctness
        correct = self._check_correctness(result['response'], target_fact.answer)

        # Calculate retrieval precision
        relevant_retrieved, retrieval_precision = self._calculate_retrieval_precision(
            result['retrieved_docs'],
            target_fact.id
        )

        retrieved_doc_ids = [doc['id'] for doc in result['retrieved_docs']]

        self.exp_logger.debug(
            f"Expected: {target_fact.answer}, "
            f"Got: {result['response'][:50]}..., "
            f"Correct: {correct}, "
            f"Relevant retrieved: {relevant_retrieved}, "
            f"Precision: {retrieval_precision:.2f}"
        )

        return RAGExperimentResult(
            fact_id=target_fact.id,
            noise_ratio=noise_ratio,
            top_k=top_k,
            reranking_enabled=self.reranking_enabled,
            n_core_facts=self.n_core_facts,
            n_noise_docs=n_noise,
            query=query,
            expected_answer=target_fact.answer,
            generated_response=result['response'],
            correct=correct,
            relevant_retrieved=relevant_retrieved,
            retrieval_precision=retrieval_precision,
            generation_time_ms=result['generation_time_ms'],
            tokens=result['tokens'],
            retrieved_doc_ids=retrieved_doc_ids
        )

    def run(self, n_runs: int = 5) -> List[RAGExperimentResult]:
        """
        Run complete experiment

        Args:
            n_runs: Number of runs per configuration

        Returns:
            List of all experiment results
        """
        self.exp_logger.experiment_start(
            "RAG Solution",
            {
                "n_core_facts": self.n_core_facts,
                "noise_levels": self.noise_levels,
                "top_k_values": self.top_k_values,
                "reranking_enabled": self.reranking_enabled,
                "n_runs": n_runs,
                "total_trials": len(self.facts) * len(self.noise_levels) * len(self.top_k_values) * n_runs
            }
        )

        results = []
        total_trials = len(self.facts) * len(self.noise_levels) * len(self.top_k_values) * n_runs

        trial_num = 0
        for run in range(n_runs):
            self.exp_logger.info(f"\nRun {run + 1}/{n_runs}")

            for fact in self.facts:
                for noise_ratio in self.noise_levels:
                    for top_k in self.top_k_values:
                        trial_num += 1

                        self.exp_logger.progress(
                            trial_num,
                            total_trials,
                            f"Fact {fact.id}, Noise {noise_ratio:.0%}, top_k={top_k}"
                        )

                        result = self.run_single_trial(fact, noise_ratio, top_k)
                        results.append(result)

        # Calculate summary statistics
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        relevant_retrieved = sum(1 for r in results if r.relevant_retrieved)
        accuracy = (correct / total) * 100 if total > 0 else 0
        retrieval_success = (relevant_retrieved / total) * 100 if total > 0 else 0

        # Per-noise-level accuracy
        noise_stats = {}
        for noise_ratio in self.noise_levels:
            noise_results = [r for r in results if r.noise_ratio == noise_ratio]
            noise_correct = sum(1 for r in noise_results if r.correct)
            noise_accuracy = (noise_correct / len(noise_results)) * 100 if noise_results else 0
            noise_stats[f"noise_{noise_ratio:.0%}_accuracy"] = f"{noise_accuracy:.1f}%"

        self.exp_logger.experiment_end(
            "RAG Solution",
            {
                "total_trials": total,
                "correct": correct,
                "overall_accuracy": f"{accuracy:.1f}%",
                "retrieval_success_rate": f"{retrieval_success:.1f}%",
                **noise_stats
            }
        )

        return results

    def save_results(self, results: List[RAGExperimentResult], output_path: str) -> None:
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
                'top_k': result.top_k,
                'reranking_enabled': result.reranking_enabled,
                'n_core_facts': result.n_core_facts,
                'n_noise_docs': result.n_noise_docs,
                'query': result.query,
                'expected_answer': result.expected_answer,
                'generated_response': result.generated_response,
                'correct': result.correct,
                'relevant_retrieved': result.relevant_retrieved,
                'retrieval_precision': result.retrieval_precision,
                'generation_time_ms': result.generation_time_ms,
                'tokens': result.tokens,
                'retrieved_doc_ids': result.retrieved_doc_ids
            })

        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        self.exp_logger.info(f"Saved {len(results)} results to {output_path}")

    @staticmethod
    def load_results(input_path: str) -> List[RAGExperimentResult]:
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
            results.append(RAGExperimentResult(**item))

        return results
