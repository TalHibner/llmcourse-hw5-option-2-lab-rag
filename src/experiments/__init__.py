"""
Experiment Modules

Three core experiments for RAG research:
1. Context Window - Lost in the Middle
2. Noise Impact - Performance degradation with noise
3. RAG Solution - Effectiveness of retrieval-based approach
"""

from .experiment1_context_window import ContextWindowExperiment
from .experiment2_noise_impact import NoiseImpactExperiment
from .experiment3_rag_solution import RAGSolutionExperiment

__all__ = [
    "ContextWindowExperiment",
    "NoiseImpactExperiment",
    "RAGSolutionExperiment",
]
