"""
Configuration Management System

Loads configuration from config.yaml and environment variables.
Provides type-safe access to all settings.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """LLM configuration settings"""
    provider: str
    model_name: str
    base_url: str
    temperature: float
    max_tokens: int
    timeout: int


@dataclass
class EmbeddingConfig:
    """Embedding configuration settings"""
    model_name: str
    dimension: int
    batch_size: int


@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    type: str
    persist_directory: str
    collection_name: str
    distance_metric: str


@dataclass
class Experiment1Config:
    """Experiment 1 specific configuration"""
    n_facts: int
    position_categories: List[str]


@dataclass
class Experiment2Config:
    """Experiment 2 specific configuration"""
    n_core_facts: int
    noise_levels: List[float]
    n_noise_pool: int


@dataclass
class Experiment3Config:
    """Experiment 3 specific configuration"""
    top_k_values: List[int]
    reranking_enabled: bool


@dataclass
class ExperimentConfig:
    """Overall experiment configuration"""
    random_seed: int
    n_runs: int
    experiment1: Experiment1Config
    experiment2: Experiment2Config
    experiment3: Experiment3Config


@dataclass
class AnalysisConfig:
    """Statistical analysis configuration"""
    confidence_level: float
    significance_alpha: float


@dataclass
class VisualizationConfig:
    """Visualization settings"""
    dpi: int
    figure_size: List[int]
    style: str
    palette: str
    save_formats: List[str]


@dataclass
class Settings:
    """
    Global settings container

    Loads from config/config.yaml and environment variables.
    Provides type-safe access to all configuration values.
    """
    llm: LLMConfig
    embeddings: EmbeddingConfig
    vector_store: VectorStoreConfig
    experiments: ExperimentConfig
    analysis: AnalysisConfig
    visualization: VisualizationConfig
    paths: Dict[str, str]

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> "Settings":
        """
        Load configuration from YAML file

        Args:
            config_path: Path to configuration YAML file

        Returns:
            Settings object with all configuration loaded

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Override with environment variables if present
        ollama_host = os.getenv('OLLAMA_HOST')
        if ollama_host:
            config['llm']['base_url'] = ollama_host

        ollama_model = os.getenv('OLLAMA_MODEL')
        if ollama_model:
            config['llm']['model_name'] = ollama_model

        random_seed = os.getenv('RANDOM_SEED')
        if random_seed:
            config['experiments']['random_seed'] = int(random_seed)

        # Construct nested dataclasses
        return cls(
            llm=LLMConfig(**config['llm']),
            embeddings=EmbeddingConfig(**config['embeddings']),
            vector_store=VectorStoreConfig(**config['vector_store']),
            experiments=ExperimentConfig(
                random_seed=config['experiments']['random_seed'],
                n_runs=config['experiments']['n_runs'],
                experiment1=Experiment1Config(**config['experiments']['experiment1']),
                experiment2=Experiment2Config(**config['experiments']['experiment2']),
                experiment3=Experiment3Config(**config['experiments']['experiment3']),
            ),
            analysis=AnalysisConfig(**config['analysis']),
            visualization=VisualizationConfig(**config['visualization']),
            paths=config['paths']
        )

    def get_path(self, key: str) -> Path:
        """
        Get path from configuration as Path object

        Args:
            key: Path key from config

        Returns:
            Path object
        """
        return Path(self.paths.get(key, '.'))


# Global settings instance
# Can be imported anywhere: from src.config.settings import settings
try:
    settings = Settings.load()
except FileNotFoundError:
    # During initial setup, config might not exist yet
    # Will be loaded when config file is created
    settings = None
