"""
Structured Logging System

Provides consistent logging across all experiments with:
- Experiment tracking
- Performance metrics
- Error handling
- Structured output
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ExperimentLogger:
    """
    Custom logger for experiment tracking

    Setup Data:
    - name: str - logger name
    - log_file: Optional[str] - file path for logs
    - level: int - logging level

    Output Data:
    - Structured log messages with timestamps
    - Experiment metrics
    - Error traces
    """

    def __init__(self, name: str, log_file: Optional[str] = None, level: int = logging.INFO):
        """
        Initialize experiment logger

        Args:
            name: Logger name (typically module name)
            log_file: Optional file path for logging
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error message"""
        self.logger.error(message, exc_info=exc_info)

    def experiment_start(self, experiment_name: str, config: dict) -> None:
        """
        Log experiment start

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting Experiment: {experiment_name}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 80)

    def experiment_end(self, experiment_name: str, results: dict) -> None:
        """
        Log experiment completion

        Args:
            experiment_name: Name of the experiment
            results: Results dictionary
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Completed Experiment: {experiment_name}")
        self.logger.info("Results Summary:")
        for key, value in results.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 80)

    def metric(self, name: str, value: float, unit: str = "") -> None:
        """
        Log performance metric

        Args:
            name: Metric name
            value: Metric value
            unit: Optional unit
        """
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"Metric [{name}]: {value:.4f}{unit_str}")

    def progress(self, current: int, total: int, description: str = "") -> None:
        """
        Log progress

        Args:
            current: Current iteration
            total: Total iterations
            description: Optional description
        """
        pct = (current / total) * 100 if total > 0 else 0
        desc_str = f" - {description}" if description else ""
        self.logger.info(f"Progress: {current}/{total} ({pct:.1f}%){desc_str}")


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> ExperimentLogger:
    """
    Set up and return an experiment logger

    Args:
        name: Logger name
        log_file: Optional file path for logs
        level: Logging level

    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(name, log_file, level)


def get_logger(name: str) -> logging.Logger:
    """
    Get standard Python logger

    Args:
        name: Logger name

    Returns:
        Standard logging.Logger instance
    """
    return logging.getLogger(name)
