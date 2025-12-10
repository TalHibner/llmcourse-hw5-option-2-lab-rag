"""
Helper Utilities

Common utility functions for:
- File operations
- String manipulation
- Time formatting
- Statistical calculations
"""

import re
import time
from pathlib import Path
from typing import List, Dict, Any, Union
import statistics


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters

    Input Data:
    - filename: str - raw filename

    Output Data:
    - str - sanitized filename safe for filesystem

    Args:
        filename: Raw filename

    Returns:
        Sanitized filename

    Example:
        >>> sanitize_filename("experiment: test/run #1")
        "experiment_test_run_1"
    """
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r'[^\w\-.]', '_', filename)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if needed

    Input Data:
    - path: Union[str, Path] - directory path

    Output Data:
    - Path - validated directory path

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format

    Input Data:
    - seconds: float - duration in seconds

    Output Data:
    - str - formatted duration string

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s", "2.5s")

    Example:
        >>> format_duration(3665.5)
        "1h 1m 5.5s"
        >>> format_duration(2.5)
        "2.5s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.1f}s"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """
    Calculate descriptive statistics

    Input Data:
    - values: List[float] - numerical values

    Output Data:
    - Dict with: mean, median, std, min, max, count

    Args:
        values: List of numerical values

    Returns:
        Dictionary with statistics

    Raises:
        ValueError: If values list is empty

    Example:
        >>> calculate_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        {
            'mean': 3.0,
            'median': 3.0,
            'std': 1.58,
            'min': 1.0,
            'max': 5.0,
            'count': 5
        }
    """
    if not values:
        raise ValueError("Cannot calculate statistics for empty list")

    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks

    Input Data:
    - items: List[Any] - items to chunk
    - chunk_size: int - size of each chunk

    Output Data:
    - List[List[Any]] - list of chunks

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def retry_with_backoff(
    func: callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Any:
    """
    Retry function with exponential backoff

    Input Data:
    - func: callable - function to retry
    - max_retries: int - maximum retry attempts
    - initial_delay: float - initial delay in seconds
    - backoff_factor: float - exponential backoff multiplier

    Output Data:
    - Any - result from successful function call

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries
        backoff_factor: Exponential backoff factor

    Returns:
        Result from successful function call

    Raises:
        Exception: Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= backoff_factor

    raise last_exception


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length

    Input Data:
    - text: str - text to truncate
    - max_length: int - maximum length
    - suffix: str - suffix to append if truncated

    Output Data:
    - str - truncated text

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append if truncated

    Returns:
        Truncated text

    Example:
        >>> truncate_text("This is a long text", 10)
        "This is..."
    """
    if len(text) <= max_length:
        return text

    truncate_at = max_length - len(suffix)
    if truncate_at <= 0:
        return suffix[:max_length]

    return text[:truncate_at] + suffix


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary

    Input Data:
    - d: Dict - nested dictionary
    - parent_key: str - parent key prefix
    - sep: str - separator for keys

    Output Data:
    - Dict - flattened dictionary

    Args:
        d: Nested dictionary
        parent_key: Parent key prefix
        sep: Separator for nested keys

    Returns:
        Flattened dictionary

    Example:
        >>> flatten_dict({'a': {'b': 1, 'c': 2}})
        {'a.b': 1, 'a.c': 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
