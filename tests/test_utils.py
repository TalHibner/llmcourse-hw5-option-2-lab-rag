"""
Tests for utility modules
"""

import pytest
from pathlib import Path
from src.utils.helpers import (
    sanitize_filename,
    ensure_dir,
    format_duration,
    calculate_stats,
    chunk_list,
    truncate_text,
    flatten_dict
)
from src.utils.logging import setup_logger


class TestHelpers:
    """Test helper functions"""

    def test_sanitize_filename(self):
        """Test filename sanitization"""
        assert sanitize_filename("test file.txt") == "test_file.txt"
        assert sanitize_filename("test/file:name") == "test_file_name"
        assert sanitize_filename("test___file") == "test_file"

    def test_ensure_dir(self, tmp_path):
        """Test directory creation"""
        test_dir = tmp_path / "test" / "nested"
        result = ensure_dir(test_dir)

        assert test_dir.exists()
        assert test_dir.is_dir()
        assert result == test_dir

    def test_format_duration(self):
        """Test duration formatting"""
        assert format_duration(1.5) == "1.5s"
        assert format_duration(65.0) == "1m 5.0s"
        assert format_duration(3665.0) == "1h 1m 5.0s"

    def test_calculate_stats(self):
        """Test statistics calculation"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = calculate_stats(values)

        assert stats['mean'] == 3.0
        assert stats['median'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['count'] == 5
        assert stats['std'] > 0

    def test_calculate_stats_empty(self):
        """Test stats with empty list"""
        with pytest.raises(ValueError, match="empty"):
            calculate_stats([])

    def test_chunk_list(self):
        """Test list chunking"""
        items = [1, 2, 3, 4, 5]
        chunks = chunk_list(items, 2)

        assert len(chunks) == 3
        assert chunks[0] == [1, 2]
        assert chunks[1] == [3, 4]
        assert chunks[2] == [5]

    def test_chunk_list_invalid(self):
        """Test chunking with invalid size"""
        with pytest.raises(ValueError, match="positive"):
            chunk_list([1, 2, 3], 0)

    def test_truncate_text(self):
        """Test text truncation"""
        text = "This is a long text"
        assert truncate_text(text, 10) == "This is..."
        assert truncate_text(text, 100) == text

    def test_flatten_dict(self):
        """Test dictionary flattening"""
        nested = {
            'a': {'b': 1, 'c': 2},
            'd': 3
        }
        flat = flatten_dict(nested)

        assert flat == {'a.b': 1, 'a.c': 2, 'd': 3}


class TestLogging:
    """Test logging utilities"""

    def test_setup_logger(self):
        """Test logger setup"""
        logger = setup_logger("test_logger")
        assert logger is not None
        assert logger.logger.name == "test_logger"

    def test_logger_with_file(self, tmp_path):
        """Test logger with file output"""
        log_file = tmp_path / "test.log"
        logger = setup_logger("test", log_file=str(log_file))

        logger.info("Test message")
        assert log_file.exists()

    def test_logger_methods(self):
        """Test logger methods"""
        logger = setup_logger("test")

        # Should not raise
        logger.info("Info message")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.metric("test_metric", 1.23, "ms")
        logger.progress(5, 10, "Test progress")
