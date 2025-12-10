"""
Tests for statistical analysis module
"""

import pytest
import numpy as np
from src.analysis.statistics import StatisticalAnalyzer


class TestStatisticalAnalyzer:
    """Test statistical analysis functions"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return StatisticalAnalyzer(confidence_level=0.95, significance_alpha=0.05)

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.confidence_level == 0.95
        assert analyzer.significance_alpha == 0.05

    def test_descriptive_stats(self, analyzer, sample_data):
        """Test descriptive statistics"""
        stats = analyzer.descriptive_stats(sample_data)

        assert stats['mean'] == 3.0
        assert stats['median'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['count'] == 5
        assert stats['std'] > 0

    def test_descriptive_stats_empty(self, analyzer):
        """Test descriptive stats with empty data"""
        with pytest.raises(ValueError, match="empty"):
            analyzer.descriptive_stats([])

    def test_confidence_interval(self, analyzer, sample_data):
        """Test confidence interval calculation"""
        mean, ci_lower, ci_upper = analyzer.confidence_interval(sample_data)

        assert mean == 3.0
        assert ci_lower < mean
        assert ci_upper > mean
        assert ci_upper - ci_lower > 0

    def test_confidence_interval_insufficient_data(self, analyzer):
        """Test CI with insufficient data"""
        with pytest.raises(ValueError, match="at least 2"):
            analyzer.confidence_interval([1.0])

    def test_cohens_d(self, analyzer):
        """Test Cohen's d calculation"""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [3.0, 4.0, 5.0, 6.0, 7.0]

        d = analyzer.cohens_d(group1, group2)

        assert d < 0  # group1 mean < group2 mean
        assert abs(d) > 0

    def test_cohens_d_interpretation(self, analyzer):
        """Test Cohen's d interpretation"""
        assert analyzer.interpret_cohens_d(0.1) == "negligible"
        assert analyzer.interpret_cohens_d(0.3) == "small"
        assert analyzer.interpret_cohens_d(0.6) == "medium"
        assert analyzer.interpret_cohens_d(1.0) == "large"

    def test_one_way_anova(self, analyzer):
        """Test one-way ANOVA"""
        groups = {
            'group1': [1.0, 2.0, 3.0],
            'group2': [4.0, 5.0, 6.0],
            'group3': [7.0, 8.0, 9.0]
        }

        result = analyzer.one_way_anova(groups)

        assert 'f_statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert isinstance(result['significant'], bool)

    def test_anova_insufficient_groups(self, analyzer):
        """Test ANOVA with insufficient groups"""
        groups = {'group1': [1.0, 2.0, 3.0]}

        with pytest.raises(ValueError, match="at least 2"):
            analyzer.one_way_anova(groups)

    def test_t_test(self, analyzer):
        """Test independent samples t-test"""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [3.0, 4.0, 5.0, 6.0, 7.0]

        result = analyzer.t_test_independent(group1, group2)

        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert isinstance(result['significant'], bool)

    def test_correlation_pearson(self, analyzer):
        """Test Pearson correlation"""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect positive correlation

        result = analyzer.correlation(x, y, method='pearson')

        assert 'correlation' in result
        assert 'p_value' in result
        assert result['correlation'] > 0.99  # Nearly perfect

    def test_correlation_spearman(self, analyzer):
        """Test Spearman correlation"""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 4.0, 9.0, 16.0, 25.0]  # Monotonic but not linear

        result = analyzer.correlation(x, y, method='spearman')

        assert 'correlation' in result
        assert result['correlation'] > 0.9

    def test_correlation_length_mismatch(self, analyzer):
        """Test correlation with mismatched lengths"""
        with pytest.raises(ValueError, match="same length"):
            analyzer.correlation([1, 2, 3], [1, 2], method='pearson')

    def test_summary_statistics(self, analyzer, sample_data):
        """Test comprehensive summary statistics"""
        summary = analyzer.summary_statistics(sample_data, label="Test")

        assert summary['label'] == "Test"
        assert 'mean' in summary
        assert 'median' in summary
        assert 'std' in summary
        assert 'ci_lower' in summary
        assert 'ci_upper' in summary
