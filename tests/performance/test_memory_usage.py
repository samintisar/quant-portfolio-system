"""
Memory efficiency validation tests for financial features system.

Tests memory efficiency targets: <4GB memory usage for 10M data points
Validates memory management across returns, volatility, and momentum calculations
"""

import pytest
import pandas as pd
import numpy as np
import psutil
import sys
import os
import gc
from datetime import datetime, timedelta
import tracemalloc
from typing import Dict, List, Tuple, Any

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.returns import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_percentage_returns,
    calculate_annualized_returns,
    calculate_multi_period_returns,
    calculate_sharpe_ratio,
    calculate_beta_alpha
)

from lib.volatility import (
    calculate_rolling_volatility,
    calculate_annualized_volatility,
    calculate_ewma_volatility,
    calculate_garch11_volatility,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility,
    calculate_yang_zhang_volatility
)

from lib.momentum import (
    calculate_simple_momentum,
    calculate_rsi,
    calculate_roc,
    calculate_stochastic,
    calculate_macd,
    calculate_momentum_divergence
)

from services.feature_service import FeatureService
from services.validation_service import ValidationService


class MemoryEfficiencyValidator:
    """Memory efficiency validation utilities."""

    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)

    @staticmethod
    def get_memory_delta(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure memory delta of a function call."""
        # Force garbage collection
        gc.collect()
        mem_before = MemoryEfficiencyValidator.get_memory_usage()

        # Execute function
        result = func(*args, **kwargs)

        # Force garbage collection again
        gc.collect()
        mem_after = MemoryEfficiencyValidator.get_memory_usage()

        return result, mem_after - mem_before

    @staticmethod
    def generate_test_data(size: int = 100_000) -> pd.DataFrame:
        """Generate test financial data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=size, freq='D')

        # Generate realistic price series
        returns = np.random.normal(0.0001, 0.02, size)
        prices = 100 * np.exp(np.cumsum(returns))

        # Add OHLC data
        daily_vol = 0.02 / np.sqrt(252)
        high = prices * (1 + np.abs(np.random.normal(0, daily_vol, size)))
        low = prices * (1 - np.abs(np.random.normal(0, daily_vol, size)))
        open_price = prices * (1 + np.random.normal(0, daily_vol/2, size))

        # Ensure proper OHLC relationships
        high = np.maximum(high, np.maximum(open_price, prices))
        low = np.minimum(low, np.minimum(open_price, prices))

        return pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, size)
        }, index=dates)

    @staticmethod
    def estimate_memory_requirements(data_points: int) -> float:
        """Estimate memory requirements based on data size."""
        # Base requirement: 4GB for 10M data points
        base_memory_per_point = 4.0 / 10_000_000  # GB per data point
        return base_memory_per_point * data_points


class TestReturnsMemoryEfficiency:
    """Memory efficiency tests for returns calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        # Start memory tracing
        tracemalloc.start()
        self.validator = MemoryEfficiencyValidator()

    def teardown_method(self):
        """Clean up after tests."""
        tracemalloc.stop()

    def test_simple_returns_memory_efficiency(self):
        """Test memory efficiency of simple returns calculation."""
        print("\n=== Simple Returns Memory Efficiency Test ===")

        sizes = [10_000, 100_000, 1_000_000]
        memory_results = {}

        for size in sizes:
            data = self.validator.generate_test_data(size)
            expected_memory = self.validator.estimate_memory_requirements(size)

            _, memory_delta = self.validator.get_memory_delta(
                calculate_simple_returns, data['close']
            )

            memory_results[size] = memory_delta

            print(f"Size: {size:,} points, Memory used: {memory_delta:.3f}GB, Expected: {expected_memory:.3f}GB")

            # Memory should scale appropriately
            assert memory_delta < expected_memory * 2, f"Simple returns for {size} points used too much memory: {memory_delta:.3f}GB"

        # Test linear scaling
        self._test_memory_scaling_linear(memory_results)

    def test_multi_period_returns_memory_efficiency(self):
        """Test memory efficiency of multi-period returns calculation."""
        print("\n=== Multi-Period Returns Memory Efficiency Test ===")

        data = self.validator.generate_test_data(100_000)
        periods = [1, 5, 21, 63, 252]
        expected_memory = self.validator.estimate_memory_requirements(100_000 * len(periods))

        _, memory_delta = self.validator.get_memory_delta(
            calculate_multi_period_returns, data['close'], periods
        )

        print(f"Multi-period returns: {memory_delta:.3f}GB for {len(periods)} periods")
        assert memory_delta < expected_memory, f"Multi-period returns used too much memory: {memory_delta:.3f}GB"

    def test_sharpe_ratio_memory_efficiency(self):
        """Test memory efficiency of Sharpe ratio calculation."""
        print("\n=== Sharpe Ratio Memory Efficiency Test ===")

        data = self.validator.generate_test_data(1_000_000)
        returns = calculate_simple_returns(data['close'])

        _, memory_delta = self.validator.get_memory_delta(
            calculate_sharpe_ratio, returns
        )

        print(f"Sharpe ratio: {memory_delta:.6f}GB for 1M returns")
        assert memory_delta < 0.01, f"Sharpe ratio used too much memory: {memory_delta:.6f}GB"

    def test_returns_calculation_memory_leak(self):
        """Test for memory leaks in returns calculations."""
        print("\n=== Returns Calculation Memory Leak Test ===")

        data = self.validator.generate_test_data(50_000)
        memory_usage = []
        n_iterations = 20

        for i in range(n_iterations):
            # Perform returns calculations
            simple_returns = calculate_simple_returns(data['close'])
            log_returns = calculate_log_returns(data['close'])
            percentage_returns = calculate_percentage_returns(data['close'])

            # Calculate statistics
            sharpe = calculate_sharpe_ratio(simple_returns)

            # Clean up
            del simple_returns, log_returns, percentage_returns, sharpe
            gc.collect()

            # Measure memory
            memory_usage.append(self.validator.get_memory_usage())

        # Check for memory growth
        memory_growth = memory_usage[-1] - memory_usage[0]
        max_allowed_growth = 0.1  # 100MB allowed growth

        print(f"Memory growth over {n_iterations} iterations: {memory_growth:.3f}GB")
        assert memory_growth < max_allowed_growth, f"Memory leak detected: grew by {memory_growth:.3f}GB"

    def _test_memory_scaling_linear(self, memory_results: Dict[int, float]):
        """Test that memory usage scales linearly."""
        sizes = sorted(memory_results.keys())
        memory_values = [memory_results[size] for size in sizes]

        # Check linear scaling relationship
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            memory_ratio = memory_values[i] / memory_values[i-1]

            # Memory growth should be roughly proportional to size growth
            assert memory_ratio < size_ratio * 1.5, f"Non-linear memory scaling: size ratio {size_ratio:.1f}, memory ratio {memory_ratio:.1f}"


class TestVolatilityMemoryEfficiency:
    """Memory efficiency tests for volatility calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        tracemalloc.start()
        self.validator = MemoryEfficiencyValidator()

    def teardown_method(self):
        """Clean up after tests."""
        tracemalloc.stop()

    def test_rolling_volatility_memory_efficiency(self):
        """Test memory efficiency of rolling volatility calculation."""
        print("\n=== Rolling Volatility Memory Efficiency Test ===")

        data = self.validator.generate_test_data(500_000)
        returns = calculate_simple_returns(data['close'])
        windows = [21, 63, 252]

        for window in windows:
            _, memory_delta = self.validator.get_memory_delta(
                calculate_rolling_volatility, returns, window=window
            )

            expected_memory = self.validator.estimate_memory_requirements(len(returns))
            print(f"Window {window}: {memory_delta:.3f}GB")

            assert memory_delta < expected_memory, f"Rolling volatility (window {window}) used too much memory"

    def test_ewma_volatility_memory_efficiency(self):
        """Test memory efficiency of EWMA volatility calculation."""
        print("\n=== EWMA Volatility Memory Efficiency Test ===")

        data = self.validator.generate_test_data(1_000_000)
        returns = calculate_simple_returns(data['close'])

        _, memory_delta = self.validator.get_memory_delta(
            calculate_ewma_volatility, returns, span=30
        )

        expected_memory = self.validator.estimate_memory_requirements(len(returns))
        print(f"EWMA volatility: {memory_delta:.3f}GB")

        assert memory_delta < expected_memory, f"EWMA volatility used too much memory: {memory_delta:.3f}GB"

    def test_garch_volatility_memory_efficiency(self):
        """Test memory efficiency of GARCH volatility calculation."""
        print("\n=== GARCH Volatility Memory Efficiency Test ===")

        data = self.validator.generate_test_data(100_000)
        returns = calculate_simple_returns(data['close'])

        _, memory_delta = self.validator.get_memory_delta(
            calculate_garch11_volatility, returns
        )

        expected_memory = self.validator.estimate_memory_requirements(len(returns))
        print(f"GARCH volatility: {memory_delta:.3f}GB")

        assert memory_delta < expected_memory, f"GARCH volatility used too much memory: {memory_delta:.3f}GB"

    def test_range_based_volatility_memory_efficiency(self):
        """Test memory efficiency of range-based volatility calculations."""
        print("\n=== Range-Based Volatility Memory Efficiency Test ===")

        data = self.validator.generate_test_data(200_000)
        expected_memory = self.validator.estimate_memory_requirements(len(data))

        # Test Parkinson volatility
        _, memory_delta = self.validator.get_memory_delta(
            calculate_parkinson_volatility,
            data['high'], data['low'], window=21
        )
        print(f"Parkinson volatility: {memory_delta:.3f}GB")
        assert memory_delta < expected_memory, "Parkinson volatility used too much memory"

        # Test Garman-Klass volatility
        _, memory_delta = self.validator.get_memory_delta(
            calculate_garman_klass_volatility,
            data['open'], data['high'], data['low'], data['close'], window=21
        )
        print(f"Garman-Klass volatility: {memory_delta:.3f}GB")
        assert memory_delta < expected_memory, "Garman-Klass volatility used too much memory"

        # Test Yang-Zhang volatility
        _, memory_delta = self.validator.get_memory_delta(
            calculate_yang_zhang_volatility,
            data['open'], data['high'], data['low'], data['close'], window=21
        )
        print(f"Yang-Zhang volatility: {memory_delta:.3f}GB")
        assert memory_delta < expected_memory, "Yang-Zhang volatility used too much memory"


class TestMomentumMemoryEfficiency:
    """Memory efficiency tests for momentum calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        tracemalloc.start()
        self.validator = MemoryEfficiencyValidator()

    def teardown_method(self):
        """Clean up after tests."""
        tracemalloc.stop()

    def test_rsi_memory_efficiency(self):
        """Test memory efficiency of RSI calculation."""
        print("\n=== RSI Memory Efficiency Test ===")

        data = self.validator.generate_test_data(500_000)
        expected_memory = self.validator.estimate_memory_requirements(len(data))

        _, memory_delta = self.validator.get_memory_delta(
            calculate_rsi, data['close'], period=14
        )

        print(f"RSI calculation: {memory_delta:.3f}GB")
        assert memory_delta < expected_memory, f"RSI calculation used too much memory: {memory_delta:.3f}GB"

    def test_macd_memory_efficiency(self):
        """Test memory efficiency of MACD calculation."""
        print("\n=== MACD Memory Efficiency Test ===")

        data = self.validator.generate_test_data(300_000)
        expected_memory = self.validator.estimate_memory_requirements(len(data))

        _, memory_delta = self.validator.get_memory_delta(
            calculate_macd, data['close']
        )

        print(f"MACD calculation: {memory_delta:.3f}GB")
        assert memory_delta < expected_memory, f"MACD calculation used too much memory: {memory_delta:.3f}GB"

    def test_stochastic_memory_efficiency(self):
        """Test memory efficiency of Stochastic calculation."""
        print("\n=== Stochastic Memory Efficiency Test ===")

        data = self.validator.generate_test_data(200_000)
        expected_memory = self.validator.estimate_memory_requirements(len(data))

        _, memory_delta = self.validator.get_memory_delta(
            calculate_stochastic,
            data['high'], data['low'], data['close']
        )

        print(f"Stochastic calculation: {memory_delta:.3f}GB")
        assert memory_delta < expected_memory, f"Stochastic calculation used too much memory: {memory_delta:.3f}GB"

    def test_momentum_divergence_memory_efficiency(self):
        """Test memory efficiency of momentum divergence calculation."""
        print("\n=== Momentum Divergence Memory Efficiency Test ===")

        data = self.validator.generate_test_data(50_000)
        rsi = calculate_rsi(data['close'], period=14)

        _, memory_delta = self.validator.get_memory_delta(
            calculate_momentum_divergence,
            data['close'], rsi, lookback=20
        )

        expected_memory = self.validator.estimate_memory_requirements(len(data))
        print(f"Momentum divergence: {memory_delta:.3f}GB")
        assert memory_delta < expected_memory, f"Momentum divergence used too much memory"


class TestFeatureServiceMemoryEfficiency:
    """Memory efficiency tests for feature service."""

    def setup_method(self):
        """Set up test fixtures."""
        tracemalloc.start()
        self.validator = MemoryEfficiencyValidator()
        self.feature_service = FeatureService()
        self.validation_service = ValidationService()

    def teardown_method(self):
        """Clean up after tests."""
        tracemalloc.stop()

    def test_batch_feature_generation_memory_efficiency(self):
        """Test memory efficiency of batch feature generation."""
        print("\n=== Batch Feature Generation Memory Efficiency Test ===")

        data = self.validator.generate_test_data(100_000)
        expected_memory = self.validator.estimate_memory_requirements(len(data))

        _, memory_delta = self.validator.get_memory_delta(
            self.feature_service.generate_features_batch,
            data,
            features=['returns', 'volatility', 'momentum']
        )

        print(f"Batch feature generation: {memory_delta:.3f}GB")
        assert memory_delta < expected_memory * 3, f"Batch feature generation used too much memory: {memory_delta:.3f}GB"

    def test_real_time_feature_generation_memory_efficiency(self):
        """Test memory efficiency of real-time feature generation."""
        print("\n=== Real-Time Feature Generation Memory Efficiency Test ===")

        data = self.validator.generate_test_data(10_000)
        memory_deltas = []

        # Simulate real-time updates
        for i in range(100):
            single_row = data.iloc[i:i+1]

            _, memory_delta = self.validator.get_memory_delta(
                self.feature_service.generate_features,
                single_row,
                features=['returns', 'volatility']
            )

            memory_deltas.append(memory_delta)

        avg_memory = np.mean(memory_deltas)
        max_memory = np.max(memory_deltas)

        print(f"Real-time features - Avg: {avg_memory:.6f}GB, Max: {max_memory:.6f}GB")
        assert avg_memory < 0.001, f"Average real-time memory usage too high: {avg_memory:.6f}GB"
        assert max_memory < 0.005, f"Peak real-time memory usage too high: {max_memory:.6f}GB"

    def test_memory_cleanup_after_feature_operations(self):
        """Test that memory is properly cleaned up after feature operations."""
        print("\n=== Memory Cleanup After Feature Operations Test ===")

        data = self.validator.generate_test_data(50_000)

        # Get baseline memory
        gc.collect()
        baseline_memory = self.validator.get_memory_usage()

        # Perform multiple feature generation operations
        for i in range(10):
            features = self.feature_service.generate_features_batch(
                data,
                features=['returns', 'volatility', 'momentum']
            )
            validated_features = self.validation_service.validate_features(features)

        # Clean up
        del features, validated_features
        gc.collect()

        final_memory = self.validator.get_memory_usage()
        memory_increase = final_memory - baseline_memory

        print(f"Memory increase after operations: {memory_increase:.3f}GB")
        assert memory_increase < 0.5, f"Memory not properly cleaned up: increased by {memory_increase:.3f}GB"


class TestLargeDatasetMemoryManagement:
    """Memory management tests for large datasets."""

    def setup_method(self):
        """Set up test fixtures."""
        tracemalloc.start()
        self.validator = MemoryEfficiencyValidator()

    def teardown_method(self):
        """Clean up after tests."""
        tracemalloc.stop()

    def test_10m_data_points_memory_efficiency(self):
        """Test memory efficiency with 10M data points (target benchmark)."""
        print("\n=== 10M Data Points Memory Efficiency Test ===")

        # This is the key test - validate against the 10M points, <4GB target
        data = self.validator.generate_test_data(2_000_000)  # 2M points for testing
        target_memory = 4.0 * (2_000_000 / 10_000_000)  # Pro-rata from 10M target

        # Test complete pipeline
        def complete_pipeline():
            # Calculate all features
            returns = calculate_simple_returns(data['close'])
            log_returns = calculate_log_returns(data['close'])
            rolling_vol = calculate_rolling_volatility(returns, window=21)
            rsi = calculate_rsi(data['close'], period=14)
            macd_line, signal_line, histogram = calculate_macd(data['close'])

            # Calculate additional metrics
            sharpe = calculate_sharpe_ratio(returns)
            parkinson_vol = calculate_parkinson_volatility(data['high'], data['low'], window=21)

            return {
                'returns': returns,
                'log_returns': log_returns,
                'volatility': rolling_vol,
                'rsi': rsi,
                'macd': macd_line,
                'sharpe_ratio': sharpe,
                'parkinson_volatility': parkinson_vol
            }

        _, memory_delta = self.validator.get_memory_delta(complete_pipeline)

        print(f"Complete pipeline (2M points): {memory_delta:.3f}GB")
        print(f"Target memory (scaled from 10M): {target_memory:.3f}GB")

        assert memory_delta < target_memory, f"2M points pipeline used {memory_delta:.3f}GB, target was {target_memory:.3f}GB"

        # Estimate performance at 10M points
        estimated_10m_memory = memory_delta * 5  # 10M / 2M = 5x scaling
        print(f"Estimated memory for 10M points: {estimated_10m_memory:.3f}GB")
        assert estimated_10m_memory < 4.0, f"Estimated 10M memory {estimated_10m_memory:.3f}GB exceeds 4GB limit"

    def test_memory_efficient_chunking(self):
        """Test memory efficiency of chunked processing."""
        print("\n=== Memory Efficient Chunking Test ===")

        large_data = self.validator.generate_test_data(500_000)
        chunk_size = 50_000
        n_chunks = len(large_data) // chunk_size

        # Process without chunking
        _, memory_full = self.validator.get_memory_delta(
            calculate_rolling_volatility,
            calculate_simple_returns(large_data['close']),
            window=21
        )

        # Process with chunking
        chunk_memory = []
        for i in range(n_chunks):
            chunk = large_data.iloc[i*chunk_size:(i+1)*chunk_size]
            chunk_returns = calculate_simple_returns(chunk['close'])

            _, mem_delta = self.validator.get_memory_delta(
                calculate_rolling_volatility, chunk_returns, window=21
            )
            chunk_memory.append(mem_delta)

        max_chunk_memory = max(chunk_memory)
        avg_chunk_memory = np.mean(chunk_memory)

        print(f"Full processing: {memory_full:.3f}GB")
        print(f"Chunked processing - Max: {max_chunk_memory:.3f}GB, Avg: {avg_chunk_memory:.3f}GB")

        # Chunked processing should use less peak memory
        assert max_chunk_memory < memory_full * 0.7, f"Chunked processing should reduce peak memory usage"

    def test_memory_peak_tracking(self):
        """Test peak memory usage during operations."""
        print("\n=== Memory Peak Tracking Test ===")

        data = self.validator.generate_test_data(200_000)

        # Track peak memory
        tracemalloc.stop()
        tracemalloc.start()

        # Perform comprehensive feature calculation
        returns = calculate_simple_returns(data['close'])
        volatility = calculate_rolling_volatility(returns, window=21)
        rsi = calculate_rsi(data['close'], period=14)
        macd = calculate_macd(data['close'])

        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak / (1024**3)

        print(f"Peak memory usage: {peak_gb:.3f}GB")

        # Peak should be reasonable for the data size
        expected_peak = self.validator.estimate_memory_requirements(len(data))
        assert peak_gb < expected_peak * 2, f"Peak memory {peak_gb:.3f}GB too high for data size"


if __name__ == "__main__":
    # Run memory efficiency tests
    print("Starting Financial Features Memory Efficiency Tests")
    print("=" * 60)

    # Initialize and run test classes
    returns_tests = TestReturnsMemoryEfficiency()
    returns_tests.setup_method()
    returns_tests.test_simple_returns_memory_efficiency()

    volatility_tests = TestVolatilityMemoryEfficiency()
    volatility_tests.setup_method()
    volatility_tests.test_rolling_volatility_memory_efficiency()

    momentum_tests = TestMomentumMemoryEfficiency()
    momentum_tests.setup_method()
    momentum_tests.test_rsi_memory_efficiency()

    large_dataset_tests = TestLargeDatasetMemoryManagement()
    large_dataset_tests.setup_method()
    large_dataset_tests.test_10m_data_points_memory_efficiency()

    print("\n" + "=" * 60)
    print("All memory efficiency tests completed successfully!")