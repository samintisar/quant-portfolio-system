"""
Performance benchmarking tests for financial features system.

Tests system performance against targets:
- Processing: 10M data points in <30 seconds
- Memory: <4GB usage for large datasets
- Throughput: Sub-second processing for 1K data batches
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import sys
import os
from typing import Dict, List, Tuple
import warnings
from unittest.mock import patch

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.returns import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_percentage_returns,
    calculate_annualized_returns,
    calculate_sharpe_ratio,
    calculate_multi_period_returns
)

from lib.volatility import (
    calculate_rolling_volatility,
    calculate_annualized_volatility,
    calculate_ewma_volatility,
    calculate_garch11_volatility,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility
)

from lib.momentum import (
    calculate_simple_momentum,
    calculate_rsi,
    calculate_roc,
    calculate_stochastic,
    calculate_macd
)

from services.feature_service import FeatureService
from services.validation_service import ValidationService


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @staticmethod
    def time_execution(func, *args, **kwargs) -> Tuple[float, any]:
        """Time function execution and return result."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result

    @staticmethod
    def generate_large_dataset(size: int = 10_000_000,
                              volatility: float = 0.02) -> pd.DataFrame:
        """Generate large synthetic price dataset for testing."""
        np.random.seed(42)  # For reproducible results

        # Generate dates to avoid pandas limitations
        # Use minute frequency for large datasets to stay within pandas limits
        if size >= 100000:
            # Use minute frequency instead of daily for large datasets
            dates = pd.date_range(start='2010-01-01', periods=size, freq='min')
        else:
            dates = pd.date_range(start='2010-01-01', periods=size, freq='D')

        # Generate realistic price series with drift and volatility
        # Use smaller drift to avoid overflow with large datasets
        drift = 0.0001 / (size / 1000)  # Scale drift with dataset size
        returns = np.random.normal(drift, volatility, size)
        prices = 100 * np.exp(np.minimum(np.cumsum(returns), 100))  # Cap the exponent to avoid overflow

        # Add OHLC data
        daily_vol = volatility / np.sqrt(252)
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


class TestReturnsPerformance:
    """Performance tests for returns calculation library."""

    def setup_method(self):
        """Set up test data."""
        self.benchmark = PerformanceBenchmark()

        # Test datasets of different sizes
        self.small_data = self.benchmark.generate_large_dataset(1_000)
        self.medium_data = self.benchmark.generate_large_dataset(100_000)
        self.large_data = self.benchmark.generate_large_dataset(1_000_000)
        self.huge_data = self.benchmark.generate_large_dataset(10_000_000)

        # Performance targets
        self.time_target_1k = 1.0  # 1 second for 1K points
        self.time_target_1m = 10.0  # 10 seconds for 1M points
        self.time_target_10m = 30.0  # 30 seconds for 10M points
        self.memory_target = 4096  # 4GB in MB

    def test_simple_returns_performance(self):
        """Test simple returns calculation performance."""
        print("\n=== Simple Returns Performance Test ===")

        for name, data in [
            ("1K", self.small_data),
            ("100K", self.medium_data),
            ("1M", self.large_data),
            ("10M", self.huge_data)
        ]:
            initial_memory = self.benchmark.get_memory_usage()

            execution_time, result = self.benchmark.time_execution(
                calculate_simple_returns, data['close']
            )

            final_memory = self.benchmark.get_memory_usage()
            memory_used = final_memory - initial_memory

            print(f"{name} points:")
            print(f"  Time: {execution_time:.3f}s")
            print(f"  Memory: {memory_used:.1f}MB")
            if execution_time > 0:
                print(f"  Points/sec: {len(data)/execution_time:,.0f}")
            else:
                print(f"  Points/sec: {len(data)/0.001:,.0f}")  # Assume 1ms if time is too small

            # Assert performance targets
            if name == "1K":
                assert execution_time < self.time_target_1k, f"1K processing too slow: {execution_time:.3f}s"
            elif name == "1M":
                assert execution_time < self.time_target_1m, f"1M processing too slow: {execution_time:.3f}s"
            elif name == "10M":
                assert execution_time < self.time_target_10m, f"10M processing too slow: {execution_time:.3f}s"
                assert memory_used < self.memory_target, f"Memory usage too high: {memory_used:.1f}MB"

            # Verify result correctness
            assert len(result) == len(data), "Result length mismatch"
            assert result.isna().sum() <= 1, "Too many NaN values in result"

    def test_log_returns_performance(self):
        """Test log returns calculation performance."""
        print("\n=== Log Returns Performance Test ===")

        execution_time, result = self.benchmark.time_execution(
            calculate_log_returns, self.large_data['close']
        )

        print(f"1M points - Time: {execution_time:.3f}s")
        assert execution_time < self.time_target_1m, "Log returns processing too slow"
        assert len(result) == len(self.large_data), "Result length mismatch"

    def test_multi_period_returns_performance(self):
        """Test multi-period returns calculation performance."""
        print("\n=== Multi-Period Returns Performance Test ===")

        periods = [1, 5, 21, 63, 252]

        execution_time, results = self.benchmark.time_execution(
            calculate_multi_period_returns, self.medium_data['close'], periods
        )

        print(f"100K points, {len(periods)} periods - Time: {execution_time:.3f}s")
        assert execution_time < 5.0, "Multi-period returns processing too slow"
        assert len(results) == len(periods), "Number of results mismatch"

    def test_sharpe_ratio_performance(self):
        """Test Sharpe ratio calculation performance."""
        print("\n=== Sharpe Ratio Performance Test ===")

        returns = calculate_simple_returns(self.large_data['close'])

        execution_time, sharpe_ratio = self.benchmark.time_execution(
            calculate_sharpe_ratio, returns
        )

        print(f"1M returns - Time: {execution_time:.6f}s")
        assert execution_time < 0.1, "Sharpe ratio calculation too slow"
        assert isinstance(sharpe_ratio, float), "Sharpe ratio should be a float"


class TestVolatilityPerformance:
    """Performance tests for volatility calculation library."""

    def setup_method(self):
        """Set up test data."""
        self.benchmark = PerformanceBenchmark()
        self.returns_data = self.benchmark.generate_large_dataset(1_000_000)
        self.returns = calculate_simple_returns(self.returns_data['close'])

    def test_rolling_volatility_performance(self):
        """Test rolling volatility calculation performance."""
        print("\n=== Rolling Volatility Performance Test ===")

        windows = [21, 63, 252]

        for window in windows:
            execution_time, result = self.benchmark.time_execution(
                calculate_rolling_volatility, self.returns, window=window
            )

            print(f"Window {window}: {execution_time:.3f}s")
            assert execution_time < 5.0, f"Rolling volatility with window {window} too slow"

    def test_ewma_volatility_performance(self):
        """Test EWMA volatility calculation performance."""
        print("\n=== EWMA Volatility Performance Test ===")

        execution_time, result = self.benchmark.time_execution(
            calculate_ewma_volatility, self.returns, span=30
        )

        print(f"EWMA volatility: {execution_time:.3f}s")
        assert execution_time < 3.0, "EWMA volatility calculation too slow"

    def test_parkinson_volatility_performance(self):
        """Test Parkinson volatility calculation performance."""
        print("\n=== Parkinson Volatility Performance Test ===")

        execution_time, result = self.benchmark.time_execution(
            calculate_parkinson_volatility,
            self.returns_data['high'],
            self.returns_data['low'],
            window=21
        )

        print(f"Parkinson volatility: {execution_time:.3f}s")
        assert execution_time < 8.0, "Parkinson volatility calculation too slow"


class TestMomentumPerformance:
    """Performance tests for momentum calculation library."""

    def setup_method(self):
        """Set up test data."""
        self.benchmark = PerformanceBenchmark()
        self.price_data = self.benchmark.generate_large_dataset(500_000)

    def test_rsi_performance(self):
        """Test RSI calculation performance."""
        print("\n=== RSI Performance Test ===")

        execution_time, result = self.benchmark.time_execution(
            calculate_rsi, self.price_data['close'], period=14
        )

        print(f"RSI calculation: {execution_time:.3f}s")
        assert execution_time < 15.0, "RSI calculation too slow"

    def test_macd_performance(self):
        """Test MACD calculation performance."""
        print("\n=== MACD Performance Test ===")

        execution_time, result = self.benchmark.time_execution(
            calculate_macd, self.price_data['close']
        )

        print(f"MACD calculation: {execution_time:.3f}s")
        assert execution_time < 10.0, "MACD calculation too slow"

    def test_stochastic_performance(self):
        """Test Stochastic oscillator calculation performance."""
        print("\n=== Stochastic Performance Test ===")

        execution_time, result = self.benchmark.time_execution(
            calculate_stochastic,
            self.price_data['high'],
            self.price_data['low'],
            self.price_data['close']
        )

        print(f"Stochastic calculation: {execution_time:.3f}s")
        assert execution_time < 12.0, "Stochastic calculation too slow"


class TestFeatureServicePerformance:
    """Performance tests for feature service."""

    def setup_method(self):
        """Set up test data."""
        self.benchmark = PerformanceBenchmark()
        self.feature_service = FeatureService()
        self.validation_service = ValidationService()

        # Create test dataset
        self.data = self.benchmark.generate_large_dataset(100_000)

    def test_batch_feature_generation_performance(self):
        """Test batch feature generation performance."""
        print("\n=== Batch Feature Generation Performance Test ===")

        initial_memory = self.benchmark.get_memory_usage()

        execution_time, features = self.benchmark.time_execution(
            self.feature_service.generate_features_batch,
            self.data,
            features=['returns', 'volatility', 'momentum']
        )

        final_memory = self.benchmark.get_memory_usage()
        memory_used = final_memory - initial_memory

        print(f"Batch feature generation: {execution_time:.3f}s")
        print(f"Memory used: {memory_used:.1f}MB")
        print(f"Features generated: {len(features)}")

        assert execution_time < 20.0, "Batch feature generation too slow"
        assert memory_used < 1000, "Memory usage too high for batch processing"

    def test_real_time_feature_generation_performance(self):
        """Test real-time feature generation performance."""
        print("\n=== Real-time Feature Generation Performance Test ===")

        # Test with smaller batch for real-time simulation
        small_batch = self.data.tail(1000)

        times = []
        for i in range(100):  # Simulate 100 real-time updates
            execution_time, _ = self.benchmark.time_execution(
                self.feature_service.generate_features,
                small_batch.iloc[i:i+1],
                features=['returns', 'volatility']
            )
            times.append(execution_time)

        avg_time = np.mean(times)
        max_time = np.max(times)

        print(f"Real-time feature generation:")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Max time: {max_time:.6f}s")
        print(f"  Throughput: {1/avg_time:.1f} features/second")

        assert avg_time < 0.01, f"Real-time processing too slow: {avg_time:.6f}s"
        assert max_time < 0.05, f"Real-time spike too high: {max_time:.6f}s"


class TestSystemIntegrationPerformance:
    """Integration performance tests for the complete system."""

    def setup_method(self):
        """Set up test data."""
        self.benchmark = PerformanceBenchmark()
        self.feature_service = FeatureService()
        self.validation_service = ValidationService()

        # Large dataset for integration testing
        self.data = self.benchmark.generate_large_dataset(2_000_000)

    def test_end_to_end_pipeline_performance(self):
        """Test complete pipeline performance."""
        print("\n=== End-to-End Pipeline Performance Test ===")

        initial_memory = self.benchmark.get_memory_usage()

        def complete_pipeline():
            # Step 1: Data validation
            validated_data = self.validation_service.validate_dataset(self.data)

            # Step 2: Feature generation
            features = self.feature_service.generate_features_batch(
                validated_data,
                features=['returns', 'volatility', 'momentum', 'risk_metrics']
            )

            # Step 3: Feature validation
            validated_features = self.validation_service.validate_features(features)

            return validated_features

        execution_time, result = self.benchmark.time_execution(complete_pipeline)

        final_memory = self.benchmark.get_memory_usage()
        memory_used = final_memory - initial_memory

        print(f"End-to-end pipeline:")
        print(f"  Total time: {execution_time:.3f}s")
        print(f"  Memory used: {memory_used:.1f}MB")
        print(f"  Data points processed: {len(self.data):,}")
        print(f"  Processing rate: {len(self.data)/execution_time:,.0f} points/second")

        # Performance assertions
        assert execution_time < 60.0, f"End-to-end pipeline too slow: {execution_time:.3f}s"
        assert memory_used < 2048, f"Memory usage too high: {memory_used:.1f}MB"
        assert len(self.data)/execution_time > 50000, "Processing rate too low"

    def test_concurrent_processing_performance(self):
        """Test concurrent processing performance."""
        print("\n=== Concurrent Processing Performance Test ===")

        import concurrent.futures
        import threading

        def process_chunk(chunk_data, chunk_id):
            """Process a chunk of data."""
            features = self.feature_service.generate_features_batch(
                chunk_data,
                features=['returns', 'volatility']
            )
            return len(features), chunk_id

        # Split data into chunks
        chunk_size = 100_000
        chunks = [self.data.iloc[i:i+chunk_size]
                 for i in range(0, len(self.data), chunk_size)]

        initial_memory = self.benchmark.get_memory_usage()
        start_time = time.time()

        # Process chunks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, chunk, i)
                      for i, chunk in enumerate(chunks)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        execution_time = time.time() - start_time
        final_memory = self.benchmark.get_memory_usage()
        memory_used = final_memory - initial_memory

        print(f"Concurrent processing:")
        print(f"  Total time: {execution_time:.3f}s")
        print(f"  Memory used: {memory_used:.1f}MB")
        print(f"  Chunks processed: {len(results)}")
        print(f"  Processing rate: {len(self.data)/execution_time:,.0f} points/second")

        assert execution_time < 30.0, f"Concurrent processing too slow: {execution_time:.3f}s"
        assert len(results) == len(chunks), "Not all chunks were processed"


if __name__ == "__main__":
    # Run performance benchmarks
    print("Starting Financial Features Performance Benchmarks")
    print("=" * 60)

    # Run specific test classes
    test_returns = TestReturnsPerformance()
    test_returns.setup_method()
    test_returns.test_simple_returns_performance()

    test_volatility = TestVolatilityPerformance()
    test_volatility.setup_method()
    test_volatility.test_rolling_volatility_performance()

    test_momentum = TestMomentumPerformance()
    test_momentum.setup_method()
    test_momentum.test_rsi_performance()

    test_integration = TestSystemIntegrationPerformance()
    test_integration.setup_method()
    test_integration.test_end_to_end_pipeline_performance()

    print("\n" + "=" * 60)
    print("All performance benchmarks completed successfully!")