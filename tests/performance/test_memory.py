"""
Memory usage validation tests for preprocessing operations.

Tests memory efficiency targets: <4GB memory usage for 10M data points
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

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.cleaning import DataCleaner
from lib.validation import DataValidator
from lib.normalization import DataNormalizer


class TestMemoryUsage:
    """Memory usage validation tests."""

    def setup_method(self):
        """Set up test fixtures."""
        # Start memory tracing
        tracemalloc.start()

        # Create preprocessing components
        self.cleaner = DataCleaner()
        self.validator = DataValidator()
        self.normalizer = DataNormalizer()

    def teardown_method(self):
        """Clean up after tests."""
        tracemalloc.stop()

    def get_memory_usage(self):
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)

    def get_memory_delta(self, func, *args, **kwargs):
        """Measure memory delta of a function call."""
        # Force garbage collection
        gc.collect()
        mem_before = self.get_memory_usage()

        # Execute function
        result = func(*args, **kwargs)

        # Force garbage collection again
        gc.collect()
        mem_after = self.get_memory_usage()

        return result, mem_after - mem_before

    def test_cleaning_memory_efficiency(self):
        """Test memory efficiency of cleaning operations."""
        np.random.seed(42)

        # Create test dataset (1M data points for memory testing)
        n_rows = 10000
        n_cols = 100
        data_points = n_rows * n_cols

        df = pd.DataFrame(np.random.random((n_rows, n_cols)))
        df.columns = [f'col_{i}' for i in range(n_cols)]

        # Test missing value handling memory
        _, mem_delta = self.get_memory_delta(
            self.cleaner.handle_missing_values, df, method='forward_fill'
        )

        # Memory should be proportional to data size
        expected_max_memory = 4.0 * (data_points / 10_000_000)  # Scale from 10M target
        assert mem_delta < expected_max_memory, f"Missing value handling used {mem_delta:.2f}GB, expected < {expected_max_memory:.2f}GB"

        # Test outlier detection memory
        _, mem_delta = self.get_memory_delta(
            self.cleaner.detect_outliers, df, method='iqr', action='flag'
        )
        assert mem_delta < expected_max_memory, f"Outlier detection used {mem_delta:.2f}GB, expected < {expected_max_memory:.2f}GB"

    def test_validation_memory_efficiency(self):
        """Test memory efficiency of validation operations."""
        np.random.seed(42)

        # Create test dataset
        n_rows = 10000
        n_cols = 50
        data_points = n_rows * n_cols

        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=n_rows, freq='1min'),
            **{f'col_{i}': np.random.random(n_rows) for i in range(n_cols-1)}
        })

        # Test comprehensive validation memory
        _, mem_delta = self.get_memory_delta(
            self.validator.run_comprehensive_validation, df
        )

        expected_max_memory = 4.0 * (data_points / 10_000_000)
        assert mem_delta < expected_max_memory, f"Validation used {mem_delta:.2f}GB, expected < {expected_max_memory:.2f}GB"

    def test_normalization_memory_efficiency(self):
        """Test memory efficiency of normalization operations."""
        np.random.seed(42)

        # Create test dataset
        n_rows = 10000
        n_cols = 100
        data_points = n_rows * n_cols

        df = pd.DataFrame(np.random.random((n_rows, n_cols)))

        # Test z-score normalization memory
        _, mem_delta = self.get_memory_delta(
            self.normalizer.normalize_zscore, df
        )
        expected_max_memory = 4.0 * (data_points / 10_000_000)
        assert mem_delta < expected_max_memory, f"Z-score normalization used {mem_delta:.2f}GB, expected < {expected_max_memory:.2f}GB"

        # Test min-max normalization memory
        _, mem_delta = self.get_memory_delta(
            self.normalizer.normalize_minmax, df
        )
        assert mem_delta < expected_max_memory, f"Min-max normalization used {mem_delta:.2f}GB, expected < {expected_max_memory:.2f}GB"

        # Test robust normalization memory
        _, mem_delta = self.get_memory_delta(
            self.normalizer.normalize_robust, df
        )
        assert mem_delta < expected_max_memory, f"Robust normalization used {mem_delta:.2f}GB, expected < {expected_max_memory:.2f}GB"

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        np.random.seed(42)

        # Create test dataset
        df = pd.DataFrame(np.random.random((1000, 50)))

        # Track memory over multiple iterations
        memory_usage = []
        n_iterations = 10

        for i in range(n_iterations):
            # Perform preprocessing operations
            cleaned = self.cleaner.handle_missing_values(df, method='forward_fill')
            normalized, _ = self.normalizer.normalize_zscore(cleaned)
            validation_results = self.validator.run_comprehensive_validation(normalized)

            # Measure memory
            gc.collect()
            memory_usage.append(self.get_memory_usage())

        # Memory should not grow significantly over iterations
        memory_growth = memory_usage[-1] - memory_usage[0]
        max_allowed_growth = 0.1  # 100MB allowed growth

        assert memory_growth < max_allowed_growth, f"Memory leak detected: grew by {memory_growth:.2f}GB over {n_iterations} iterations"

    def test_large_dataset_memory_management(self):
        """Test memory management with large datasets."""
        np.random.seed(42)

        # Create large dataset
        n_rows = 50000
        n_cols = 200
        data_points = n_rows * n_cols

        df = pd.DataFrame(np.random.random((n_rows, n_cols)))

        # Test that operations complete without excessive memory
        expected_max_memory = 4.0 * (data_points / 10_000_000)

        # Full preprocessing pipeline
        _, mem_delta = self.get_memory_delta(
            self._full_preprocessing_pipeline, df
        )

        assert mem_delta < expected_max_memory, f"Large dataset processing used {mem_delta:.2f}GB, expected < {expected_max_memory:.2f}GB"

    def _full_preprocessing_pipeline(self, df):
        """Helper method for full preprocessing pipeline."""
        # Clean
        cleaned = self.cleaner.handle_missing_values(df, method='forward_fill')
        cleaned, _ = self.cleaner.detect_outliers(cleaned, method='iqr', action='clip')

        # Validate
        validation_results = self.validator.run_comprehensive_validation(cleaned)

        # Normalize
        normalized, _ = self.normalizer.normalize_zscore(cleaned)

        return normalized

    def test_memory_scaling_linearity(self):
        """Test that memory usage scales linearly with data size."""
        np.random.seed(42)

        sizes = [1000, 5000, 10000, 20000]
        memory_usage = []

        for size in sizes:
            df = pd.DataFrame(np.random.random((size, 50)))

            _, mem_delta = self.get_memory_delta(
                self.normalizer.normalize_zscore, df
            )
            memory_usage.append(mem_delta)

        # Check linear scaling
        # Memory usage should roughly double when data size doubles
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            memory_ratio = memory_usage[i] / memory_usage[i-1]

            # Allow some overhead but memory growth should be proportional
            assert memory_ratio < size_ratio * 2, f"Non-linear memory scaling: size ratio {size_ratio:.1f}, memory ratio {memory_ratio:.1f}"

    def test_memory_efficient_operations(self):
        """Test memory efficiency of specific operations."""
        np.random.seed(42)

        df = pd.DataFrame(np.random.random((10000, 100)))

        # Test in-place vs copy operations
        _, mem_delta_copy = self.get_memory_delta(
            self.cleaner.handle_missing_values, df, method='forward_fill'
        )

        # Should not use excessive memory even for large operations
        assert mem_delta_copy < 1.0, f"Operation used {mem_delta_copy:.2f}GB, expected < 1GB"

    def test_memory_peak_tracking(self):
        """Test peak memory usage during operations."""
        np.random.seed(42)

        # Create moderately large dataset
        df = pd.DataFrame(np.random.random((50000, 100)))

        # Track peak memory
        tracemalloc.stop()
        tracemalloc.start()

        # Perform operation
        self._full_preprocessing_pipeline(df)

        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak / (1024**3)

        # Peak should be reasonable
        assert peak_gb < 4.0, f"Peak memory usage {peak_gb:.2f}GB exceeds 4GB limit"

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after operations."""
        np.random.seed(42)

        df = pd.DataFrame(np.random.random((10000, 100)))

        # Get baseline memory
        gc.collect()
        baseline_memory = self.get_memory_usage()

        # Perform operations
        for i in range(5):
            cleaned = self.cleaner.handle_missing_values(df, method='forward_fill')
            normalized, _ = self.normalizer.normalize_zscore(cleaned)
            validation_results = self.validator.run_comprehensive_validation(normalized)

        # Force cleanup and check memory
        del cleaned, normalized, validation_results
        gc.collect()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - baseline_memory

        # Memory should return close to baseline
        assert memory_increase < 0.5, f"Memory not properly cleaned up: increased by {memory_increase:.2f}GB"

    def test_memory_efficient_data_structures(self):
        """Test memory efficiency of different data structures."""
        np.random.seed(42)

        # Test different dtypes
        n_rows = 10000

        # Float64 vs Float32
        df_float64 = pd.DataFrame(np.random.random((n_rows, 50)))
        df_float32 = pd.DataFrame(np.random.random((n_rows, 50)).astype(np.float32))

        # Compare memory usage
        _, mem_delta_64 = self.get_memory_delta(
            self.normalizer.normalize_zscore, df_float64
        )
        _, mem_delta_32 = self.get_memory_delta(
            self.normalizer.normalize_zscore, df_float32
        )

        # Float32 should use less memory
        assert mem_delta_32 < mem_delta_64, f"Float32 should use less memory than Float64"

    def test_memory_usage_in_concurrent_processing(self):
        """Test memory usage in concurrent processing scenarios."""
        import concurrent.futures
        np.random.seed(42)

        # Create test data
        df = pd.DataFrame(np.random.random((10000, 50)))

        # Split for concurrent processing
        n_chunks = 4
        chunk_size = len(df) // n_chunks
        chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

        # Get baseline memory
        gc.collect()
        baseline_memory = self.get_memory_usage()

        # Process concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_chunks) as executor:
            futures = [executor.submit(self.normalizer.normalize_zscore, chunk) for chunk in chunks]
            results = concurrent.futures.wait(futures)

        # Check memory usage
        gc.collect()
        final_memory = self.get_memory_usage()
        memory_delta = final_memory - baseline_memory

        # Concurrent processing should not use excessive memory
        assert memory_delta < 2.0, f"Concurrent processing used {memory_delta:.2f}GB, expected < 2GB"

    def test_memory_comprehensive_validation(self):
        """Comprehensive memory validation across all preprocessing operations."""
        np.random.seed(42)

        # Create comprehensive test dataset
        n_rows = 20000
        n_cols = 100
        data_points = n_rows * n_cols

        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=n_rows, freq='1min'),
            **{f'price_{i}': np.random.lognormal(4.0, 0.2, n_rows) for i in range(25)},
            **{f'volume_{i}': np.random.lognormal(15.0, 0.5, n_rows).astype(int) for i in range(25)},
            **{f'indicator_{i}': np.random.normal(0, 1, n_rows) for i in range(49)}
        })

        # Calculate expected memory limit
        expected_max_memory = 4.0 * (data_points / 10_000_000)

        # Test all operations
        operations = [
            ('Missing Value Handling', lambda df: self.cleaner.handle_missing_values(df, 'forward_fill')),
            ('Outlier Detection', lambda df: self.cleaner.detect_outliers(df, 'iqr', 'flag')[0]),
            ('Validation', lambda df: self.validator.run_comprehensive_validation(df)),
            ('Z-Score Normalization', lambda df: self.normalizer.normalize_zscore(df)[0]),
            ('Min-Max Normalization', lambda df: self.normalizer.normalize_minmax(df)[0]),
            ('Robust Normalization', lambda df: self.normalizer.normalize_robust(df)[0]),
        ]

        results = {}
        for name, operation in operations:
            _, mem_delta = self.get_memory_delta(operation, df.copy())
            results[name] = mem_delta

            # Assert memory usage for each operation
            assert mem_delta < expected_max_memory, f"{name} used {mem_delta:.2f}GB, expected < {expected_max_memory:.2f}GB"

        # Test full pipeline
        _, pipeline_memory = self.get_memory_delta(self._full_preprocessing_pipeline, df.copy())
        results['Full Pipeline'] = pipeline_memory

        # Full pipeline can use more memory but should still be within limits
        assert pipeline_memory < expected_max_memory * 2, f"Full pipeline used {pipeline_memory:.2f}GB, expected < {expected_max_memory * 2:.2f}GB"

        return {
            'data_points': data_points,
            'memory_limit_gb': expected_max_memory,
            'operation_memory_usage': results,
            'all_operations_within_limits': all(v <= expected_max_memory for v in results.values() if v != 'Full Pipeline')
        }