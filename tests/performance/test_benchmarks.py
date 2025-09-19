"""
Performance benchmarking tests for large datasets.

Tests performance targets: Process 10M data points in <30 seconds, memory usage <4GB
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import sys
import os
from datetime import datetime, timedelta

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.cleaning import DataCleaner
from lib.validation import DataValidator
from lib.normalization import DataNormalizer


class TestPerformanceBenchmarks:
    """Performance benchmarking tests for preprocessing operations."""

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for benchmarking."""
        np.random.seed(42)

        # Performance targets: 10M data points
        n_rows = 100000  # 100K rows for testing
        n_cols = 100  # 100 columns → 10M data points

        # Generate synthetic financial data
        dates = pd.date_range(start='2020-01-01', periods=n_rows, freq='1min')

        data = {}
        for i in range(n_cols):
            if i % 4 == 0:
                col_name = f'price_{i}'
                # Price data with lognormal distribution
                data[col_name] = np.random.lognormal(4.0, 0.2, n_rows)
            elif i % 4 == 1:
                col_name = f'volume_{i}'
                # Volume data with lognormal distribution
                data[col_name] = np.random.lognormal(15.0, 0.5, n_rows).astype(int)
            elif i % 4 == 2:
                col_name = f'indicator_{i}'
                # Technical indicator data
                data[col_name] = np.random.normal(0, 1, n_rows)
            else:
                col_name = f'metric_{i}'
                # General metric data
                data[col_name] = np.random.uniform(0, 100, n_rows)

        data['timestamp'] = dates

        df = pd.DataFrame(data)

        # Add some data quality issues
        # Missing values (5% random)
        missing_mask = np.random.random(df.shape) < 0.05
        for col in df.columns:
            if col != 'timestamp':
                df.loc[missing_mask[:, df.columns.get_loc(col)], col] = np.nan

        # Outliers (1% of data)
        outlier_mask = np.random.random(df.shape) < 0.01
        for col in df.columns:
            if col != 'timestamp' and df[col].dtype in ['float64', 'int64']:
                df.loc[outlier_mask[:, df.columns.get_loc(col)], col] *= 10

        return df

    @pytest.fixture
    def preprocessing_components(self):
        """Set up preprocessing components."""
        return {
            'cleaner': DataCleaner(),
            'validator': DataValidator(),
            'normalizer': DataNormalizer()
        }

    def get_memory_usage(self):
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)  # Convert to GB

    def benchmark_cleaning_operations(self, large_dataset, preprocessing_components):
        """Benchmark data cleaning operations."""
        cleaner = preprocessing_components['cleaner']

        # Memory before
        mem_before = self.get_memory_usage()

        # Time cleaning operations
        start_time = time.time()

        # Missing value handling
        cleaned = cleaner.handle_missing_values(large_dataset, method='forward_fill')

        # Outlier detection
        cleaned, outlier_masks = cleaner.detect_outliers(
            cleaned, method='iqr', action='clip'
        )

        # Duplicate removal
        cleaned = cleaner.remove_duplicate_rows(cleaned)

        end_time = time.time()
        mem_after = self.get_memory_usage()

        # Performance metrics
        execution_time = end_time - start_time
        memory_used = mem_after - mem_before
        data_points = large_dataset.shape[0] * large_dataset.shape[1]

        # Performance assertions
        # Target: <30 seconds for 10M data points (pro-rated for test size)
        expected_time = 30.0 * (data_points / 10_000_000)
        assert execution_time < expected_time, f"Cleaning took {execution_time:.2f}s, expected < {expected_time:.2f}s"

        # Memory usage <4GB
        assert memory_used < 4.0, f"Cleaning used {memory_used:.2f}GB, expected < 4GB"

        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'data_points': data_points,
            'rows_per_second': large_dataset.shape[0] / execution_time
        }

    def benchmark_validation_operations(self, large_dataset, preprocessing_components):
        """Benchmark data validation operations."""
        validator = preprocessing_components['validator']

        mem_before = self.get_memory_usage()
        start_time = time.time()

        # Comprehensive validation
        validation_results = validator.run_comprehensive_validation(
            large_dataset,
            config={
                'required_columns': ['timestamp'],
                'expected_frequency': '1min',
                'validate_ratios': True,
                'validate_cross_asset': False
            }
        )

        end_time = time.time()
        mem_after = self.get_memory_usage()

        execution_time = end_time - start_time
        memory_used = mem_after - mem_before
        data_points = large_dataset.shape[0] * large_dataset.shape[1]

        # Performance assertions
        expected_time = 30.0 * (data_points / 10_000_000)
        assert execution_time < expected_time, f"Validation took {execution_time:.2f}s, expected < {expected_time:.2f}s"
        assert memory_used < 4.0, f"Validation used {memory_used:.2f}GB, expected < 4GB"

        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'data_points': data_points,
            'validation_score': validator.get_data_quality_score(validation_results)
        }

    def benchmark_normalization_operations(self, large_dataset, preprocessing_components):
        """Benchmark data normalization operations."""
        normalizer = preprocessing_components['normalizer']

        mem_before = self.get_memory_usage()
        start_time = time.time()

        # Z-score normalization
        normalized_zscore, _ = normalizer.normalize_zscore(large_dataset)

        # Min-max normalization
        normalized_minmax, _ = normalizer.normalize_minmax(large_dataset)

        # Robust normalization
        normalized_robust, _ = normalizer.normalize_robust(large_dataset)

        end_time = time.time()
        mem_after = self.get_memory_usage()

        execution_time = end_time - start_time
        memory_used = mem_after - mem_before
        data_points = large_dataset.shape[0] * large_dataset.shape[1]

        # Performance assertions
        expected_time = 30.0 * (data_points / 10_000_000)
        assert execution_time < expected_time, f"Normalization took {execution_time:.2f}s, expected < {expected_time:.2f}s"
        assert memory_used < 4.0, f"Normalization used {memory_used:.2f}GB, expected < 4GB"

        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'data_points': data_points,
            'methods_tested': ['zscore', 'minmax', 'robust']
        }

    def benchmark_full_preprocessing_pipeline(self, large_dataset, preprocessing_components):
        """Benchmark complete preprocessing pipeline."""
        cleaner = preprocessing_components['cleaner']
        validator = preprocessing_components['validator']
        normalizer = preprocessing_components['normalizer']

        mem_before = self.get_memory_usage()
        start_time = time.time()

        # Step 1: Cleaning
        cleaned = cleaner.handle_missing_values(large_dataset, method='forward_fill')
        cleaned, _ = cleaner.detect_outliers(cleaned, method='iqr', action='clip')

        # Step 2: Validation
        validation_results = validator.run_comprehensive_validation(cleaned)

        # Step 3: Normalization
        normalized, _ = normalizer.normalize_zscore(cleaned)

        end_time = time.time()
        mem_after = self.get_memory_usage()

        execution_time = end_time - start_time
        memory_used = mem_after - mem_before
        data_points = large_dataset.shape[0] * large_dataset.shape[1]

        # Performance assertions for full pipeline
        expected_time = 60.0 * (data_points / 10_000_000)  # Allow more time for full pipeline
        assert execution_time < expected_time, f"Full pipeline took {execution_time:.2f}s, expected < {expected_time:.2f}s"
        assert memory_used < 4.0, f"Full pipeline used {memory_used:.2f}GB, expected < 4GB"

        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'data_points': data_points,
            'validation_score': validator.get_data_quality_score(validation_results)
        }

    def test_scalability_benchmarks(self):
        """Test scalability with different dataset sizes."""
        sizes = [1000, 10000, 100000]  # Different row counts
        results = {}

        for size in sizes:
            # Create dataset of specific size
            df = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=size, freq='1min'),
                'price': np.random.lognormal(4.0, 0.2, size),
                'volume': np.random.lognormal(15.0, 0.5, size).astype(int),
                'indicator': np.random.normal(0, 1, size)
            })

            cleaner = DataCleaner()

            start_time = time.time()
            cleaned = cleaner.handle_missing_values(df, method='forward_fill')
            execution_time = time.time() - start_time

            results[size] = {
                'execution_time': execution_time,
                'rows_per_second': size / execution_time
            }

        # Test linear scalability (execution time should scale roughly linearly with size)
        sizes_sorted = sorted(sizes)
        times = [results[size]['execution_time'] for size in sizes_sorted]

        # Check that larger datasets don't have super-linear time increase
        for i in range(1, len(times)):
            size_ratio = sizes_sorted[i] / sizes_sorted[i-1]
            time_ratio = times[i] / times[i-1]

            # Time increase should be less than size increase squared (indicating good scalability)
            assert time_ratio < size_ratio ** 2, f"Non-linear scaling detected: size ratio {size_ratio:.1f}, time ratio {time_ratio:.1f}"

    def test_concurrent_processing_benchmark(self, large_dataset):
        """Test performance of concurrent processing capabilities."""
        import concurrent.futures
        import threading

        cleaner = DataCleaner()

        # Split dataset for concurrent processing
        n_chunks = 4
        chunk_size = len(large_dataset) // n_chunks
        chunks = [large_dataset.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

        mem_before = self.get_memory_usage()
        start_time = time.time()

        # Process chunks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_chunks) as executor:
            futures = [executor.submit(cleaner.handle_missing_values, chunk, 'forward_fill') for chunk in chunks]
            results = concurrent.futures.wait(futures)

        end_time = time.time()
        mem_after = self.get_memory_usage()

        execution_time = end_time - start_time
        memory_used = mem_after - mem_before

        # Concurrent processing should be faster than sequential
        # Allow some overhead for threading
        expected_time = 15.0 * (large_dataset.shape[0] * large_dataset.shape[1] / 10_000_000)
        assert execution_time < expected_time, f"Concurrent processing took {execution_time:.2f}s, expected < {expected_time:.2f}s"

        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'chunks_processed': n_chunks
        }

    def test_memory_efficiency_benchmark(self):
        """Test memory efficiency with large datasets."""
        # Test memory usage grows linearly with data size
        sizes = [10000, 50000, 100000]
        memory_usage = {}

        cleaner = DataCleaner()

        for size in sizes:
            df = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=size, freq='1min'),
                'price': np.random.lognormal(4.0, 0.2, size),
                'volume': np.random.lognormal(15.0, 0.5, size).astype(int)
            })

            # Force garbage collection
            import gc
            gc.collect()

            mem_before = self.get_memory_usage()

            # Perform operation
            cleaned = cleaner.handle_missing_values(df, method='forward_fill')

            mem_after = self.get_memory_usage()
            memory_usage[size] = mem_after - mem_before

        # Check linear memory growth
        sizes_sorted = sorted(sizes)
        memory_values = [memory_usage[size] for size in sizes_sorted]

        for i in range(1, len(memory_values)):
            size_ratio = sizes_sorted[i] / sizes_sorted[i-1]
            memory_ratio = memory_values[i] / memory_values[i-1]

            # Memory usage should grow linearly with data size
            assert memory_ratio < size_ratio * 1.5, f"Non-linear memory growth: size ratio {size_ratio:.1f}, memory ratio {memory_ratio:.1f}"

    def test_real_time_processing_benchmark(self):
        """Test real-time processing performance."""
        # Simulate real-time data stream
        batch_size = 1000
        n_batches = 100
        total_points = batch_size * n_batches

        cleaner = DataCleaner()
        processing_times = []

        for batch in range(n_batches):
            # Generate batch
            hour = batch % 24  # Ensure hour is always 0-23
            day = batch // 24   # Increment day when hour wraps around
            batch_data = pd.DataFrame({
                'timestamp': pd.date_range(start=f'2023-01-{day+1:02d} {hour:02d}:00:00', periods=batch_size, freq='1s'),
                'price': np.random.lognormal(4.0, 0.1, batch_size),
                'volume': np.random.lognormal(15.0, 0.3, batch_size).astype(int)
            })

            # Process batch
            start_time = time.time()
            cleaned = cleaner.handle_missing_values(batch_data, method='forward_fill')
            processing_time = time.time() - start_time

            processing_times.append(processing_time)

            # Real-time constraint: each batch should process faster than it arrives
            assert processing_time < 1.0, f"Batch {batch} took {processing_time:.3f}s, should be < 1s for real-time"

        # Check overall performance
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)

        assert avg_processing_time < 0.5, f"Average processing time {avg_processing_time:.3f}s too slow for real-time"
        assert max_processing_time < 1.0, f"Max processing time {max_processing_time:.3f}s too slow for real-time"

        return {
            'avg_processing_time': avg_processing_time,
            'max_processing_time': max_processing_time,
            'total_points': total_points,
            'throughput': total_points / sum(processing_times)
        }

    def test_comprehensive_performance_report(self, large_dataset, preprocessing_components):
        """Generate comprehensive performance report."""
        # Run all benchmarks
        cleaning_results = self.benchmark_cleaning_operations(large_dataset, preprocessing_components)
        validation_results = self.benchmark_validation_operations(large_dataset, preprocessing_components)
        normalization_results = self.benchmark_normalization_operations(large_dataset, preprocessing_components)
        pipeline_results = self.benchmark_full_preprocessing_pipeline(large_dataset, preprocessing_components)

        # Generate performance report
        report = {
            'dataset_info': {
                'rows': large_dataset.shape[0],
                'columns': large_dataset.shape[1],
                'total_data_points': large_dataset.shape[0] * large_dataset.shape[1],
                'memory_size_mb': large_dataset.memory_usage(deep=True).sum() / 1024**2
            },
            'performance_targets': {
                'max_execution_time_seconds': 30,
                'max_memory_usage_gb': 4.0,
                'min_rows_per_second': 333333  # 10M points / 30 seconds / 100 columns
            },
            'benchmark_results': {
                'cleaning': cleaning_results,
                'validation': validation_results,
                'normalization': normalization_results,
                'full_pipeline': pipeline_results
            },
            'performance_summary': {
                'all_targets_met': (
                    cleaning_results['execution_time'] < 30 and
                    validation_results['execution_time'] < 30 and
                    normalization_results['execution_time'] < 30 and
                    pipeline_results['execution_time'] < 60 and
                    max(cleaning_results['memory_used'],
                        validation_results['memory_used'],
                        normalization_results['memory_used'],
                        pipeline_results['memory_used']) < 4.0
                ),
                'fastest_operation': min([
                    cleaning_results['execution_time'],
                    validation_results['execution_time'],
                    normalization_results['execution_time']
                ]),
                'slowest_operation': max([
                    cleaning_results['execution_time'],
                    validation_results['execution_time'],
                    normalization_results['execution_time'],
                    pipeline_results['execution_time']
                ])
            }
        }

        # Assert overall performance targets are met
        assert report['performance_summary']['all_targets_met'], "Performance targets not met"

        return report