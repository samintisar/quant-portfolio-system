"""
Preprocessing System Usage Examples

This file contains practical examples of using the preprocessing system
for various financial data scenarios.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import sys

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.cleaning import DataCleaner
from lib.validation import DataValidator
from lib.normalization import DataNormalizer


def example_1_basic_stock_preprocessing():
    """Example 1: Basic stock data preprocessing."""
    print("=== Example 1: Basic Stock Data Preprocessing ===")

    # Create sample stock data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = np.random.lognormal(4.5, 0.15, 100)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.995, 1.005, 100),
        'high': prices * np.random.uniform(1.005, 1.015, 100),
        'low': prices * np.random.uniform(0.985, 0.995, 100),
        'close': prices,
        'volume': np.random.lognormal(15, 0.4, 100).astype(int)
    })

    # Add some data quality issues
    df.loc[10:15, 'close'] = np.nan  # Missing values
    df.loc[20, 'close'] *= 5  # Outlier
    df.loc[25, 'volume'] = -1000  # Invalid volume

    print(f"Original data shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Initialize preprocessing components
    cleaner = DataCleaner()
    validator = DataValidator()
    normalizer = DataNormalizer()

    # Step 1: Clean the data
    cleaned = cleaner.handle_missing_values(df, method='forward_fill')
    cleaned, outlier_masks = cleaner.detect_outliers(cleaned, method='zscore', action='clip')

    print(f"After cleaning shape: {cleaned.shape}")
    print(f"Missing values after cleaning: {cleaned.isnull().sum().sum()}")

    # Step 2: Validate data quality
    validation_results = validator.run_comprehensive_validation(cleaned)
    quality_score = validator.get_data_quality_score(validation_results)

    print(f"Data quality score: {quality_score:.3f}")

    # Step 3: Normalize data
    normalized, norm_params = normalizer.normalize_zscore(cleaned)

    print("Preprocessing completed successfully!")
    return cleaned, normalized, quality_score


def example_2_custom_configuration():
    """Example 2: Using custom configuration."""
    print("\n=== Example 2: Custom Configuration ===")

    # Create configuration
    config = {
        "pipeline_id": "custom_equity_v1",
        "missing_value_handling": {
            "method": "interpolation",
            "threshold": 0.05,
            "window_size": 5
        },
        "outlier_detection": {
            "method": "iqr",
            "threshold": 1.5,
            "action": "flag"
        },
        "normalization": {
            "method": "robust",
            "preserve_stats": True
        },
        "quality_thresholds": {
            "completeness": 0.95,
            "consistency": 0.90,
            "accuracy": 0.95
        }
    }

    # Create test data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    prices = np.random.lognormal(4.0, 0.25, 200)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.99, 1.01, 200),
        'high': prices * np.random.uniform(1.01, 1.05, 200),
        'low': prices * np.random.uniform(0.95, 0.99, 200),
        'close': prices,
        'volume': np.random.lognormal(15, 0.6, 200).astype(int)
    })

    # Apply custom preprocessing
    cleaner = DataCleaner()
    validator = DataValidator()
    normalizer = DataNormalizer()

    # Apply configuration
    processed = df.copy()

    # Missing value handling
    mv_config = config['missing_value_handling']
    processed = cleaner.handle_missing_values(
        processed,
        method=mv_config['method'],
        threshold=mv_config['threshold'],
        window_size=mv_config['window_size']
    )

    # Outlier detection
    out_config = config['outlier_detection']
    processed, outlier_masks = cleaner.detect_outliers(
        processed,
        method=out_config['method'],
        threshold=out_config['threshold'],
        action=out_config['action']
    )

    # Normalization
    norm_config = config['normalization']
    if norm_config['method'] == 'robust':
        processed, norm_params = normalizer.normalize_robust(processed)
    else:
        processed, norm_params = normalizer.normalize_zscore(processed)

    # Validate against quality thresholds
    validation_results = validator.run_comprehensive_validation(processed)
    quality_score = validator.get_data_quality_score(validation_results)

    thresholds = config['quality_thresholds']
    quality_met = (
        quality_score >= thresholds['completeness'] and
        validation_results['detailed_results']['structure']['is_valid'] and
        validation_results['detailed_results']['prices']['is_valid']
    )

    print(f"Custom configuration applied")
    print(f"Quality score: {quality_score:.3f}")
    print(f"Quality thresholds met: {quality_met}")

    return processed, config, quality_score


def example_3_multi_asset_processing():
    """Example 3: Processing multiple asset types."""
    print("\n=== Example 3: Multi-Asset Processing ===")

    # Create data for different asset types
    dates = pd.date_range(start='2023-01-01', periods=150, freq='D')

    # Equities
    equity_prices = np.random.lognormal(4.5, 0.15, 150)
    equity_data = pd.DataFrame({
        'timestamp': dates,
        'asset_type': 'equity',
        'open': equity_prices * np.random.uniform(0.995, 1.005, 150),
        'high': equity_prices * np.random.uniform(1.005, 1.015, 150),
        'low': equity_prices * np.random.uniform(0.985, 0.995, 150),
        'close': equity_prices,
        'volume': np.random.lognormal(15, 0.4, 150).astype(int)
    })

    # Bonds (different characteristics)
    bond_prices = 100 + np.random.normal(0, 2, 150)  # Bonds around $100
    bond_data = pd.DataFrame({
        'timestamp': dates,
        'asset_type': 'bond',
        'price': bond_prices,
        'yield': np.random.normal(0.03, 0.01, 150),  # 3% yield with 1% std
        'duration': np.random.uniform(2, 10, 150)
    })

    # Combine data
    combined_data = pd.concat([equity_data, bond_data], ignore_index=True)

    print(f"Combined data shape: {combined_data.shape}")
    print(f"Asset types: {combined_data['asset_type'].unique()}")

    # Asset-specific preprocessing
    cleaner = DataCleaner()
    validator = DataValidator()
    normalizer = DataNormalizer()

    processed_assets = {}

    for asset_type in combined_data['asset_type'].unique():
        print(f"\nProcessing {asset_type} data...")

        # Filter data for this asset type
        asset_data = combined_data[combined_data['asset_type'] == asset_type].copy()

        if asset_type == 'equity':
            # Equity-specific preprocessing
            cleaned = cleaner.handle_missing_values(asset_data, method='forward_fill')
            cleaned, _ = cleaner.detect_outliers(cleaned, method='iqr', action='clip')
            normalized, _ = normalizer.normalize_zscore(cleaned)

        elif asset_type == 'bond':
            # Bond-specific preprocessing
            # Bonds have different characteristics and requirements
            cleaned = asset_data.copy()

            # Handle missing bond-specific data
            if 'price' in cleaned.columns:
                cleaned = cleaner.handle_missing_values(cleaned, method='interpolation')

            # Bond yield validation (should be positive)
            if 'yield' in cleaned.columns:
                cleaned.loc[cleaned['yield'] < 0, 'yield'] = 0.001  # Floor at 0.1%

            # Normalize bond data differently
            normalized, _ = normalizer.normalize_minmax(cleaned, feature_range=(0, 1))

        processed_assets[asset_type] = {
            'cleaned': cleaned,
            'normalized': normalized
        }

        # Validate quality
        validation_results = validator.run_comprehensive_validation(cleaned)
        quality_score = validator.get_data_quality_score(validation_results)
        print(f"{asset_type} quality score: {quality_score:.3f}")

    print("\nMulti-asset processing completed!")
    return processed_assets


def example_4_real_time_processing_simulation():
    """Example 4: Simulate real-time data processing."""
    print("\n=== Example 4: Real-Time Processing Simulation ===")

    # Simulate real-time data stream
    cleaner = DataCleaner()
    normalizer = DataNormalizer()

    # Process data in batches
    batch_size = 10
    n_batches = 10

    print(f"Processing {n_batches} batches of {batch_size} records each...")

    all_results = []

    for batch_num in range(n_batches):
        # Generate batch data
        start_time = datetime.now() + timedelta(minutes=batch_num * batch_size)
        batch_dates = pd.date_range(start=start_time, periods=batch_size, freq='1min')

        batch_prices = np.random.lognormal(4.5, 0.1, batch_size)
        batch_data = pd.DataFrame({
            'timestamp': batch_dates,
            'price': batch_prices,
            'volume': np.random.lognormal(14, 0.3, batch_size).astype(int)
        })

        # Add some noise/issues
        if np.random.random() < 0.3:  # 30% chance of missing data
            batch_data.loc[np.random.randint(0, batch_size), 'price'] = np.nan

        if np.random.random() < 0.2:  # 20% chance of outlier
            outlier_idx = np.random.randint(0, batch_size)
            batch_data.loc[outlier_idx, 'price'] *= np.random.uniform(3, 8)

        # Process batch
        start_process = datetime.now()

        # Quick preprocessing for real-time
        cleaned = cleaner.handle_missing_values(batch_data, method='forward_fill')
        cleaned, _ = cleaner.detect_outliers(cleaned, method='zscore', action='clip')
        normalized, _ = normalizer.normalize_zscore(cleaned)

        process_time = (datetime.now() - start_process).total_seconds()

        # Quality check
        quality_score = cleaner.get_data_quality_score(cleaned)

        batch_result = {
            'batch_num': batch_num + 1,
            'records_processed': len(normalized),
            'processing_time_ms': process_time * 1000,
            'quality_score': quality_score,
            'issues_found': cleaned.isnull().sum().sum()
        }

        all_results.append(batch_result)

        print(f"Batch {batch_num + 1}: {process_time*1000:.1f}ms, "
              f"Quality: {quality_score:.3f}, "
              f"Issues: {batch_result['issues_found']}")

    # Summary
    avg_time = np.mean([r['processing_time_ms'] for r in all_results])
    avg_quality = np.mean([r['quality_score'] for r in all_results])

    print(f"\nReal-time processing summary:")
    print(f"Average processing time: {avg_time:.1f}ms per batch")
    print(f"Average quality score: {avg_quality:.3f}")
    print(f"Total records processed: {sum(r['records_processed'] for r in all_results)}")

    return all_results


def example_5_quality_reporting():
    """Example 5: Generate quality reports."""
    print("\n=== Example 5: Quality Reporting ===")

    # Create test data with various issues
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    prices = np.random.lognormal(4.5, 0.2, 300)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.99, 1.01, 300),
        'high': prices * np.random.uniform(1.01, 1.04, 300),
        'low': prices * np.random.uniform(0.96, 0.99, 300),
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, 300).astype(int)
    })

    # Add various data quality issues
    # Missing values
    missing_mask = np.random.random(df.shape) < 0.05
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df.loc[missing_mask[:, df.columns.get_loc(col)], col] = np.nan

    # Outliers
    outlier_mask = np.random.random(len(df)) < 0.02
    df.loc[outlier_mask, 'close'] *= np.random.uniform(3, 10)

    # Duplicates
    duplicates = df.sample(n=20, replace=False)
    df = pd.concat([df, duplicates], ignore_index=True)

    print(f"Test data created with {len(df)} records")
    print(f"Initial missing values: {df.isnull().sum().sum()}")
    print(f"Initial duplicates: {df.duplicated().sum()}")

    # Process data
    cleaner = DataCleaner()
    validator = DataValidator()
    normalizer = DataNormalizer()

    # Apply preprocessing
    cleaned = cleaner.handle_missing_values(df, method='forward_fill')
    cleaned, outlier_masks = cleaner.detect_outliers(cleaned, method='iqr', action='clip')
    cleaned = cleaner.remove_duplicate_rows(cleaned)
    normalized, _ = normalizer.normalize_zscore(cleaned)

    # Generate comprehensive quality report
    validation_results = validator.run_comprehensive_validation(normalized)

    # Create detailed quality report
    quality_report = {
        'dataset_info': {
            'original_records': len(df),
            'processed_records': len(normalized),
            'processing_date': datetime.now().isoformat(),
            'pipeline_version': '1.0.0'
        },
        'quality_metrics': {
            'overall_score': validator.get_data_quality_score(validation_results),
            'completeness': 1.0 - (normalized.isnull().sum().sum() / (normalized.shape[0] * normalized.shape[1])),
            'consistency': validation_results['detailed_results'].get('time_series', {}).get('is_valid', True),
            'accuracy': validation_results['detailed_results'].get('prices', {}).get('is_valid', True)
        },
        'preprocessing_summary': {
            'missing_values_handled': df.isnull().sum().sum() - normalized.isnull().sum().sum(),
            'outliers_detected': sum(mask.sum() for mask in outlier_masks.values()),
            'duplicates_removed': df.duplicated().sum(),
            'records_dropped': len(df) - len(normalized)
        },
        'validation_details': validation_results,
        'recommendations': validator._generate_recommendations(validation_results['detailed_results'])
    }

    # Print report
    print("\nQuality Report Summary:")
    print(f"Overall Quality Score: {quality_report['quality_metrics']['overall_score']:.3f}")
    print(f"Completeness: {quality_report['quality_metrics']['completeness']:.3f}")
    print(f"Missing Values Handled: {quality_report['preprocessing_summary']['missing_values_handled']}")
    print(f"Outliers Detected: {quality_report['preprocessing_summary']['outliers_detected']}")
    print(f"Duplicates Removed: {quality_report['preprocessing_summary']['duplicates_removed']}")

    if quality_report['recommendations']:
        print("\nRecommendations:")
        for rec in quality_report['recommendations']:
            print(f"  - {rec}")

    # Save report to file
    report_filename = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)

    print(f"\nDetailed report saved to: {report_filename}")

    return quality_report


def example_6_performance_benchmarking():
    """Example 6: Performance benchmarking."""
    print("\n=== Example 6: Performance Benchmarking ===")

    import time
    import psutil

    # Create large dataset for benchmarking
    n_records = 50000
    print(f"Creating benchmark dataset with {n_records} records...")

    dates = pd.date_range(start='2020-01-01', periods=n_records, freq='H')
    prices = np.random.lognormal(4.5, 0.2, n_records)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.99, 1.01, n_records),
        'high': prices * np.random.uniform(1.01, 1.04, n_records),
        'low': prices * np.random.uniform(0.96, 0.99, n_records),
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, n_records).astype(int)
    })

    # Add data quality issues
    missing_mask = np.random.random(df.shape) < 0.05
    for col in df.columns:
        if col != 'timestamp':
            df.loc[missing_mask[:, df.columns.get_loc(col)], col] = np.nan

    print(f"Benchmark dataset ready: {df.shape}")

    # Initialize components
    cleaner = DataCleaner()
    validator = DataValidator()
    normalizer = DataNormalizer()

    # Benchmark individual operations
    operations = [
        ("Missing Value Handling", lambda d: cleaner.handle_missing_values(d, 'forward_fill')),
        ("Outlier Detection", lambda d: cleaner.detect_outliers(d, 'iqr', 'clip')[0]),
        ("Duplicate Removal", lambda d: cleaner.remove_duplicate_rows(d)),
        ("Z-Score Normalization", lambda d: normalizer.normalize_zscore(d)[0]),
        ("Comprehensive Validation", lambda d: validator.run_comprehensive_validation(d))
    ]

    benchmark_results = []

    for op_name, op_func in operations:
        # Measure memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024**2)  # MB

        # Measure execution time
        start_time = time.time()
        result = op_func(df.copy())
        execution_time = time.time() - start_time

        # Measure memory after
        mem_after = process.memory_info().rss / (1024**2)  # MB
        memory_used = mem_after - mem_before

        result_info = {
            'operation': op_name,
            'execution_time': execution_time,
            'memory_used_mb': memory_used,
            'records_per_second': len(df) / execution_time
        }

        benchmark_results.append(result_info)

        print(f"{op_name}: {execution_time:.3f}s, {memory_used:.1f}MB, {len(df)/execution_time:.0f} records/sec")

    # Full pipeline benchmark
    print("\nFull Pipeline Benchmark:")
    mem_before = process.memory_info().rss / (1024**2)
    start_time = time.time()

    # Apply full preprocessing pipeline
    processed = cleaner.handle_missing_values(df.copy(), method='forward_fill')
    processed, _ = cleaner.detect_outliers(processed, method='iqr', action='clip')
    processed = cleaner.remove_duplicate_rows(processed)
    processed, _ = normalizer.normalize_zscore(processed)
    validation_results = validator.run_comprehensive_validation(processed)

    total_time = time.time() - start_time
    mem_after = process.memory_info().rss / (1024**2)
    total_memory = mem_after - mem_before

    print(f"Full pipeline: {total_time:.3f}s, {total_memory:.1f}MB")
    print(f"Quality score: {validator.get_data_quality_score(validation_results):.3f}")

    # Performance summary
    print("\nPerformance Summary:")
    total_ops_time = sum(r['execution_time'] for r in benchmark_results)
    print(f"Total individual operations time: {total_ops_time:.3f}s")
    print(f"Pipeline efficiency: {(total_ops_time/total_time)*100:.1f}%")

    return benchmark_results


def main():
    """Run all examples."""
    print("Data Preprocessing System Usage Examples")
    print("=" * 50)

    try:
        # Run all examples
        example_1_basic_stock_preprocessing()
        example_2_custom_configuration()
        example_3_multi_asset_processing()
        example_4_real_time_processing_simulation()
        example_5_quality_reporting()
        example_6_performance_benchmarking()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()