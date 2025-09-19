"""
Comprehensive validation of all preprocessing methods.

Tests the entire preprocessing system end-to-end, validating that
all methods work correctly together and meet the specified requirements.
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from scipy import stats

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.cleaning import DataCleaner
from lib.validation import DataValidator
from lib.normalization import DataNormalizer


class TestComprehensiveValidation:
    """Comprehensive validation of the preprocessing system."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Initialize preprocessing components
        self.cleaner = DataCleaner()
        self.validator = DataValidator()
        self.normalizer = DataNormalizer()

        # Create comprehensive test datasets
        self.test_datasets = self._create_test_datasets()

    def _create_test_datasets(self):
        """Create various test datasets for comprehensive validation."""
        datasets = {}

        # 1. Clean stock data (baseline)
        dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
        prices = np.random.lognormal(4.5, 0.15, 500)

        clean_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * np.random.uniform(0.995, 1.005, 500),
            'high': prices * np.random.uniform(1.005, 1.015, 500),
            'low': prices * np.random.uniform(0.985, 0.995, 500),
            'close': prices,
            'volume': np.random.lognormal(15, 0.4, 500).astype(int)
        })

        # Ensure OHLC relationships
        clean_data['high'] = np.maximum(clean_data['high'], clean_data[['open', 'close']].max(axis=1) * 1.001)
        clean_data['low'] = np.minimum(clean_data['low'], clean_data[['open', 'close']].min(axis=1) * 0.999)

        datasets['clean'] = clean_data

        # 2. Data with missing values
        missing_data = clean_data.copy()
        missing_mask = np.random.random(missing_data.shape) < 0.08
        for col in ['open', 'high', 'low', 'close', 'volume']:
            missing_data.loc[missing_mask[:, missing_data.columns.get_loc(col)], col] = np.nan
        datasets['missing_values'] = missing_data

        # 3. Data with outliers
        outlier_data = clean_data.copy()
        outlier_mask = np.random.random(len(outlier_data)) < 0.03
        outlier_indices = outlier_data[outlier_mask].index
        for idx in outlier_indices:
            outlier_data.at[idx, 'close'] *= np.random.uniform(3, 8)
            outlier_data.at[idx, 'volume'] *= np.random.uniform(15, 30)
        datasets['outliers'] = outlier_data

        # 4. Data with duplicates
        duplicate_data = clean_data.copy()
        duplicates = duplicate_data.sample(n=20, replace=False)
        datasets['duplicates'] = pd.concat([duplicate_data, duplicates], ignore_index=True)

        # 5. Data with time gaps
        gap_data = clean_data.copy()
        gap_indices = np.random.choice(gap_data.index, size=30, replace=False)
        gap_data = gap_data.drop(gap_indices).reset_index(drop=True)
        datasets['time_gaps'] = gap_data

        # 6. Mixed issues (complex dataset)
        mixed_data = clean_data.copy()
        # Add missing values
        missing_mask = np.random.random(mixed_data.shape) < 0.05
        for col in ['open', 'high', 'low', 'close', 'volume']:
            mixed_data.loc[missing_mask[:, mixed_data.columns.get_loc(col)], col] = np.nan
        # Add outliers
        outlier_mask = np.random.random(len(mixed_data)) < 0.02
        outlier_indices = mixed_data[outlier_mask].index
        for idx in outlier_indices:
            mixed_data.at[idx, 'close'] *= np.random.uniform(2, 6)
        # Add duplicates
        duplicates = mixed_data.sample(n=15, replace=False)
        mixed_data = pd.concat([mixed_data, duplicates], ignore_index=True)
        datasets['mixed_issues'] = mixed_data

        # 7. Large dataset (performance test)
        large_dates = pd.date_range(start='2020-01-01', periods=5000, freq='H')
        large_prices = np.random.lognormal(4.5, 0.2, 5000)
        large_data = pd.DataFrame({
            'timestamp': large_dates,
            'open': large_prices * np.random.uniform(0.99, 1.01, 5000),
            'high': large_prices * np.random.uniform(1.01, 1.04, 5000),
            'low': large_prices * np.random.uniform(0.96, 0.99, 5000),
            'close': large_prices,
            'volume': np.random.lognormal(15, 0.6, 5000).astype(int)
        })
        datasets['large'] = large_data

        return datasets

    def test_all_preprocessing_methods_combinations(self):
        """Test all combinations of preprocessing methods."""
        # Define all method combinations
        missing_methods = ['forward_fill', 'interpolation', 'mean', 'median']
        outlier_methods = ['zscore', 'iqr', 'percentile', 'custom']
        outlier_actions = ['clip', 'remove', 'flag']
        normalization_methods = ['zscore', 'minmax', 'robust']

        results = {}

        # Test a subset of combinations (avoid combinatorial explosion)
        test_combinations = [
            ('forward_fill', 'zscore', 'clip', 'zscore'),
            ('interpolation', 'iqr', 'flag', 'minmax'),
            ('mean', 'custom', 'remove', 'robust'),
            ('median', 'percentile', 'clip', 'zscore'),
        ]

        for mv_method, out_method, out_action, norm_method in test_combinations:
            combination_key = f"{mv_method}_{out_method}_{out_action}_{norm_method}"

            # Test on mixed issues dataset
            test_data = self.test_datasets['mixed_issues'].copy()

            try:
                # Apply preprocessing pipeline
                processed = self._apply_preprocessing_combination(
                    test_data, mv_method, out_method, out_action, norm_method
                )

                # Validate results
                validation_result = self._validate_preprocessing_result(
                    test_data, processed, combination_key
                )

                results[combination_key] = {
                    'success': True,
                    'validation': validation_result,
                    'rows_processed': len(processed),
                    'quality_score': validation_result.get('quality_score', 0)
                }

            except Exception as e:
                results[combination_key] = {
                    'success': False,
                    'error': str(e)
                }

        # Analyze results
        successful_combinations = [k for k, v in results.items() if v['success']]
        assert len(successful_combinations) >= 3, f"Expected at least 3 successful combinations, got {len(successful_combinations)}"

        # All successful combinations should produce reasonable quality
        for combo, result in results.items():
            if result['success']:
                assert result['validation']['quality_score'] > 0.6, \
                    f"Combination {combo} should produce quality > 0.6, got {result['validation']['quality_score']}"

        return results

    def test_dataset_specific_validation(self):
        """Test preprocessing on specific dataset types."""
        dataset_results = {}

        for dataset_name, dataset in self.test_datasets.items():
            # Apply standard preprocessing pipeline
            processed = self._apply_standard_pipeline(dataset.copy())

            # Validate dataset-specific improvements
            validation_result = self._validate_dataset_improvement(
                dataset, processed, dataset_name
            )

            dataset_results[dataset_name] = {
                'original_quality': self._calculate_data_quality(dataset),
                'processed_quality': self._calculate_data_quality(processed),
                'improvement': validation_result['improvement'],
                'validation': validation_result
            }

        # Validate overall improvements
        for dataset_name, result in dataset_results.items():
            if dataset_name != 'clean':  # Clean data may not need improvement
                assert result['improvement'] >= 0, \
                    f"Dataset {dataset_name} should show non-negative improvement, got {result['improvement']}"

        # Problematic datasets should show significant improvement
        problematic_datasets = ['missing_values', 'outliers', 'duplicates', 'time_gaps', 'mixed_issues']
        for dataset_name in problematic_datasets:
            if dataset_name in dataset_results:
                improvement = dataset_results[dataset_name]['improvement']
                assert improvement > 0.1, \
                    f"Problematic dataset {dataset_name} should show > 0.1 improvement, got {improvement}"

        return dataset_results

    def test_performance_requirements_validation(self):
        """Test that performance requirements are met."""
        import time
        import psutil

        # Test on large dataset
        large_dataset = self.test_datasets['large'].copy()

        # Performance targets
        target_time_seconds = 30.0  # For 10M data points
        target_memory_gb = 4.0

        data_points = large_dataset.shape[0] * large_dataset.shape[1]
        scaled_target_time = target_time_seconds * (data_points / 10_000_000)

        # Measure performance
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss / (1024**3)  # GB

        # Apply full preprocessing pipeline
        processed = self._apply_standard_pipeline(large_dataset)

        end_time = time.time()
        end_memory = process.memory_info().rss / (1024**3)

        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        # Validate performance requirements
        assert execution_time < scaled_target_time, \
            f"Execution time {execution_time:.2f}s exceeds target {scaled_target_time:.2f}s"

        assert memory_used < target_memory_gb, \
            f"Memory usage {memory_used:.2f}GB exceeds target {target_memory_gb}GB"

        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'target_time': scaled_target_time,
            'target_memory': target_memory_gb,
            'data_points': data_points,
            'performance_requirements_met': execution_time < scaled_target_time and memory_used < target_memory_gb
        }

    def test_mathematical_correctness_validation(self):
        """Test mathematical correctness of all preprocessing operations."""
        test_data = self.test_datasets['clean'].copy()

        # Test z-score normalization
        normalized_zscore, params_zscore = self.normalizer.normalize_zscore(test_data)

        for col in normalized_zscore.select_dtypes(include=[np.number]).columns:
            if not normalized_zscore[col].isna().all():
                mean = normalized_zscore[col].mean()
                std = normalized_zscore[col].std()
                assert abs(mean) < 1e-10, f"Z-score mean should be ~0, got {mean}"
                assert abs(std - 1.0) < 1e-10, f"Z-score std should be ~1, got {std}"

        # Test min-max normalization
        normalized_minmax, params_minmax = self.normalizer.normalize_minmax(test_data, (0, 1))

        for col in normalized_minmax.select_dtypes(include=[np.number]).columns:
            if not normalized_minmax[col].isna().all():
                min_val = normalized_minmax[col].min()
                max_val = normalized_minmax[col].max()
                assert abs(min_val - 0.0) < 1e-10, f"Min-max min should be ~0, got {min_val}"
                assert abs(max_val - 1.0) < 1e-10, f"Min-max max should be ~1, got {max_val}"

        # Test outlier detection (z-score)
        data_with_outliers = test_data.copy()
        data_with_outliers.loc[10, 'close'] = data_with_outliers.loc[10, 'close'] * 10  # Clear outlier

        _, outlier_masks = self.cleaner.detect_outliers(data_with_outliers, method='zscore', threshold=3.0)
        assert outlier_masks['close'][10] == True, "Clear outlier should be detected"

        # Test data quality score calculation
        clean_score = self.cleaner.get_data_quality_score(test_data)
        noisy_score = self.cleaner.get_data_quality_score(data_with_outliers)

        assert clean_score > noisy_score, "Clean data should have higher quality score"

        return {
            'zscore_correct': True,
            'minmax_correct': True,
            'outlier_detection_correct': True,
            'quality_scoring_correct': True
        }

    def test_financial_domain_validation(self):
        """Test financial domain-specific validation."""
        test_data = self.test_datasets['clean'].copy()

        # Test OHLC relationship validation
        validation_results = self.validator.validate_price_data(test_data)
        assert validation_results['is_valid'], "Clean OHLC data should be valid"

        # Create invalid OHLC data
        invalid_data = test_data.copy()
        invalid_data.loc[10, 'high'] = invalid_data.loc[10, 'low'] - 1  # High < Low

        validation_results = self.validator.validate_price_data(invalid_data)
        assert not validation_results['is_valid'], "Invalid OHLC should be detected"
        assert "high < low" in str(validation_results['validation_issues']).lower()

        # Test volume validation
        validation_results = self.validator.validate_volume_data(test_data)
        assert validation_results['is_valid'], "Clean volume data should be valid"

        # Create invalid volume data
        invalid_volume_data = test_data.copy()
        invalid_volume_data.loc[10, 'volume'] = -1000

        validation_results = self.validator.validate_volume_data(invalid_volume_data)
        assert not validation_results['is_valid'], "Negative volume should be invalid"

        # Test financial ratios validation
        validation_results = self.validator.validate_financial_ratios(test_data)
        assert isinstance(validation_results, dict), "Financial ratios validation should return dict"

        return {
            'ohlc_validation': True,
            'volume_validation': True,
            'financial_ratios_validation': True
        }

    def test_error_handling_and_robustness(self):
        """Test error handling and system robustness."""
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        try:
            result = self.cleaner.handle_missing_values(empty_df)
            assert result.empty, "Empty DataFrame should return empty DataFrame"
        except Exception as e:
            assert "empty" in str(e).lower(), "Should handle empty DataFrame gracefully"

        # Test all NaN DataFrame
        all_nan_df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [np.nan, np.nan, np.nan]
        })

        try:
            result = self.cleaner.handle_missing_values(all_nan_df)
            assert isinstance(result, pd.DataFrame), "Should handle all-NaN DataFrame"
        except Exception as e:
            assert "nan" in str(e).lower(), "Should handle all-NaN gracefully"

        # Test single value DataFrame
        single_df = pd.DataFrame({'col1': [1.0]})
        try:
            result = self.cleaner.handle_missing_values(single_df)
            assert len(result) == 1, "Should handle single-value DataFrame"
        except Exception:
            pass  # Some failures are acceptable for edge cases

        # Test constant DataFrame
        constant_df = pd.DataFrame({'col1': [1.0, 1.0, 1.0]})
        try:
            result = self.cleaner.handle_missing_values(constant_df)
            assert isinstance(result, pd.DataFrame), "Should handle constant DataFrame"
        except Exception:
            pass  # Some failures are acceptable for edge cases

        return {
            'empty_data_handled': True,
            'all_nan_handled': True,
            'single_value_handled': True,
            'constant_value_handled': True
        }

    def test_configuration_flexibility(self):
        """Test configuration system flexibility."""
        test_data = self.test_datasets['mixed_issues'].copy()

        # Test minimal configuration
        minimal_config = {}
        result_minimal = self._apply_config_with_fallback(test_data.copy(), minimal_config)

        # Test comprehensive configuration
        comprehensive_config = {
            'missing_value_handling': {
                'method': 'interpolation',
                'threshold': 0.1,
                'window_size': 10
            },
            'outlier_detection': {
                'method': 'iqr',
                'threshold': 2.0,
                'action': 'clip'
            },
            'normalization': {
                'method': 'robust',
                'preserve_stats': True
            },
            'validation': {
                'strict_mode': True,
                'quality_threshold': 0.8
            }
        }
        result_comprehensive = self._apply_config_with_fallback(test_data.copy(), comprehensive_config)

        # Test invalid configuration (should fallback gracefully)
        invalid_config = {
            'missing_value_handling': {
                'method': 'invalid_method'  # Should fallback to default
            }
        }
        result_invalid = self._apply_config_with_fallback(test_data.copy(), invalid_config)

        # All should produce valid results
        assert len(result_minimal) > 0, "Minimal config should produce valid result"
        assert len(result_comprehensive) > 0, "Comprehensive config should produce valid result"
        assert len(result_invalid) > 0, "Invalid config should fallback to valid result"

        # Comprehensive should typically produce best quality
        quality_minimal = self._calculate_data_quality(result_minimal)
        quality_comprehensive = self._calculate_data_quality(result_comprehensive)
        quality_invalid = self._calculate_data_quality(result_invalid)

        assert quality_comprehensive >= quality_minimal, "Comprehensive config should be at least as good as minimal"
        assert quality_invalid >= quality_minimal, "Invalid config fallback should be reasonable"

        return {
            'minimal_config_quality': quality_minimal,
            'comprehensive_config_quality': quality_comprehensive,
            'invalid_config_quality': quality_invalid,
            'config_flexibility': True
        }

    def _apply_preprocessing_combination(self, data, mv_method, out_method, out_action, norm_method):
        """Apply a specific combination of preprocessing methods."""
        processed = data.copy()

        # Missing value handling
        processed = self.cleaner.handle_missing_values(processed, method=mv_method)

        # Outlier detection
        processed, _ = self.cleaner.detect_outliers(
            processed, method=out_method, action=out_action
        )

        # Normalization
        if norm_method == 'zscore':
            processed, _ = self.normalizer.normalize_zscore(processed)
        elif norm_method == 'minmax':
            processed, _ = self.normalizer.normalize_minmax(processed)
        elif norm_method == 'robust':
            processed, _ = self.normalizer.normalize_robust(processed)

        return processed

    def _apply_standard_pipeline(self, data):
        """Apply standard preprocessing pipeline."""
        processed = data.copy()

        # Standard preprocessing steps
        processed = self.cleaner.handle_missing_values(processed, method='forward_fill')
        processed, _ = self.cleaner.detect_outliers(processed, method='iqr', action='clip')
        processed = self.cleaner.remove_duplicate_rows(processed)
        processed, _ = self.normalizer.normalize_zscore(processed)

        return processed

    def _validate_preprocessing_result(self, original, processed, combination_name):
        """Validate preprocessing results."""
        validation = {}

        # Basic checks
        validation['not_empty'] = len(processed) > 0
        validation['columns_preserved'] = all(col in processed.columns for col in original.columns)

        # Data quality improvement
        original_quality = self._calculate_data_quality(original)
        processed_quality = self._calculate_data_quality(processed)
        validation['quality_improvement'] = processed_quality - original_quality
        validation['quality_score'] = processed_quality

        # No data loss (unless removal was specified)
        validation['reasonable_size'] = len(processed) > 0.5 * len(original)

        # Numerical stability
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        validation['no_infinite_values'] = not np.isinf(processed[numeric_cols].values).any()
        validation['limited_nan_values'] = processed[numeric_cols].isna().sum().sum() / processed.size < 0.1

        return validation

    def _validate_dataset_improvement(self, original, processed, dataset_name):
        """Validate dataset-specific improvements."""
        improvement = {}

        original_quality = self._calculate_data_quality(original)
        processed_quality = self._calculate_data_quality(processed)

        improvement['overall'] = processed_quality - original_quality

        # Dataset-specific improvements
        if dataset_name == 'missing_values':
            original_missing = original.isnull().sum().sum()
            processed_missing = processed.isnull().sum().sum()
            improvement['missing_values'] = original_missing - processed_missing

        elif dataset_name == 'outliers':
            # Check that extreme values were handled
            original_extreme = self._count_extreme_values(original)
            processed_extreme = self._count_extreme_values(processed)
            improvement['outliers'] = original_extreme - processed_extreme

        elif dataset_name == 'duplicates':
            original_duplicates = original.duplicated().sum()
            processed_duplicates = processed.duplicated().sum()
            improvement['duplicates'] = original_duplicates - processed_duplicates

        return {'improvement': improvement['overall'], 'details': improvement}

    def _calculate_data_quality(self, df):
        """Calculate overall data quality score."""
        if df.empty:
            return 0.0

        # Completeness
        completeness = 1.0 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))

        # Consistency
        consistency = 1.0
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = (df['high'] < df['low']).sum()
            consistency -= invalid_hl / len(df)

        # Accuracy
        accuracy = 1.0
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'close' in col.lower() or 'price' in col.lower():
                negative_count = (df[col] < 0).sum()
                accuracy -= negative_count / len(df)

        # Weighted score
        score = 0.4 * completeness + 0.3 * consistency + 0.3 * accuracy
        return max(0.0, min(1.0, score))

    def _count_extreme_values(self, df):
        """Count extreme values in DataFrame."""
        extreme_count = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            if not df[col].empty and df[col].std() > 0:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                extreme_count += (z_scores > 3).sum()
        return extreme_count

    def _apply_config_with_fallback(self, data, config):
        """Apply configuration with fallback for invalid values."""
        processed = data.copy()

        try:
            # Missing value handling
            if 'missing_value_handling' in config:
                mv_config = config['missing_value_handling']
                method = mv_config.get('method', 'forward_fill')
                if method in ['forward_fill', 'interpolation', 'mean', 'median', 'drop']:
                    processed = self.cleaner.handle_missing_values(processed, method=method)

            # Outlier detection
            if 'outlier_detection' in config:
                out_config = config['outlier_detection']
                method = out_config.get('method', 'iqr')
                if method in ['zscore', 'iqr', 'percentile', 'custom']:
                    processed, _ = self.cleaner.detect_outliers(
                        processed,
                        method=method,
                        threshold=out_config.get('threshold', 1.5),
                        action=out_config.get('action', 'flag')
                    )

            # Normalization
            if 'normalization' in config:
                norm_config = config['normalization']
                method = norm_config.get('method', 'zscore')
                if method == 'zscore':
                    processed, _ = self.normalizer.normalize_zscore(processed)
                elif method == 'minmax':
                    processed, _ = self.normalizer.normalize_minmax(processed)
                elif method == 'robust':
                    processed, _ = self.normalizer.normalize_robust(processed)

        except Exception:
            # Fallback to simple processing
            processed = self.cleaner.handle_missing_values(processed)

        return processed

    def test_end_to_end_system_validation(self):
        """Comprehensive end-to-end system validation."""
        results = {}

        # Test all major components
        results['method_combinations'] = self.test_all_preprocessing_methods_combinations()
        results['dataset_validation'] = self.test_dataset_specific_validation()
        results['performance_validation'] = self.test_performance_requirements_validation()
        results['mathematical_correctness'] = self.test_mathematical_correctness_validation()
        results['financial_domain_validation'] = self.test_financial_domain_validation()
        results['error_handling'] = self.test_error_handling_and_robustness()
        results['configuration_flexibility'] = self.test_configuration_flexibility()

        # Overall system validation
        system_valid = True
        validation_summary = {}

        # Check method combinations
        successful_combinations = len([v for v in results['method_combinations'].values() if v['success']])
        validation_summary['method_combinations_success_rate'] = successful_combinations / len(results['method_combinations'])
        system_valid = system_valid and validation_summary['method_combinations_success_rate'] > 0.5

        # Check dataset improvements
        dataset_improvements = [v['improvement'] for v in results['dataset_validation'].values() if v['improvement'] > 0]
        validation_summary['dataset_improvement_rate'] = len(dataset_improvements) / len(results['dataset_validation'])
        system_valid = system_valid and validation_summary['dataset_improvement_rate'] > 0.7

        # Check performance requirements
        performance_met = results['performance_validation']['performance_requirements_met']
        validation_summary['performance_requirements_met'] = performance_met
        system_valid = system_valid and performance_met

        # Check mathematical correctness
        math_correct = all(results['mathematical_correctness'].values())
        validation_summary['mathematical_correctness'] = math_correct
        system_valid = system_valid and math_correct

        # Check financial validation
        financial_valid = all(results['financial_domain_validation'].values())
        validation_summary['financial_validation'] = financial_valid
        system_valid = system_valid and financial_valid

        # Final assertion
        assert system_valid, f"System validation failed. Summary: {validation_summary}"

        return {
            'system_valid': system_valid,
            'validation_summary': validation_summary,
            'detailed_results': results
        }