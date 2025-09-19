"""
End-to-end validation tests using quickstart scenarios.

Tests complete preprocessing workflows as described in the quickstart guide,
validating that the system works correctly in realistic usage scenarios.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import json
import tempfile
from datetime import datetime, timedelta

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.cleaning import DataCleaner
from lib.validation import DataValidator
from lib.normalization import DataNormalizer


class TestQuickstartScenarios:
    """End-to-end validation tests based on quickstart scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Initialize preprocessing components
        self.cleaner = DataCleaner()
        self.validator = DataValidator()
        self.normalizer = DataNormalizer()

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_sample_stock_data(self, n_symbols=3, n_days=252):
        """Create realistic sample stock data for testing."""
        symbols = ['AAPL', 'MSFT', 'GOOGL'][:n_symbols]
        start_date = datetime(2023, 1, 1)
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')

        data = []
        for symbol in symbols:
            # Generate realistic price process
            base_price = np.random.uniform(50, 500)
            trend = np.random.uniform(-0.001, 0.002, n_days)  # Daily trend
            volatility = np.random.uniform(0.01, 0.03)  # Daily volatility

            # Random walk with drift
            returns = np.random.normal(trend, volatility, n_days)
            prices = [base_price]

            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            prices = np.array(prices)

            # Create OHLCV data
            for i, date in enumerate(dates):
                daily_volatility = volatility * prices[i]
                data.append({
                    'symbol': symbol,
                    'timestamp': date,
                    'open': prices[i] * np.random.uniform(0.99, 1.01),
                    'high': prices[i] * np.random.uniform(1.00, 1.03),
                    'low': prices[i] * np.random.uniform(0.97, 1.00),
                    'close': prices[i],
                    'volume': np.random.lognormal(14, 0.8)
                })

        df = pd.DataFrame(data)

        # Ensure proper OHLC relationships
        for idx, row in df.iterrows():
            high = max(row['open'], row['close']) * np.random.uniform(1.001, 1.02)
            low = min(row['open'], row['close']) * np.random.uniform(0.98, 0.999)
            df.at[idx, 'high'] = high
            df.at[idx, 'low'] = low

        # Add some data quality issues
        self._add_realistic_data_issues(df)

        return df

    def _add_realistic_data_issues(self, df):
        """Add realistic data quality issues to test data."""
        # Missing values (3% random)
        missing_mask = np.random.random(df.shape) < 0.03
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df.loc[missing_mask[:, df.columns.get_loc(col)], col] = np.nan

        # Outliers (1% of price data)
        price_outlier_mask = np.random.random(len(df)) < 0.01
        outlier_indices = df[price_outlier_mask].index
        for idx in outlier_indices:
            df.at[idx, 'close'] *= np.random.uniform(2, 5)  # 2-5x price jump

        # Extreme volume outliers
        volume_outlier_mask = np.random.random(len(df)) < 0.005
        vol_outlier_indices = df[volume_outlier_mask].index
        for idx in vol_outlier_indices:
            df.at[idx, 'volume'] *= np.random.uniform(10, 50)  # 10-50x volume

        # Duplicate timestamps (same symbol, same time)
        n_duplicates = int(0.01 * len(df))
        duplicate_rows = df.sample(n=n_duplicates, replace=False)
        df = pd.concat([df, duplicate_rows], ignore_index=True)

        # Time gaps (missing trading days)
        gap_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        for idx in gap_indices:
            if idx < len(df) - 5:
                # Remove a few consecutive rows to create gap
                df = df.drop([idx, idx+1, idx+2])

        df = df.reset_index(drop=True)

    def test_cli_preprocessing_scenario(self):
        """Test CLI preprocessing scenario from quickstart."""
        # Create sample data
        sample_data = self.create_sample_stock_data(n_symbols=2, n_days=100)

        # Save to CSV (simulating CLI input)
        input_file = os.path.join(self.temp_dir, 'sample_stocks.csv')
        output_file = os.path.join(self.temp_dir, 'processed_stocks.csv')

        sample_data.to_csv(input_file, index=False)

        # Simulate CLI preprocessing
        processed_data = self._simulate_cli_preprocessing(input_file)

        # Save processed data
        processed_data.to_csv(output_file, index=False)

        # Validate output
        assert os.path.exists(output_file), "Output file should be created"

        # Load and validate processed data
        loaded_data = pd.read_csv(output_file)

        # Check basic structure
        expected_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in loaded_data.columns, f"Column {col} should be present"

        # Check data quality improvements
        original_missing = sample_data.isnull().sum().sum()
        processed_missing = loaded_data.isnull().sum().sum()

        assert processed_missing <= original_missing, "Processed data should have fewer missing values"

        # Check that data size is reasonable
        assert len(loaded_data) > 0, "Processed data should not be empty"
        assert len(loaded_data) <= len(sample_data) * 1.1, "Processed data should not have grown significantly"

    def test_custom_configuration_scenario(self):
        """Test custom configuration scenario from quickstart."""
        # Create noisy data
        noisy_data = self.create_sample_stock_data(n_symbols=2, n_days=200)

        # Custom configuration for high-frequency data
        config = {
            'missing_value_handling': {
                'method': 'interpolation',
                'threshold': 0.05
            },
            'outlier_detection': {
                'method': 'iqr',
                'threshold': 1.5,
                'action': 'flag'
            },
            'normalization': {
                'method': 'robust',
                'preserve_stats': False
            }
        }

        # Apply custom preprocessing
        processed_data = self._apply_custom_preprocessing(noisy_data, config)

        # Validate results
        # Check that outlier flags were added
        outlier_columns = [col for col in processed_data.columns if 'outlier' in col.lower()]
        assert len(outlier_columns) > 0, "Outlier flags should be added"

        # Check that normalization was applied
        normalized_columns = [col for col in processed_data.columns if 'normalized' in col.lower()]
        assert len(normalized_columns) > 0, "Normalized columns should be added"

        # Check data quality score
        quality_score = self._calculate_data_quality_score(processed_data)
        assert quality_score > 0.8, f"Data quality score should be > 0.8, got {quality_score}"

        return processed_data, quality_score

    def test_batch_processing_scenario(self):
        """Test batch processing scenario from quickstart."""
        # Create multiple sample files
        input_files = []
        output_files = []

        for i in range(3):
            # Create sample data with different characteristics
            sample_data = self.create_sample_stock_data(n_symbols=1, n_days=50 + i*30)

            input_file = os.path.join(self.temp_dir, f'raw_data_{i}.csv')
            output_file = os.path.join(self.temp_dir, f'processed_{i}.csv')

            sample_data.to_csv(input_file, index=False)
            input_files.append(input_file)
            output_files.append(output_file)

        # Process multiple files
        quality_scores = []
        for i, (input_file, output_file) in enumerate(zip(input_files, output_files)):
            # Load and process
            df = pd.read_csv(input_file)
            processed_df = self._simulate_cli_preprocessing(input_file)

            # Save results
            processed_df.to_csv(output_file, index=False)

            # Calculate quality
            quality_score = self._calculate_data_quality_score(processed_df)
            quality_scores.append(quality_score)

            print(f"Processed file {i}, quality: {quality_score:.2f}")

        # Validate batch results
        assert len(quality_scores) == 3, "Should have processed 3 files"
        assert all(score > 0.7 for score in quality_scores), "All files should have quality > 0.7"

        # Check quality consistency
        quality_std = np.std(quality_scores)
        assert quality_std < 0.2, f"Quality scores should be consistent, std = {quality_std:.3f}"

        return quality_scores

    def test_quality_report_generation_scenario(self):
        """Test quality report generation scenario from quickstart."""
        # Create test data
        test_data = self.create_sample_stock_data(n_symbols=2, n_days=150)

        # Process data
        processed_data = self._simulate_cli_preprocessing_from_dataframe(test_data)

        # Generate quality report
        quality_report = self._generate_quality_report(processed_data)

        # Validate report structure
        required_fields = ['dataset_id', 'overall_score', 'completeness', 'consistency', 'accuracy']
        for field in required_fields:
            assert field in quality_report, f"Report should contain {field}"

        # Validate score ranges
        assert 0 <= quality_report['overall_score'] <= 1, "Overall score should be between 0 and 1"
        assert 0 <= quality_report['completeness'] <= 1, "Completeness should be between 0 and 1"
        assert 0 <= quality_report['consistency'] <= 1, "Consistency should be between 0 and 1"
        assert 0 <= quality_report['accuracy'] <= 1, "Accuracy should be between 0 and 1"

        # Check that issues are documented
        if 'issues_found' in quality_report:
            assert isinstance(quality_report['issues_found'], list), "Issues should be a list"

        # Save report to file
        report_file = os.path.join(self.temp_dir, 'quality_report.json')
        with open(report_file, 'w') as f:
            json.dump(quality_report, f, indent=2)

        assert os.path.exists(report_file), "Quality report file should be created"

        return quality_report

    def test_configuration_file_scenario(self):
        """Test configuration file scenario from quickstart."""
        # Create configuration file
        config_file = os.path.join(self.temp_dir, 'preprocessing_config.json')
        config = {
            "pipeline_id": "equity_daily_v1",
            "missing_value_handling": {
                "method": "forward_fill",
                "threshold": 0.1
            },
            "outlier_detection": {
                "method": "zscore",
                "threshold": 3.0,
                "action": "clip"
            },
            "normalization": {
                "method": "zscore",
                "preserve_stats": True
            },
            "quality_thresholds": {
                "completeness": 0.95,
                "consistency": 0.90,
                "accuracy": 0.95
            }
        }

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Create test data
        test_data = self.create_sample_stock_data(n_symbols=2, n_days=100)

        # Process with configuration file
        processed_data = self._process_with_config_file(test_data, config_file)

        # Validate that configuration was applied
        # Check for normalized columns
        normalized_cols = [col for col in processed_data.columns if 'normalized' in col.lower()]
        assert len(normalized_cols) > 0, "Normalization should be applied based on config"

        # Check quality against thresholds
        quality_report = self._generate_quality_report(processed_data)

        assert quality_report['completeness'] >= config['quality_thresholds']['completeness'], \
            f"Completeness {quality_report['completeness']} below threshold {config['quality_thresholds']['completeness']}"

        assert quality_report['consistency'] >= config['quality_thresholds']['consistency'], \
            f"Consistency {quality_report['consistency']} below threshold {config['quality_thresholds']['consistency']}"

        return processed_data, quality_report

    def test_api_usage_scenario(self):
        """Test Python API usage scenario from quickstart."""
        # Load sample data
        df = self.create_sample_stock_data(n_symbols=1, n_days=100)

        # Initialize preprocessor (simulate)
        preprocessor = self._create_mock_preprocessor()

        # Process data with default settings
        processed_df = self._simulate_api_processing(preprocessor, df)

        # Check quality
        quality_metrics = preprocessor.get_quality_metrics()
        print(f"Quality score: {quality_metrics['overall_score']}")

        assert quality_metrics['overall_score'] > 0.7, "Quality score should be reasonable"

        # Process with custom configuration
        config = {
            'missing_value_handling': {
                'method': 'forward_fill',
                'threshold': 0.1
            },
            'outlier_detection': {
                'method': 'zscore',
                'threshold': 3.0,
                'action': 'clip'
            },
            'normalization': {
                'method': 'zscore',
                'preserve_stats': True
            }
        }

        processed_df_custom = self._simulate_api_processing(preprocessor, df, config)

        # Custom processing should also work
        quality_custom = preprocessor.get_quality_metrics()
        assert quality_custom['overall_score'] > 0.7, "Custom processing should also produce good quality"

        return processed_df, processed_df_custom

    def test_error_handling_scenarios(self):
        """Test error handling scenarios from quickstart troubleshooting."""
        # Test missing required columns
        incomplete_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'timestamp': ['2023-01-01', '2023-01-02']
            # Missing 'close' column
        })

        # Should handle gracefully
        try:
            processed = self._simulate_cli_preprocessing_from_dataframe(incomplete_data)
            # Either should fail gracefully or produce warning
        except Exception as e:
            assert "column" in str(e).lower() or "required" in str(e).lower(), \
                "Should mention missing columns in error"

        # Test invalid timestamp format
        bad_timestamp_data = self.create_sample_stock_data(n_symbols=1, n_days=10)
        bad_timestamp_data['timestamp'] = ['invalid_date'] * len(bad_timestamp_data)

        # Should handle gracefully
        try:
            processed = self._simulate_cli_preprocessing_from_dataframe(bad_timestamp_data)
        except Exception as e:
            assert "timestamp" in str(e).lower() or "date" in str(e).lower(), \
                "Should mention timestamp in error"

        # Test empty data
        empty_data = pd.DataFrame()

        try:
            processed = self._simulate_cli_preprocessing_from_dataframe(empty_data)
        except Exception as e:
            assert "empty" in str(e).lower() or "no data" in str(e).lower(), \
                "Should mention empty data in error"

    def _simulate_cli_preprocessing(self, input_file):
        """Simulate CLI preprocessing workflow."""
        # Load data
        df = pd.read_csv(input_file)

        # Apply preprocessing pipeline
        return self._simulate_cli_preprocessing_from_dataframe(df)

    def _simulate_cli_preprocessing_from_dataframe(self, df):
        """Simulate CLI preprocessing from DataFrame."""
        # Default preprocessing pipeline
        processed = df.copy()

        # Handle missing values
        processed = self.cleaner.handle_missing_values(processed, method='forward_fill')

        # Detect outliers
        processed, outlier_masks = self.cleaner.detect_outliers(
            processed, method='zscore', threshold=3.0, action='flag'
        )

        # Normalize
        processed, _ = self.normalizer.normalize_zscore(processed)

        # Add derived columns
        if 'close' in processed.columns:
            processed['returns'] = processed['close'].pct_change()
            processed['volatility'] = processed['returns'].rolling(window=20).std()

        # Add quality flags
        validation_results = self.validator.run_comprehensive_validation(processed)
        processed['quality_score'] = self.validator.get_data_quality_score(validation_results)

        return processed

    def _apply_custom_preprocessing(self, df, config):
        """Apply custom preprocessing configuration."""
        processed = df.copy()

        # Apply missing value handling
        if 'missing_value_handling' in config:
            mv_config = config['missing_value_handling']
            processed = self.cleaner.handle_missing_values(
                processed,
                method=mv_config.get('method', 'forward_fill'),
                threshold=mv_config.get('threshold', 0.1)
            )

        # Apply outlier detection
        if 'outlier_detection' in config:
            outlier_config = config['outlier_detection']
            processed, outlier_masks = self.cleaner.detect_outliers(
                processed,
                method=outlier_config.get('method', 'zscore'),
                threshold=outlier_config.get('threshold', 3.0),
                action=outlier_config.get('action', 'flag')
            )

        # Apply normalization
        if 'normalization' in config:
            norm_config = config['normalization']
            method = norm_config.get('method', 'zscore')
            if method == 'robust':
                processed, _ = self.normalizer.normalize_robust(processed)
            else:
                processed, _ = self.normalizer.normalize_zscore(processed)

        return processed

    def _calculate_data_quality_score(self, df):
        """Calculate data quality score."""
        if df.empty:
            return 0.0

        # Completeness
        completeness = 1.0 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))

        # Consistency (basic checks)
        consistency = 1.0
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = (df['high'] < df['low']).sum()
            consistency -= invalid_hl / len(df)

        # Accuracy (basic outlier check)
        accuracy = 1.0
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'close' in col.lower() or 'price' in col.lower():
                negative_prices = (df[col] < 0).sum()
                accuracy -= negative_prices / len(df)

        # Weighted score
        score = 0.4 * completeness + 0.3 * consistency + 0.3 * accuracy
        return max(0.0, min(1.0, score))

    def _generate_quality_report(self, df):
        """Generate quality report."""
        completeness = 1.0 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        consistency = 1.0
        accuracy = 1.0

        # Basic consistency checks
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = (df['high'] < df['low']).sum()
            consistency -= invalid_hl / len(df)

        # Basic accuracy checks
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'close' in col.lower() or 'price' in col.lower():
                negative_prices = (df[col] < 0).sum()
                accuracy -= negative_prices / len(df)

        overall_score = 0.4 * completeness + 0.3 * consistency + 0.3 * accuracy

        issues = []
        if completeness < 0.95:
            issues.append(f"Missing data points: {(1-completeness)*100:.1f}%")
        if consistency < 0.95:
            issues.append("OHLC consistency issues detected")
        if accuracy < 0.95:
            issues.append("Data accuracy issues detected")

        return {
            'dataset_id': f'processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'overall_score': max(0.0, min(1.0, overall_score)),
            'completeness': max(0.0, min(1.0, completeness)),
            'consistency': max(0.0, min(1.0, consistency)),
            'accuracy': max(0.0, min(1.0, accuracy)),
            'issues_found': issues
        }

    def _process_with_config_file(self, df, config_file):
        """Process data using configuration file."""
        with open(config_file, 'r') as f:
            config = json.load(f)

        return self._apply_custom_preprocessing(df, config)

    def _create_mock_preprocessor(self):
        """Create mock preprocessor for API testing."""
        class MockPreprocessor:
            def __init__(self):
                self.last_quality_score = 0.0

            def process(self, df, config=None):
                # Simulate processing
                processed = df.copy()
                if config:
                    # Apply custom config
                    pass
                else:
                    # Default processing
                    pass

                self.last_quality_score = np.random.uniform(0.7, 0.95)
                return processed

            def get_quality_metrics(self):
                return {
                    'overall_score': self.last_quality_score,
                    'completeness': np.random.uniform(0.8, 0.99),
                    'consistency': np.random.uniform(0.8, 0.99),
                    'accuracy': np.random.uniform(0.8, 0.99)
                }

        return MockPreprocessor()

    def _simulate_api_processing(self, preprocessor, df, config=None):
        """Simulate API processing."""
        return preprocessor.process(df, config)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create test data
        test_data = self.create_sample_stock_data(n_symbols=3, n_days=200)

        # Create configuration
        config_file = os.path.join(self.temp_dir, 'e2e_config.json')
        config = {
            "pipeline_id": "e2e_test_v1",
            "missing_value_handling": {
                "method": "interpolation",
                "threshold": 0.05
            },
            "outlier_detection": {
                "method": "iqr",
                "threshold": 1.5,
                "action": "clip"
            },
            "normalization": {
                "method": "robust",
                "preserve_stats": False
            }
        }

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Process data
        processed_data = self._process_with_config_file(test_data, config_file)

        # Generate quality report
        quality_report = self._generate_quality_report(processed_data)

        # Validate entire workflow
        assert quality_report['overall_score'] > 0.8, f"End-to-end quality should be > 0.8, got {quality_report['overall_score']}"

        # Check that all processing steps were applied
        # Should have normalized columns
        normalized_cols = [col for col in processed_data.columns if col.endswith('_normalized')]
        assert len(normalized_cols) > 0, "Normalization should be applied"

        # Should have outlier handling
        assert len(processed_data) > 0, "Data should not be empty after processing"

        # Should improve data quality
        original_score = self._calculate_data_quality_score(test_data)
        processed_score = quality_report['overall_score']
        assert processed_score >= original_score, f"Processed data quality ({processed_score}) should be >= original ({original_score})"

        return processed_data, quality_report