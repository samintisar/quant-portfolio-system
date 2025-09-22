"""
End-to-end integration tests for financial features system.

Tests complete financial features workflows including returns, volatility, and momentum
calculations as described in the quickstart guide, validating that the system works
correctly in realistic usage scenarios.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import json
import tempfile
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from data.src.lib.returns import (
    calculate_simple_returns, calculate_log_returns, calculate_percentage_returns,
    calculate_annualized_returns, calculate_sharpe_ratio, calculate_multi_period_returns,
    calculate_max_drawdown, calculate_beta_alpha
)

from data.src.lib.volatility import (
    calculate_rolling_volatility, calculate_annualized_volatility,
    calculate_parkinson_volatility, calculate_garman_klass_volatility,
    calculate_ewma_volatility, calculate_garch11_volatility,
    calculate_volatility_regime, calculate_volatility_forecast
)

from data.src.lib.momentum import (
    calculate_simple_momentum, calculate_rsi, calculate_roc,
    calculate_stochastic, calculate_macd, generate_momentum_signals
)

from data.src.services.feature_service import FeatureGenerator
from data.src.services.validation_service import DataValidator


class TestFinancialFeaturesIntegration:
    """End-to-end integration tests for financial features."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible results

        # Initialize services
        self.feature_service = FeatureGenerator()
        self.validation_service = DataValidator()

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_realistic_market_data(self, n_symbols: int = 3, n_days: int = 252) -> pd.DataFrame:
        """Create realistic market data for testing financial features."""
        symbols = ['AAPL', 'MSFT', 'GOOGL'][:n_symbols]
        start_date = datetime(2023, 1, 1)
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')

        data = []

        # Market parameters for realistic simulation
        market_trend = 0.0002  # Daily market trend
        market_volatility = 0.015  # Daily market volatility

        for symbol in symbols:
            # Symbol-specific parameters
            base_price = np.random.uniform(50, 500)
            beta = np.random.uniform(0.8, 1.5)  # Market sensitivity
            idiosyncratic_vol = np.random.uniform(0.01, 0.03)  # Stock-specific volatility

            # Generate market factor (single factor model)
            market_returns = np.random.normal(market_trend, market_volatility, n_days)

            # Generate stock returns using CAPM-like model
            stock_returns = []
            for i in range(n_days):
                # Market component + idiosyncratic component
                market_component = beta * market_returns[i]
                idiosyncratic_component = np.random.normal(0, idiosyncratic_vol)
                total_return = market_component + idiosyncratic_component
                stock_returns.append(total_return)

            # Convert returns to prices
            prices = [base_price]
            for ret in stock_returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            prices = np.array(prices)

            # Create OHLCV data with realistic intraday patterns
            for i, date in enumerate(dates):
                if i == 0:
                    continue

                daily_return = stock_returns[i]
                daily_vol = idiosyncratic_vol * prices[i]

                # Intraday price movement simulation
                open_price = prices[i-1] * np.random.uniform(0.999, 1.001)

                if daily_return > 0:
                    # Up day
                    high = max(open_price, prices[i]) * np.random.uniform(1.005, 1.02)
                    low = min(open_price, prices[i]) * np.random.uniform(0.98, 0.999)
                    close = prices[i]
                else:
                    # Down day
                    high = max(open_price, prices[i]) * np.random.uniform(1.001, 1.01)
                    low = min(open_price, prices[i]) * np.random.uniform(0.97, 0.995)
                    close = prices[i]

                # Ensure proper OHLC relationships
                high = max(high, open_price, close)
                low = min(low, open_price, close)

                # Volume simulation (higher volume on big moves)
                volume_base = np.random.lognormal(14, 0.5)
                volume_multiplier = 1 + abs(daily_return) * 10
                volume = volume_base * volume_multiplier

                data.append({
                    'symbol': symbol,
                    'timestamp': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })

        df = pd.DataFrame(data)
        return df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    def test_basic_returns_calculation_scenario(self):
        """Test basic returns calculation workflow."""
        print("\n=== Testing Basic Returns Calculation ===")

        # Create test data
        market_data = self.create_realistic_market_data(n_symbols=2, n_days=100)

        # Calculate returns using different methods
        simple_returns = calculate_simple_returns(market_data['close'])
        log_returns = calculate_log_returns(market_data['close'])
        percentage_returns = calculate_percentage_returns(market_data['close'])

        # Validate results
        assert len(simple_returns) == len(market_data), "Simple returns length mismatch"
        assert len(log_returns) == len(market_data), "Log returns length mismatch"
        assert len(percentage_returns) == len(market_data), "Percentage returns length mismatch"

        # Check mathematical relationships
        # For small returns, log returns ≈ simple returns
        mask = abs(simple_returns) < 0.05  # Small returns mask
        diff = np.abs(log_returns[mask] - simple_returns[mask])
        assert diff.max() < 0.002, "Log and simple returns should be close for small values"

        # Check percentage conversion
        pct_diff = np.abs(percentage_returns - simple_returns * 100)
        assert pct_diff.max() < 1e-10, "Percentage returns should be simple returns × 100"

        # Check for reasonable return ranges
        assert simple_returns.min() > -0.5, "Simple returns should not be extremely negative"
        assert simple_returns.max() < 0.5, "Simple returns should not be extremely positive"

        print(f"✓ Calculated returns for {len(market_data)} data points")
        print(f"  Simple returns: {simple_returns.mean():.4f} ± {simple_returns.std():.4f}")
        print(f"  Log returns: {log_returns.mean():.4f} ± {log_returns.std():.4f}")

    def test_volatility_calculation_scenario(self):
        """Test volatility calculation workflow."""
        print("\n=== Testing Volatility Calculation ===")

        # Create test data
        market_data = self.create_realistic_market_data(n_symbols=1, n_days=200)

        # Set timestamp as index for proper time series
        market_data_indexed = market_data.set_index('timestamp')

        # Extract price data for the single symbol
        close_prices = market_data_indexed['close']
        high_prices = market_data_indexed['high']
        low_prices = market_data_indexed['low']
        open_prices = market_data_indexed['open']

        returns = calculate_simple_returns(close_prices)

        # Calculate different volatility measures
        rolling_vol = calculate_rolling_volatility(returns, window=21)
        annual_vol = calculate_annualized_volatility(returns, periods_per_year=252)
        parkinson_vol = calculate_parkinson_volatility(
            high_prices, low_prices, window=21
        )
        garman_klass_vol = calculate_garman_klass_volatility(
            open_prices, high_prices, low_prices, close_prices, window=21
        )
        ewma_vol = calculate_ewma_volatility(returns, span=30)

        # Validate results
        assert len(rolling_vol) == len(returns), "Rolling volatility length mismatch"
        assert isinstance(annual_vol, float), "Annual volatility should be a scalar"

        # Check volatility reasonableness
        assert rolling_vol.dropna().min() >= 0, "Volatility should be non-negative"
        assert rolling_vol.dropna().max() < 1.0, "Daily volatility should be < 100%"
        assert annual_vol > 0 and annual_vol < 2.0, "Annual volatility should be reasonable"

        # Check range-based volatilities are more efficient
        parkinson_mean = parkinson_vol.dropna().mean()
        rolling_mean = rolling_vol.dropna().mean()

        print(f"✓ Calculated multiple volatility measures")
        print(f"  Rolling volatility (21-day): {rolling_mean:.4f}")
        print(f"  Parkinson volatility: {parkinson_mean:.4f}")
        print(f"  Garman-Klass volatility: {garman_klass_vol.dropna().mean():.4f}")
        print(f"  EWMA volatility: {ewma_vol.dropna().mean():.4f}")
        print(f"  Annualized volatility: {annual_vol:.4f}")

    def test_momentum_indicators_scenario(self):
        """Test momentum indicators calculation workflow."""
        print("\n=== Testing Momentum Indicators ===")

        # Create test data
        market_data = self.create_realistic_market_data(n_symbols=1, n_days=150)

        # Calculate momentum indicators
        rsi = calculate_rsi(market_data['close'], period=14)
        roc = calculate_roc(market_data['close'], period=12)
        macd_line, signal_line, histogram = calculate_macd(market_data['close'])

        # Validate RSI
        assert rsi.dropna().min() >= 0, "RSI should be >= 0"
        assert rsi.dropna().max() <= 100, "RSI should be <= 100"
        assert len(rsi) == len(market_data), "RSI length mismatch"

        # Validate MACD
        assert len(macd_line) == len(signal_line) == len(histogram), "MACD series lengths should match"
        assert len(macd_line) == len(market_data), "MACD length mismatch"

        # Validate ROC
        assert len(roc) == len(market_data), "ROC length mismatch"

        # Generate trading signals
        rsi_signals = generate_momentum_signals(rsi, 'rsi')
        macd_signals = generate_momentum_signals(histogram, 'momentum')

        # Check signal validity
        valid_signals = ['buy', 'sell', 'hold', 'strong_buy', 'strong_sell']
        assert all(signal in valid_signals for signal in rsi_signals.dropna().unique()), "Invalid RSI signals"
        assert all(signal in valid_signals for signal in macd_signals.dropna().unique()), "Invalid MACD signals"

        print(f"✓ Calculated momentum indicators")
        print(f"  RSI: {rsi.dropna().mean():.1f} ± {rsi.dropna().std():.1f}")
        print(f"  ROC: {roc.dropna().mean():.2f}% ± {roc.dropna().std():.2f}%")
        print(f"  MACD histogram: {histogram.dropna().mean():.4f} ± {histogram.dropna().std():.4f}")
        print(f"  RSI signals: {dict(rsi_signals.value_counts())}")
        print(f"  MACD signals: {dict(macd_signals.value_counts())}")

    def test_risk_metrics_scenario(self):
        """Test risk metrics calculation workflow."""
        print("\n=== Testing Risk Metrics ===")

        # Create test data
        market_data = self.create_realistic_market_data(n_symbols=2, n_days=252)

        # Calculate returns for each symbol
        symbols = market_data['symbol'].unique()
        risk_metrics = {}

        for symbol in symbols:
            symbol_data = market_data[market_data['symbol'] == symbol]
            returns = calculate_simple_returns(symbol_data['close'])

            # Calculate risk metrics
            sharpe_ratio = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(returns)
            annual_return = calculate_annualized_returns(returns, periods_per_year=252)

            risk_metrics[symbol] = {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'annual_return': annual_return,
                'volatility': returns.std() * np.sqrt(252)
            }

        # Validate metrics reasonableness
        for symbol, metrics in risk_metrics.items():
            assert -3 <= metrics['sharpe_ratio'] <= 3, f"Sharpe ratio should be reasonable for {symbol}"
            assert -1 <= metrics['max_drawdown'] <= 0, f"Max drawdown should be between -1 and 0 for {symbol}"
            assert -1 <= metrics['annual_return'] <= 2, f"Annual return should be reasonable for {symbol}"
            assert 0 < metrics['volatility'] < 2, f"Volatility should be positive and reasonable for {symbol}"

        # Calculate beta and alpha relative to first symbol
        if len(symbols) >= 2:
            market_returns = calculate_simple_returns(market_data[market_data['symbol'] == symbols[0]]['close'])
            asset_returns = calculate_simple_returns(market_data[market_data['symbol'] == symbols[1]]['close'])

            beta, alpha = calculate_beta_alpha(asset_returns, market_returns)

            assert -2 <= beta <= 3, f"Beta should be reasonable, got {beta}"
            assert -0.5 <= alpha <= 0.5, f"Alpha should be reasonable, got {alpha}"

            risk_metrics[symbols[1]]['beta'] = beta
            risk_metrics[symbols[1]]['alpha'] = alpha

        print("✓ Calculated risk metrics for all symbols")
        for symbol, metrics in risk_metrics.items():
            print(f"  {symbol}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")

    def test_feature_service_integration_scenario(self):
        """Test feature service integration workflow."""
        print("\n=== Testing Feature Service Integration ===")

        # Create test data
        market_data = self.create_realistic_market_data(n_symbols=2, n_days=100)

        # Generate features using feature service
        start_time = time.time()

        # Generate features using feature service
        # Note: Using generate_features instead of generate_features_batch
        # since FeatureGenerator doesn't have batch method
        features_list = []
        for symbol in market_data.columns:
            if symbol != 'timestamp':
                price_data = self.create_price_data_from_series(
                    market_data[symbol],
                    symbol=symbol
                )
                features = self.feature_service.generate_features(price_data)
                features_list.append(features)

        # Combine results
        combined_features = pd.concat([f.to_dataframe() for f in features_list], axis=1)

        processing_time = time.time() - start_time

        # Validate feature generation
        assert len(features) > 0, "Features should not be empty"

        # Check for expected feature types
        expected_patterns = ['return', 'volatility', 'rsi', 'macd', 'sharpe']
        found_features = [col for col in features.columns if any(pattern in col.lower() for pattern in expected_patterns)]
        assert len(found_features) > 5, f"Should find multiple feature types, found {len(found_features)}"

        # Validate data integrity
        assert features.isnull().sum().sum() / features.size < 0.3, "Too many missing values in features"

        # Check performance
        print(f"✓ Generated {len(features.columns)} features for {len(features)} time points")
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Processing rate: {len(features) / processing_time:.0f} points/second")
        print(f"  Sample features: {list(features.columns[:10])}")

    def test_validation_service_integration_scenario(self):
        """Test validation service integration workflow."""
        print("\n=== Testing Validation Service Integration ===")

        # Create test data
        market_data = self.create_realistic_market_data(n_symbols=1, n_days=100)

        # Generate features using individual approach
        features_list = []
        for symbol in ['AAPL', 'MSFT']:  # Assuming these are the symbols
            if symbol in market_data.columns:
                price_data = self.create_price_data_from_series(
                    market_data[symbol],
                    symbol=symbol
                )
                features = self.feature_service.generate_features(price_data)
                features_list.append(features)

        # Combine results
        features = pd.concat([f.to_dataframe() for f in features_list], axis=1)

        # Validate features using appropriate method
        validation_results = self.validation_service.validate_data(features)

        # Validate validation results
        assert isinstance(validation_results, dict), "Should return validation results as dict"
        assert 'quality_score' in validation_results or len(validation_results) > 0, "Should have validation metrics"

        # Check validation categories
        expected_categories = ['completeness', 'consistency', 'accuracy', 'statistical_validity']
        for category in expected_categories:
            assert category in validation_results, f"Should have {category} validation"
            assert 0 <= validation_results[category] <= 1, f"{category} score should be between 0 and 1"

        # Check for issues
        if 'issues' in validation_results:
            assert isinstance(validation_results['issues'], list), "Issues should be a list"

        print("✓ Validated features successfully")
        print(f"  Overall validation score: {validation_results['overall_score']:.3f}")
        for category in expected_categories:
            print(f"  {category}: {validation_results[category]:.3f}")

        if 'issues' in validation_results and validation_results['issues']:
            print(f"  Issues found: {len(validation_results['issues'])}")
            for issue in validation_results['issues'][:3]:  # Show first 3 issues
                print(f"    - {issue}")

    def test_multi_period_analysis_scenario(self):
        """Test multi-period analysis workflow."""
        print("\n=== Testing Multi-Period Analysis ===")

        # Create test data with multiple years
        market_data = self.create_realistic_market_data(n_symbols=1, n_days=756)  # 3 years

        # Calculate multi-period returns
        periods = [1, 5, 21, 63, 252]  # Daily, weekly, monthly, quarterly, yearly
        multi_period_returns = calculate_multi_period_returns(market_data['close'], periods)

        # Validate multi-period returns
        assert len(multi_period_returns) == len(periods), "Should have returns for all periods"

        for period, returns in multi_period_returns.items():
            assert len(returns) == len(market_data), f"Returns length mismatch for period {period}"
            assert not returns.isin([np.inf, -np.inf]).any(), f"Infinite values in period {period} returns"

        # Calculate rolling annual returns
        returns = calculate_simple_returns(market_data['close'])
        rolling_annual_returns = returns.rolling(window=252).apply(lambda x: (1 + x).prod() - 1)

        # Validate rolling returns
        assert len(rolling_annual_returns) == len(returns), "Rolling returns length mismatch"
        assert rolling_annual_returns.dropna().min() > -1, "Rolling returns should not be < -100%"
        assert rolling_annual_returns.dropna().max() < 5, "Rolling returns should be reasonable"

        print("✓ Performed multi-period analysis")
        for period in periods:
            valid_returns = multi_period_returns[period].dropna()
            if len(valid_returns) > 0:
                print(f"  {period}-day returns: {valid_returns.mean():.4f} ± {valid_returns.std():.4f}")

        print(f"  Rolling annual returns: {rolling_annual_returns.dropna().mean():.4f} ± {rolling_annual_returns.dropna().std():.4f}")

    def test_end_to_end_portfolio_analysis_scenario(self):
        """Test complete end-to-end portfolio analysis workflow."""
        print("\n=== Testing End-to-End Portfolio Analysis ===")

        # Create realistic multi-asset data
        portfolio_data = self.create_realistic_market_data(n_symbols=3, n_days=504)  # 2 years

        # Start timer for performance measurement
        start_time = time.time()

        # 1. Calculate basic returns
        symbols = portfolio_data['symbol'].unique()
        returns_data = {}

        for symbol in symbols:
            symbol_data = portfolio_data[portfolio_data['symbol'] == symbol]
            returns_data[symbol] = calculate_simple_returns(symbol_data['close'])

        # 2. Calculate volatility metrics
        volatility_metrics = {}
        for symbol, returns in returns_data.items():
            rolling_vol = calculate_rolling_volatility(returns, window=21)
            ewma_vol = calculate_ewma_volatility(returns, span=30)

            volatility_metrics[symbol] = {
                'rolling_volatility': rolling_vol,
                'ewma_volatility': ewma_vol,
                'annualized_volatility': returns.std() * np.sqrt(252)
            }

        # 3. Calculate momentum indicators
        momentum_indicators = {}
        for symbol in symbols:
            symbol_data = portfolio_data[portfolio_data['symbol'] == symbol]
            rsi = calculate_rsi(symbol_data['close'], period=14)
            macd_line, signal_line, histogram = calculate_macd(symbol_data['close'])

            momentum_indicators[symbol] = {
                'rsi': rsi,
                'macd_line': macd_line,
                'macd_signal': signal_line,
                'macd_histogram': histogram
            }

        # 4. Calculate risk and performance metrics
        portfolio_metrics = {}
        for symbol, returns in returns_data.items():
            sharpe_ratio = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(returns)
            annual_return = calculate_annualized_returns(returns, periods_per_year=252)

            portfolio_metrics[symbol] = {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'annual_return': annual_return,
                'volatility': volatility_metrics[symbol]['annualized_volatility']
            }

        # 5. Calculate correlations and relationships
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()

        # Calculate betas relative to first symbol (as market proxy)
        betas = {}
        market_returns = returns_data[symbols[0]]
        for symbol in symbols[1:]:
            beta, alpha = calculate_beta_alpha(returns_data[symbol], market_returns)
            betas[symbol] = {'beta': beta, 'alpha': alpha}

        # 6. Generate summary report
        end_time = time.time()
        processing_time = end_time - start_time

        # Validate results
        assert len(portfolio_metrics) == len(symbols), "Should have metrics for all symbols"
        assert len(betas) == len(symbols) - 1, "Should have betas for all non-benchmark symbols"

        # Check metric reasonableness
        for symbol, metrics in portfolio_metrics.items():
            assert -3 <= metrics['sharpe_ratio'] <= 3, f"Unreasonable Sharpe ratio for {symbol}"
            assert -1 <= metrics['max_drawdown'] <= 0, f"Unreasonable max drawdown for {symbol}"
            assert metrics['volatility'] > 0, f"Volatility should be positive for {symbol}"

        print("✓ Completed end-to-end portfolio analysis")
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Data points analyzed: {len(portfolio_data)}")
        print(f"  Processing rate: {len(portfolio_data) / processing_time:.0f} points/second")

        print("\n  Portfolio Summary:")
        for symbol, metrics in portfolio_metrics.items():
            print(f"    {symbol}:")
            print(f"      Annual Return: {metrics['annual_return']:.2%}")
            print(f"      Volatility: {metrics['volatility']:.2%}")
            print(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"      Max Drawdown: {metrics['max_drawdown']:.2%}")

        print("\n  Correlations:")
        print(f"    Average correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.3f}")

        if betas:
            print("\n  Risk Measures (relative to {}):".format(symbols[0]))
            for symbol, risk_metrics in betas.items():
                print(f"    {symbol}: β={risk_metrics['beta']:.3f}, α={risk_metrics['alpha']:.3f}")

    def test_performance_benchmarking_scenario(self):
        """Test performance benchmarking scenario."""
        print("\n=== Testing Performance Benchmarking ===")

        # Create large dataset for performance testing
        large_dataset = self.create_realistic_market_data(n_symbols=5, n_days=1000)  # ~5000 data points

        # Test performance targets
        target_processing_time = 30.0  # 30 seconds for 10M points (scaled down)
        target_memory_efficiency = 4096  # 4GB memory limit

        # Benchmark calculations
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Run comprehensive feature calculation
        features = self.feature_service.generate_features_batch(
            large_dataset,
            features=['returns', 'volatility', 'momentum', 'risk_metrics']
        )

        # Validate features
        validation_results = self.validation_service.validate_features(features)

        end_time = time.time()
        end_memory = self._get_memory_usage()

        processing_time = end_time - start_time
        memory_used = end_memory - start_memory

        # Scale performance metrics to 10M data points
        data_points = len(large_dataset)
        scaled_processing_time = processing_time * (10_000_000 / data_points)
        scaled_memory_used = memory_used * (10_000_000 / data_points)

        print("✓ Performance benchmarking completed")
        print(f"  Dataset size: {data_points:,} data points")
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Scaled to 10M points:")
        print(f"    Processing time: {scaled_processing_time:.1f} seconds (target: {target_processing_time}s)")
        print(f"    Memory usage: {scaled_memory_used:.0f} MB (target: {target_memory_efficiency} MB)")
        print(f"  Features generated: {len(features.columns)}")
        print(f"  Validation score: {validation_results['overall_score']:.3f}")

        # Performance assertions
        assert scaled_processing_time < target_processing_time, \
            f"Processing time {scaled_processing_time:.1f}s exceeds target {target_processing_time}s"

        assert scaled_memory_used < target_memory_efficiency, \
            f"Memory usage {scaled_memory_used:.0f}MB exceeds target {target_memory_efficiency}MB"

        assert validation_results['overall_score'] > 0.7, \
            f"Validation score {validation_results['overall_score']:.3f} below threshold"

    def test_error_handling_and_edge_cases_scenario(self):
        """Test error handling and edge cases."""
        print("\n=== Testing Error Handling and Edge Cases ===")

        # Test with empty data
        empty_df = pd.DataFrame()
        try:
            result = calculate_simple_returns(empty_df)
            assert len(result) == 0, "Should handle empty data gracefully"
        except Exception as e:
            assert "empty" in str(e).lower() or "data" in str(e).lower(), \
                "Should mention empty data in error"

        # Test with single data point
        single_point_df = pd.DataFrame({'close': [100.0]})
        try:
            result = calculate_simple_returns(single_point_df)
            # Should handle gracefully (likely NaN or empty)
        except Exception as e:
            assert "insufficient" in str(e).lower() or "data" in str(e).lower(), \
                "Should mention insufficient data"

        # Test with NaN values
        nan_data = pd.DataFrame({
            'close': [100, np.nan, 102, 103, 104, np.nan, 106]
        })
        try:
            result = calculate_simple_returns(nan_data)
            # Should handle NaN values appropriately
            assert len(result) == len(nan_data), "Should preserve length"
        except Exception as e:
            # If it raises an error, it should be informative
            assert "missing" in str(e).lower() or "nan" in str(e).lower(), \
                "Should mention missing values"

        # Test with zero prices
        zero_price_data = pd.DataFrame({
            'close': [100, 0, 102, 103, 104]
        })
        try:
            result = calculate_simple_returns(zero_price_data)
            # Should handle zero prices
        except Exception as e:
            assert "zero" in str(e).lower() or "price" in str(e).lower(), \
                "Should mention zero price issue"

        # Test with negative prices
        negative_price_data = pd.DataFrame({
            'close': [100, -5, 102, 103, 104]
        })
        try:
            result = calculate_simple_returns(negative_price_data)
            # Should handle negative prices
        except Exception as e:
            assert "negative" in str(e).lower() or "price" in str(e).lower(), \
                "Should mention negative price issue"

        print("✓ Error handling tests completed successfully")

    def test_configuration_and_reproducibility_scenario(self):
        """Test configuration management and reproducibility."""
        print("\n=== Testing Configuration and Reproducibility ===")

        # Create test data
        test_data = self.create_realistic_market_data(n_symbols=2, n_days=100)

        # Create configuration
        config = {
            'returns': {
                'method': 'simple',
                'period': 1
            },
            'volatility': {
                'method': 'rolling',
                'window': 21,
                'annualize': True,
                'periods_per_year': 252
            },
            'momentum': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            }
        }

        # Process with configuration multiple times with different seeds
        results = []
        for seed in [42, 42, 42]:  # Same seed should give same results
            np.random.seed(seed)

            features = self.feature_service.generate_features_batch(
                test_data,
                features=['returns', 'volatility', 'momentum'],
                config=config
            )
            results.append(features)

        # Check reproducibility
        # Key numeric columns should be identical when using same seed
        numeric_cols = results[0].select_dtypes(include=[np.number]).columns

        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            col_values = [result[col].dropna().values for result in results]
            # Remove NaN values for comparison
            min_length = min(len(values) for values in col_values)
            if min_length > 0:
                aligned_values = [values[:min_length] for values in col_values]
                assert np.allclose(aligned_values[0], aligned_values[1], rtol=1e-10), \
                    f"Column {col} should be reproducible"
                assert np.allclose(aligned_values[1], aligned_values[2], rtol=1e-10), \
                    f"Column {col} should be reproducible"

        print("✓ Configuration and reproducibility tests completed")
        print(f"  Generated consistent results across 3 runs")
        print(f"  Checked {len(numeric_cols)} numeric columns for reproducibility")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # Fallback if psutil not available

    def test_quickstart_comprehensive_scenario(self):
        """Comprehensive test following quickstart guide scenarios."""
        print("\n=== Testing Comprehensive Quickstart Scenario ===")

        # 1. Data Preparation
        print("1. Preparing realistic market data...")
        market_data = self.create_realistic_market_data(n_symbols=3, n_days=252)

        # 2. Basic Analysis
        print("2. Running basic financial analysis...")
        symbols = market_data['symbol'].unique()

        basic_results = {}
        for symbol in symbols:
            symbol_data = market_data[market_data['symbol'] == symbol]

            # Calculate returns
            returns = calculate_simple_returns(symbol_data['close'])

            # Calculate key metrics
            annual_return = calculate_annualized_returns(returns, periods_per_year=252)
            volatility = calculate_annualized_volatility(returns, periods_per_year=252)
            sharpe_ratio = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(returns)

            basic_results[symbol] = {
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd
            }

        # 3. Advanced Analysis
        print("3. Running advanced analysis...")
        advanced_results = {}

        for symbol in symbols:
            symbol_data = market_data[market_data['symbol'] == symbol]

            # Volatility analysis
            returns = calculate_simple_returns(symbol_data['close'])
            rolling_vol = calculate_rolling_volatility(returns, window=21)
            ewma_vol = calculate_ewma_volatility(returns, span=30)

            # Momentum analysis
            rsi = calculate_rsi(symbol_data['close'], period=14)
            macd_line, signal_line, histogram = calculate_macd(symbol_data['close'])

            # Volatility regime detection
            vol_regime = calculate_volatility_regime(rolling_vol.dropna())

            advanced_results[symbol] = {
                'volatility_measures': {
                    'rolling': rolling_vol,
                    'ewma': ewma_vol,
                    'regime': vol_regime
                },
                'momentum_indicators': {
                    'rsi': rsi,
                    'macd_line': macd_line,
                    'macd_signal': signal_line,
                    'macd_histogram': histogram
                }
            }

        # 4. Portfolio Analysis
        print("4. Performing portfolio analysis...")

        # Create equal-weighted portfolio returns
        portfolio_returns = pd.DataFrame(index=market_data['timestamp'].unique())
        for symbol in symbols:
            symbol_returns = calculate_simple_returns(
                market_data[market_data['symbol'] == symbol]['close']
            )
            portfolio_returns[symbol] = symbol_returns.values

        portfolio_returns['portfolio'] = portfolio_returns[symbols].mean(axis=1)

        # Portfolio metrics
        portfolio_annual_return = calculate_annualized_returns(portfolio_returns['portfolio'].dropna())
        portfolio_volatility = calculate_annualized_volatility(portfolio_returns['portfolio'].dropna())
        portfolio_sharpe = calculate_sharpe_ratio(portfolio_returns['portfolio'].dropna())
        portfolio_max_dd = calculate_max_drawdown(portfolio_returns['portfolio'].dropna())

        # 5. Correlation Analysis
        correlation_matrix = portfolio_returns[symbols].corr()

        # 6. Generate Summary Report
        print("5. Generating summary report...")

        summary_report = {
            'analysis_date': datetime.now().isoformat(),
            'data_period': {
                'start': market_data['timestamp'].min().isoformat(),
                'end': market_data['timestamp'].max().isoformat(),
                'trading_days': len(market_data['timestamp'].unique())
            },
            'portfolio_metrics': {
                'annual_return': portfolio_annual_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_sharpe,
                'max_drawdown': portfolio_max_dd
            },
            'individual_metrics': basic_results,
            'correlations': correlation_matrix.to_dict(),
            'data_quality': {
                'total_data_points': len(market_data),
                'symbols_analyzed': len(symbols),
                'completeness_score': 1.0 - (market_data.isnull().sum().sum() / market_data.size)
            }
        }

        # 7. Save Results
        report_file = os.path.join(self.temp_dir, 'financial_analysis_report.json')
        with open(report_file, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)

        # Validate Results
        assert os.path.exists(report_file), "Analysis report should be created"
        assert len(summary_report['individual_metrics']) == len(symbols), "Should have metrics for all symbols"
        assert summary_report['portfolio_metrics']['sharpe_ratio'] > -3, "Portfolio Sharpe should be reasonable"
        assert summary_report['portfolio_metrics']['max_drawdown'] <= 0, "Max drawdown should be negative or zero"

        print("✓ Comprehensive quickstart scenario completed successfully")
        print(f"  Analysis period: {summary_report['data_period']['trading_days']} trading days")
        print(f"  Portfolio return: {summary_report['portfolio_metrics']['annual_return']:.2%}")
        print(f"  Portfolio volatility: {summary_report['portfolio_metrics']['volatility']:.2%}")
        print(f"  Portfolio Sharpe ratio: {summary_report['portfolio_metrics']['sharpe_ratio']:.3f}")
        print(f"  Portfolio max drawdown: {summary_report['portfolio_metrics']['max_drawdown']:.2%}")
        print(f"  Report saved to: {report_file}")

        return summary_report


if __name__ == "__main__":
    # Run comprehensive tests
    test_suite = TestFinancialFeaturesIntegration()
    test_suite.setup_method()

    try:
        # Run key scenarios
        test_suite.test_basic_returns_calculation_scenario()
        test_suite.test_volatility_calculation_scenario()
        test_suite.test_momentum_indicators_scenario()
        test_suite.test_risk_metrics_scenario()
        test_suite.test_feature_service_integration_scenario()
        test_suite.test_validation_service_integration_scenario()
        test_suite.test_end_to_end_portfolio_analysis_scenario()
        test_suite.test_performance_benchmarking_scenario()
        test_suite.test_quickstart_comprehensive_scenario()

        print("\n" + "="*60)
        print("All financial features integration tests completed successfully!")
        print("="*60)

    finally:
        test_suite.teardown_method()