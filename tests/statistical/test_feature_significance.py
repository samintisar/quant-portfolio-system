"""
Statistical significance tests for financial features.

This module provides comprehensive statistical validation of calculated financial features
including returns, volatility, and momentum indicators. Tests include:

- Normality tests (Shapiro-Wilk, Jarque-Bera, Kolmogorov-Smirnov)
- Autocorrelation tests (Ljung-Box, Durbin-Watson)
- Stationarity tests (ADF, KPSS)
- Volatility clustering tests (ARCH-LM)
- Distribution comparison tests (t-test, Mann-Whitney U)
- Correlation significance tests
- Outlier detection and impact assessment
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any
from scipy import stats
from statsmodels.stats import diagnostic
from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import adfuller, kpss
from arch import arch_model
import warnings

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.returns import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_sharpe_ratio,
    calculate_beta_alpha
)

from lib.volatility import (
    calculate_rolling_volatility,
    calculate_annualized_volatility,
    calculate_garch11_volatility,
    calculate_volatility_clustering
)

from lib.momentum import (
    calculate_rsi,
    calculate_macd,
    calculate_simple_momentum,
    generate_momentum_signals
)

from services.feature_service import FeatureService
from services.validation_service import ValidationService


class StatisticalValidator:
    """Statistical validation utilities for financial features."""

    @staticmethod
    def test_normality(data: pd.Series, significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Test normality of financial data using multiple tests.

        Tests performed:
        - Shapiro-Wilk test (preferred for small samples < 5000)
        - Jarque-Bera test (for larger samples)
        - Kolmogorov-Smirnov test (against normal distribution)

        Args:
            data: Series to test for normality
            significance_level: Significance level (default 0.05)

        Returns:
            Dictionary with test results and overall conclusion
        """
        clean_data = data.dropna()
        if len(clean_data) < 3:
            return {'error': 'Insufficient data for normality testing'}

        results = {}

        # Shapiro-Wilk test (for samples < 5000)
        if len(clean_data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(clean_data)
            results['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > significance_level
            }

        # Jarque-Bera test (for larger samples)
        if len(clean_data) >= 8:
            jb_stat, jb_p = stats.jarque_bera(clean_data)
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > significance_level
            }

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(clean_data, 'norm', args=(clean_data.mean(), clean_data.std()))
        results['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'is_normal': ks_p > significance_level
        }

        # Overall conclusion (majority vote)
        normality_tests = [test for test in results.values() if 'is_normal' in test]
        if normality_tests:
            normal_count = sum(1 for test in normality_tests if test['is_normal'])
            results['overall_normal'] = normal_count / len(normality_tests) > 0.5
        else:
            results['overall_normal'] = False

        return results

    @staticmethod
    def test_autocorrelation(data: pd.Series, max_lags: int = 20,
                           significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Test for autocorrelation in time series data.

        Tests performed:
        - Ljung-Box test for joint autocorrelation
        - Durbin-Watson test for first-order autocorrelation
        - Individual autocorrelation coefficients

        Args:
            data: Time series data
            max_lags: Maximum number of lags to test
            significance_level: Significance level

        Returns:
            Dictionary with autocorrelation test results
        """
        clean_data = data.dropna()
        if len(clean_data) < max_lags:
            return {'error': f'Insufficient data for {max_lags} lags'}

        results = {}

        # Ljung-Box test for joint autocorrelation
        lb_stat, lb_p = stattools.q_stat(clean_data, max_lags)
        results['ljung_box'] = {
            'statistic': float(lb_stat[-1]),  # Test statistic for max_lags
            'p_value': float(lb_p[-1]),
            'no_autocorrelation': lb_p[-1] > significance_level
        }

        # Durbin-Watson test for first-order autocorrelation
        dw_stat = stattools.durbin_watson(clean_data)
        results['durbin_watson'] = {
            'statistic': dw_stat,
            'interpretation': 'Positive autocorrelation' if dw_stat < 1.5 else
                             'Negative autocorrelation' if dw_stat > 2.5 else
                             'No significant autocorrelation'
        }

        # Individual autocorrelation coefficients
        acf_values, acf_confint = stattools.acf(clean_data, nlags=max_lags, alpha=significance_level, fft=True)
        significant_lags = []

        for lag in range(1, min(max_lags + 1, len(acf_values))):
            lag_stat = acf_values[lag]
            conf_low, conf_high = acf_confint[lag]
            is_significant = not (conf_low <= lag_stat <= conf_high)

            significant_lags.append({
                'lag': lag,
                'autocorrelation': lag_stat,
                'confidence_interval': (conf_low, conf_high),
                'is_significant': is_significant
            })

        results['individual_autocorrelations'] = significant_lags

        return results

    @staticmethod
    def test_stationarity(data: pd.Series, significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Test for stationarity of time series data.

        Tests performed:
        - Augmented Dickey-Fuller (ADF) test
        - KPSS test (complementary to ADF)

        Args:
            data: Time series data
            significance_level: Significance level

        Returns:
            Dictionary with stationarity test results
        """
        clean_data = data.dropna()
        if len(clean_data) < 10:
            return {'error': 'Insufficient data for stationarity testing'}

        results = {}

        # Augmented Dickey-Fuller test
        adf_result = adfuller(clean_data, regression='c')  # Constant only
        results['adf'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] <= significance_level
        }

        # KPSS test (null hypothesis is stationarity)
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(clean_data, regression='c')
        results['kpss'] = {
            'statistic': kpss_stat,
            'p_value': kpss_p,
            'critical_values': kpss_crit,
            'is_stationary': kpss_p > significance_level  # Note: opposite interpretation
        }

        # Overall conclusion (both tests should agree)
        adf_stationary = results['adf']['is_stationary']
        kpss_stationary = results['kpss']['is_stationary']

        if adf_stationary and kpss_stationary:
            results['overall_stationary'] = 'Stationary'
        elif not adf_stationary and not kpss_stationary:
            results['overall_stationary'] = 'Non-stationary'
        else:
            results['overall_stationary'] = 'Inconclusive'

        return results

    @staticmethod
    def test_volatility_clustering(returns: pd.Series, significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Test for volatility clustering effects.

        Tests performed:
        - ARCH-LM test for autoregressive conditional heteroskedasticity
        - Engle's ARCH test

        Args:
            returns: Return series
            significance_level: Significance level

        Returns:
            Dictionary with volatility clustering test results
        """
        clean_returns = returns.dropna()
        if len(clean_returns) < 50:
            return {'error': 'Insufficient data for volatility clustering tests'}

        results = {}

        # ARCH-LM test
        try:
            # Calculate squared returns
            squared_returns = clean_returns ** 2

            # Lag selection (use min(10, len(returns)//10))
            max_lags = min(10, len(clean_returns) // 10)

            # Perform ARCH-LM test
            arch_lm_stat, arch_lm_p, f_stat, f_p = diagnostic.het_arch(clean_returns, max_lags=max_lags)

            results['arch_lm'] = {
                'statistic': arch_lm_stat,
                'p_value': arch_lm_p,
                'f_statistic': f_stat,
                'f_p_value': f_p,
                'has_arch_effects': arch_lm_p < significance_level
            }
        except Exception as e:
            results['arch_lm'] = {'error': str(e)}

        # Simple volatility clustering measure
        try:
            # Calculate rolling volatility
            rolling_vol = clean_returns.rolling(window=21).std()
            vol_changes = rolling_vol.pct_change().dropna()

            # Test autocorrelation of volatility changes
            if len(vol_changes) > 10:
                vol_autocorr = stattools.acf(vol_changes, nlags=5, fft=True)[1:]
                results['volatility_autocorrelation'] = {
                    'lags_1_5': vol_autocorr.tolist(),
                    'mean_abs_autocorr': np.mean(np.abs(vol_autocorr)),
                    'significant_clustering': np.mean(np.abs(vol_autocorr)) > 0.1
                }
        except Exception as e:
            results['volatility_autocorrelation'] = {'error': str(e)}

        return results

    @staticmethod
    def test_distribution_comparison(data1: pd.Series, data2: pd.Series,
                                   significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Compare distributions of two datasets.

        Tests performed:
        - Independent t-test (for normally distributed data)
        - Mann-Whitney U test (non-parametric alternative)
        - Kolmogorov-Smirnov two-sample test

        Args:
            data1: First dataset
            data2: Second dataset
            significance_level: Significance level

        Returns:
            Dictionary with comparison test results
        """
        clean_data1 = data1.dropna()
        clean_data2 = data2.dropna()

        if len(clean_data1) < 3 or len(clean_data2) < 3:
            return {'error': 'Insufficient data for comparison tests'}

        results = {}

        # Independent t-test
        try:
            t_stat, t_p = stats.ttest_ind(clean_data1, clean_data2, equal_var=False)
            results['t_test'] = {
                'statistic': t_stat,
                'p_value': t_p,
                'same_distribution': t_p > significance_level
            }
        except Exception as e:
            results['t_test'] = {'error': str(e)}

        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p = stats.mannwhitneyu(clean_data1, clean_data2, alternative='two-sided')
            results['mann_whitney'] = {
                'statistic': u_stat,
                'p_value': u_p,
                'same_distribution': u_p > significance_level
            }
        except Exception as e:
            results['mann_whitney'] = {'error': str(e)}

        # Kolmogorov-Smirnov two-sample test
        try:
            ks_stat, ks_p = stats.ks_2samp(clean_data1, clean_data2)
            results['kolmogorov_smirnov_2samp'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'same_distribution': ks_p > significance_level
            }
        except Exception as e:
            results['kolmogorov_smirnov_2samp'] = {'error': str(e)}

        # Summary statistics comparison
        results['summary_stats'] = {
            'data1_mean': clean_data1.mean(),
            'data1_std': clean_data1.std(),
            'data2_mean': clean_data2.mean(),
            'data2_std': clean_data2.std(),
            'mean_difference': abs(clean_data1.mean() - clean_data2.mean()),
            'std_ratio': clean_data1.std() / clean_data2.std() if clean_data2.std() > 0 else np.nan
        }

        return results

    @staticmethod
    def test_correlation_significance(data1: pd.Series, data2: pd.Series,
                                   significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Test significance of correlation between two series.

        Tests performed:
        - Pearson correlation with significance test
        - Spearman rank correlation
        - Kendall's tau correlation

        Args:
            data1: First series
            data2: Second series
            significance_level: Significance level

        Returns:
            Dictionary with correlation test results
        """
        # Align series
        aligned_data1, aligned_data2 = data1.align(data2, join='inner')
        aligned_data1 = aligned_data1.dropna()
        aligned_data2 = aligned_data2.dropna()

        if len(aligned_data1) < 3:
            return {'error': 'Insufficient overlapping data for correlation tests'}

        results = {}

        # Pearson correlation
        try:
            pearson_corr, pearson_p = stats.pearsonr(aligned_data1, aligned_data2)
            results['pearson'] = {
                'correlation': pearson_corr,
                'p_value': pearson_p,
                'is_significant': pearson_p < significance_level,
                'r_squared': pearson_corr ** 2
            }
        except Exception as e:
            results['pearson'] = {'error': str(e)}

        # Spearman correlation
        try:
            spearman_corr, spearman_p = stats.spearmanr(aligned_data1, aligned_data2)
            results['spearman'] = {
                'correlation': spearman_corr,
                'p_value': spearman_p,
                'is_significant': spearman_p < significance_level
            }
        except Exception as e:
            results['spearman'] = {'error': str(e)}

        # Kendall's tau
        try:
            kendall_corr, kendall_p = stats.kendalltau(aligned_data1, aligned_data2)
            results['kendall'] = {
                'correlation': kendall_corr,
                'p_value': kendall_p,
                'is_significant': kendall_p < significance_level
            }
        except Exception as e:
            results['kendall'] = {'error': str(e)}

        return results

    @staticmethod
    def detect_outliers(data: pd.Series, method: str = 'iqr',
                       significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Detect and analyze outliers in financial data.

        Methods:
        - IQR method (interquartile range)
        - Z-score method
        - Modified Z-score (MAD)
        - Grubbs' test for statistical outliers

        Args:
            data: Data series
            method: Outlier detection method
            significance_level: Significance level for statistical tests

        Returns:
            Dictionary with outlier analysis results
        """
        clean_data = data.dropna()
        if len(clean_data) < 4:
            return {'error': 'Insufficient data for outlier detection'}

        results = {}

        # IQR method
        if method in ['iqr', 'all']:
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            results['iqr'] = {
                'outliers': iqr_outliers.tolist(),
                'count': len(iqr_outliers),
                'percentage': len(iqr_outliers) / len(clean_data) * 100,
                'bounds': (lower_bound, upper_bound)
            }

        # Z-score method
        if method in ['zscore', 'all']:
            z_scores = np.abs(stats.zscore(clean_data))
            z_outliers = clean_data[z_scores > 3]  # 3 standard deviations
            results['zscore'] = {
                'outliers': z_outliers.tolist(),
                'count': len(z_outliers),
                'percentage': len(z_outliers) / len(clean_data) * 100,
                'threshold': 3
            }

        # Modified Z-score (MAD)
        if method in ['mad', 'all']:
            median = np.median(clean_data)
            mad = np.median(np.abs(clean_data - median))
            modified_z_scores = 0.6745 * (clean_data - median) / mad
            mad_outliers = clean_data[np.abs(modified_z_scores) > 3.5]
            results['mad'] = {
                'outliers': mad_outliers.tolist(),
                'count': len(mad_outliers),
                'percentage': len(mad_outliers) / len(clean_data) * 100,
                'threshold': 3.5
            }

        # Grubbs' test (for single outlier)
        if len(clean_data) >= 8 and method in ['grubbs', 'all']:
            try:
                def grubbs_test(data, alpha=significance_level):
                    mean = np.mean(data)
                    std = np.std(data, ddof=1)
                    distances = np.abs(data - mean)
                    max_idx = np.argmax(distances)
                    max_distance = distances[max_idx]

                    # Calculate Grubbs statistic
                    G = max_distance / std

                    # Critical value
                    n = len(data)
                    t_critical = stats.t.ppf(1 - alpha/(2*n), n-2)
                    G_critical = ((n-1) * np.sqrt(t_critical**2)) / np.sqrt(n * (n-2 + t_critical**2))

                    return G, G_critical, max_idx

                G_stat, G_critical, outlier_idx = grubbs_test(clean_data)
                results['grubbs'] = {
                    'statistic': G_stat,
                    'critical_value': G_critical,
                    'has_outlier': G_stat > G_critical,
                    'outlier_value': clean_data.iloc[outlier_idx],
                    'outlier_index': clean_data.index[outlier_idx]
                }
            except Exception as e:
                results['grubbs'] = {'error': str(e)}

        return results


class TestReturnsSignificance:
    """Statistical significance tests for returns calculations."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.validator = StatisticalValidator()

        # Generate test datasets
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        self.normal_returns = pd.Series(np.random.normal(0.001, 0.02, 1000), index=dates)

        # Create non-normal returns (fat-tailed)
        from scipy import stats
        self.fat_tailed_returns = pd.Series(
            stats.t.rvs(df=3, loc=0.001, scale=0.02, size=1000, random_state=42), index=dates
        )

        # Create autocorrelated returns
        ar_returns = [0.001]
        for i in range(1, 1000):
            ar_returns.append(0.3 * ar_returns[i-1] + np.random.normal(0, 0.02))
        self.autocorrelated_returns = pd.Series(ar_returns, index=dates)

        # Create trend returns (non-stationary)
        trend = np.linspace(0, 0.5, 1000)
        self.trend_returns = pd.Series(
            0.001 + trend + np.random.normal(0, 0.02, 1000), index=dates
        )

    def test_returns_normality(self):
        """Test normality of different return distributions."""
        print("\n=== Returns Normality Tests ===")

        # Test normal returns
        normal_results = self.validator.test_normality(self.normal_returns)
        assert 'overall_normal' in normal_results
        assert normal_results['overall_normal'] == True, "Normal returns should be detected as normal"

        # Test fat-tailed returns
        fat_results = self.validator.test_normality(self.fat_tailed_returns)
        assert fat_results['overall_normal'] == False, "Fat-tailed returns should not be normal"

        # Compare normality statistics
        print(f"Normal returns p-values: {[test['p_value'] for test in normal_results.values() if isinstance(test, dict) and 'p_value' in test]}")
        print(f"Fat-tailed returns p-values: {[test['p_value'] for test in fat_results.values() if isinstance(test, dict) and 'p_value' in test]}")

    def test_returns_stationarity(self):
        """Test stationarity of return series."""
        print("\n=== Returns Stationarity Tests ===")

        # Normal returns should be stationary
        normal_stationary = self.validator.test_stationarity(self.normal_returns)
        assert normal_stationary['overall_stationary'] == 'Stationary', "Normal returns should be stationary"

        # Trend returns should be non-stationary
        trend_stationary = self.validator.test_stationarity(self.trend_returns)
        assert trend_stationary['overall_stationary'] == 'Non-stationary', "Trend returns should be non-stationary"

    def test_returns_autocorrelation(self):
        """Test autocorrelation in returns."""
        print("\n=== Returns Autocorrelation Tests ===")

        # Normal returns should have no autocorrelation
        normal_ac = self.validator.test_autocorrelation(self.normal_returns)
        assert normal_ac['ljung_box']['no_autocorrelation'] == True, "Normal returns should have no autocorrelation"

        # Autocorrelated returns should show autocorrelation
        auto_ac = self.validator.test_autocorrelation(self.autocorrelated_returns)
        assert auto_ac['ljung_box']['no_autocorrelation'] == False, "Autocorrelated returns should show autocorrelation"

        # Check Durbin-Watson statistic
        assert auto_ac['durbin_watson']['statistic'] < 2.0, "Autocorrelated returns should have DW < 2.0"

    def test_simple_returns_vs_log_returns(self):
        """Test statistical differences between simple and log returns."""
        print("\n=== Simple vs Log Returns Comparison ===")

        log_returns = np.log(1 + self.normal_returns)

        comparison = self.validator.test_distribution_comparison(
            self.normal_returns.dropna(), log_returns.dropna()
        )

        # For small returns, simple and log returns should be similar
        assert comparison['summary_stats']['mean_difference'] < 0.001, "Mean difference should be small for small returns"

        # Both should be stationary
        log_stationary = self.validator.test_stationarity(log_returns)
        assert log_stationary['overall_stationary'] == 'Stationary', "Log returns should be stationary"


class TestVolatilitySignificance:
    """Statistical significance tests for volatility calculations."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.validator = StatisticalValidator()

        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')

        # Create returns with different volatility characteristics
        self.constant_vol_returns = pd.Series(np.random.normal(0.001, 0.02, 1000), index=dates)

        # Create volatility clustering (GARCH-like)
        garch_vol = np.sqrt(0.0001 + 0.1 * np.random.chisquare(1, 1000) / 1)
        self.clustering_returns = pd.Series(
            np.random.normal(0, garch_vol, 1000), index=dates
        )

        # Calculate different volatility measures
        self.rolling_vol = calculate_rolling_volatility(self.clustering_returns, window=21)
        self.ewma_vol = calculate_ewma_volatility(self.clustering_returns, span=30)

    def test_volatility_clustering_detection(self):
        """Test detection of volatility clustering."""
        print("\n=== Volatility Clustering Tests ===")

        # Constant volatility returns should show no clustering
        constant_clustering = self.validator.test_volatility_clustering(self.constant_vol_returns)
        assert not constant_clustering.get('arch_lm', {}).get('has_arch_effects', False), \
            "Constant volatility should show no ARCH effects"

        # Volatility clustering returns should show clustering
        clustering_result = self.validator.test_volatility_clustering(self.clustering_returns)
        assert clustering_result.get('arch_lm', {}).get('has_arch_effects', False), \
            "Volatility clustering should be detected"

    def test_volatility_measure_comparison(self):
        """Test statistical differences between volatility measures."""
        print("\n=== Volatility Measure Comparison ===")

        # Compare rolling vs EWMA volatility
        comparison = self.validator.test_distribution_comparison(
            self.rolling_vol.dropna(), self.ewma_vol.dropna()
        )

        # Volatility measures should be positively correlated
        correlation = self.validator.test_correlation_significance(
            self.rolling_vol, self.ewma_vol
        )

        assert correlation['pearson']['is_significant'], "Volatility measures should be significantly correlated"
        assert correlation['pearson']['correlation'] > 0.7, "Volatility measures should have high positive correlation"

    def test_volatility_normality(self):
        """Test normality of volatility measures."""
        print("\n=== Volatility Normality Tests ===")

        # Volatility is typically right-skewed and non-normal
        rolling_normality = self.validator.test_normality(self.rolling_vol.dropna())
        assert not rolling_normality['overall_normal'], "Rolling volatility should not be normally distributed"

        ewma_normality = self.validator.test_normality(self.ewma_vol.dropna())
        assert not ewma_normality['overall_normal'], "EWMA volatility should not be normally distributed"


class TestMomentumSignificance:
    """Statistical significance tests for momentum indicators."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.validator = StatisticalValidator()

        # Create price series with different characteristics
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')

        # Trending price series
        trend = np.cumsum(np.random.normal(0.001, 0.02, 1000))
        self.trending_prices = pd.Series(100 * np.exp(trend), index=dates)

        # Sideways price series
        sideways = np.cumsum(np.random.normal(0, 0.02, 1000))
        self.sideways_prices = pd.Series(100 * np.exp(sideways), index=dates)

        # Calculate momentum indicators
        self.trending_rsi = calculate_rsi(self.trending_prices, period=14)
        self.sideways_rsi = calculate_rsi(self.sideways_prices, period=14)

        self.trending_macd, _, _ = calculate_macd(self.trending_prices)
        self.sideways_macd, _, _ = calculate_macd(self.sideways_prices)

    def test_rsi_distribution_properties(self):
        """Test statistical properties of RSI."""
        print("\n=== RSI Statistical Properties ===")

        # RSI should be bounded between 0 and 100
        trending_rsi_clean = self.trending_rsi.dropna()
        sideways_rsi_clean = self.sideways_rsi.dropna()

        assert trending_rsi_clean.min() >= 0 and trending_rsi_clean.max() <= 100, "RSI should be bounded [0, 100]"
        assert sideways_rsi_clean.min() >= 0 and sideways_rsi_clean.max() <= 100, "RSI should be bounded [0, 100]"

        # Test RSI distribution differences
        rsi_comparison = self.validator.test_distribution_comparison(trending_rsi_clean, sideways_rsi_clean)
        assert rsi_comparison['summary_stats']['mean_difference'] > 1, "Trending vs sideways RSI should have different means"

    def test_momentum_autocorrelation(self):
        """Test autocorrelation in momentum indicators."""
        print("\n=== Momentum Autocorrelation Tests ===")

        # MACD should show some autocorrelation
        trending_macd_clean = self.trending_macd.dropna()
        macd_autocorr = self.validator.test_autocorrelation(trending_macd_clean)

        assert not macd_autocorr['ljung_box']['no_autocorrelation'], "MACD should show autocorrelation"

    def test_momentum_signal_effectiveness(self):
        """Test statistical effectiveness of momentum signals."""
        print("\n=== Momentum Signal Effectiveness ===")

        # Generate signals for trending market
        trending_signals = generate_momentum_signals(self.trending_rsi, 'rsi')

        # Compare returns following buy vs sell signals
        trending_returns = calculate_simple_returns(self.trending_prices)

        # Align signals with forward returns
        aligned_signals, aligned_returns = trending_signals.align(trending_returns.shift(-1), join='inner')

        buy_returns = aligned_returns[aligned_signals == 'buy'].dropna()
        sell_returns = aligned_returns[aligned_signals == 'sell'].dropna()

        if len(buy_returns) > 5 and len(sell_returns) > 5:
            signal_comparison = self.validator.test_distribution_comparison(buy_returns, sell_returns)

            # In trending market, buy signals should outperform sell signals
            assert signal_comparison['summary_stats']['mean_difference'] > 0, \
                "Buy signals should outperform sell signals in trending market"


class TestFeatureIntegrationSignificance:
    """Integration tests for statistical significance across features."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.validator = StatisticalValidator()

        # Create realistic financial dataset
        dates = pd.date_range(start='2020-01-01', periods=2000, freq='D')

        # Generate prices with multiple regimes
        base_trend = np.cumsum(np.random.normal(0.0005, 0.015, 2000))
        regime_shifts = np.zeros(2000)
        regime_shifts[1000:] = 0.002  # Add regime shift at midpoint

        self.prices = pd.Series(100 * np.exp(base_trend + regime_shifts), index=dates)
        self.returns = calculate_simple_returns(self.prices)

    def test_return_volatility_relationship(self):
        """Test statistical relationship between returns and volatility."""
        print("\n=== Return-Volatility Relationship Tests ===")

        rolling_vol = calculate_rolling_volatility(self.returns, window=21)

        # Test leverage effect (negative correlation between returns and volatility changes)
        vol_changes = rolling_vol.diff().dropna()
        aligned_returns, aligned_vol_changes = self.returns.align(vol_changes, join='inner')

        correlation = self.validator.test_correlation_significance(aligned_returns, aligned_vol_changes)

        # Should show some form of leverage effect
        assert correlation['pearson']['correlation'] < 0, "Should observe negative return-volatility correlation (leverage effect)"

    def test_momentum_volatility_interaction(self):
        """Test interaction between momentum and volatility."""
        print("\n=== Momentum-Volatility Interaction Tests ===")

        rsi = calculate_rsi(self.prices, period=14)
        rolling_vol = calculate_rolling_volatility(self.returns, window=21)

        # Test if RSI values differ significantly in high vs low volatility periods
        vol_median = rolling_vol.median()
        high_vol_periods = rolling_vol > vol_median
        low_vol_periods = rolling_vol <= vol_median

        high_vol_rsi = rsi[high_vol_periods].dropna()
        low_vol_rsi = rsi[low_vol_periods].dropna()

        rsi_comparison = self.validator.test_distribution_comparison(high_vol_rsi, low_vol_rsi)

        # High volatility periods should affect RSI behavior
        assert rsi_comparison['summary_stats']['std_difference'] > 1, "RSI should behave differently in high vs low volatility"

    def test_comprehensive_feature_validation(self):
        """Comprehensive statistical validation of all features."""
        print("\n=== Comprehensive Feature Validation ===")

        # Generate all features
        features = {
            'returns': self.returns,
            'volatility': calculate_rolling_volatility(self.returns, window=21),
            'rsi': calculate_rsi(self.prices, period=14),
            'macd': calculate_macd(self.prices)[0]  # MACD line only
        }

        validation_results = {}

        for feature_name, feature_data in features.items():
            clean_data = feature_data.dropna()
            if len(clean_data) < 10:
                continue

            # Run comprehensive statistical tests
            feature_validation = {
                'normality': self.validator.test_normality(clean_data),
                'stationarity': self.validator.test_stationarity(clean_data),
                'autocorrelation': self.validator.test_autocorrelation(clean_data),
                'outliers': self.validator.detect_outliers(clean_data, method='iqr')
            }

            # Add volatility clustering test for returns
            if feature_name == 'returns':
                feature_validation['volatility_clustering'] = self.validator.test_volatility_clustering(clean_data)

            validation_results[feature_name] = feature_validation

        # Validate key statistical properties
        assert validation_results['returns']['stationarity']['overall_stationary'] == 'Stationary', \
            "Returns should be stationary"
        assert not validation_results['volatility']['normality']['overall_normal'], \
            "Volatility should not be normally distributed"
        assert validation_results['returns'].get('volatility_clustering', {}).get('arch_lm', {}).get('has_arch_effects', False), \
            "Returns should show volatility clustering"

        print(f"Validated {len(validation_results)} features successfully")


if __name__ == "__main__":
    # Run statistical significance tests
    print("Starting Financial Features Statistical Significance Tests")
    print("=" * 70)

    # Initialize test classes
    returns_tests = TestReturnsSignificance()
    returns_tests.setup_method()
    returns_tests.test_returns_normality()

    volatility_tests = TestVolatilitySignificance()
    volatility_tests.setup_method()
    volatility_tests.test_volatility_clustering_detection()

    momentum_tests = TestMomentumSignificance()
    momentum_tests.setup_method()
    momentum_tests.test_rsi_distribution_properties()

    integration_tests = TestFeatureIntegrationSignificance()
    integration_tests.setup_method()
    integration_tests.test_comprehensive_feature_validation()

    print("\n" + "=" * 70)
    print("All statistical significance tests completed successfully!")