"""
Enhanced data ingestion service with extreme event detection.

Implements comprehensive data ingestion capabilities including:
- Extreme event detection and handling
- Data quality monitoring
- Automated data pipeline management
- Multi-source data integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from abc import ABC, abstractmethod

# Import existing data preprocessing components
from data.src.lib.cleaning import DataCleaningLibrary
from data.src.lib.validation import DataValidationLibrary


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    async def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from source."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get data source metadata."""
        pass


class YahooFinanceDataSource(DataSource):
    """Yahoo Finance data source with enhanced capabilities."""

    def __init__(self):
        self.metadata = {
            "name": "Yahoo Finance",
            "update_frequency": "daily",
            "data_types": ["price", "volume", "dividends", "splits"],
            "coverage": "global_equities"
        }

    async def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance with error handling."""
        try:
            # This would use yfinance in actual implementation
            # For now, return mock data structure
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            data = pd.DataFrame({
                'Date': dates,
                'Open': np.random.uniform(100, 200, len(dates)),
                'High': np.random.uniform(100, 200, len(dates)),
                'Low': np.random.uniform(100, 200, len(dates)),
                'Close': np.random.uniform(100, 200, len(dates)),
                'Volume': np.random.uniform(1000000, 10000000, len(dates))
            })
            data.set_index('Date', inplace=True)
            return data
        except Exception as e:
            logging.error(f"Failed to fetch data for {symbol}: {e}")
            raise

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata


class ExtremeEventDetector:
    """Detect extreme market events in price data."""

    def __init__(self, threshold_std: float = 3.0, lookback_window: int = 252):
        self.threshold_std = threshold_std
        self.lookback_window = lookback_window

    def detect_extreme_events(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect extreme events in price data.

        Returns:
            Dictionary containing extreme events and statistics
        """
        events = []

        # Calculate returns
        returns = data['Close'].pct_change().dropna()

        # Calculate rolling statistics
        rolling_mean = returns.rolling(window=self.lookback_window).mean()
        rolling_std = returns.rolling(window=self.lookback_window).std()

        # Detect extreme returns
        extreme_mask = abs(returns) > (self.threshold_std * rolling_std)
        extreme_dates = returns[extreme_mask].index.tolist()

        for date in extreme_dates:
            event = {
                'date': date,
                'return': returns.loc[date],
                'z_score': (returns.loc[date] - rolling_mean.loc[date]) / rolling_std.loc[date],
                'type': 'positive' if returns.loc[date] > 0 else 'negative',
                'magnitude': abs(returns.loc[date])
            }
            events.append(event)

        return {
            'events': events,
            'total_events': len(events),
            'positive_events': len([e for e in events if e['type'] == 'positive']),
            'negative_events': len([e for e in events if e['type'] == 'negative']),
            'max_return': returns.max(),
            'min_return': returns.min(),
            'volatility_regime': self._classify_volatility_regime(returns)
        }

    def _classify_volatility_regime(self, returns: pd.Series) -> str:
        """Classify current volatility regime."""
        current_vol = returns.rolling(30).std().iloc[-1]
        historical_vol = returns.std()

        if current_vol > 2 * historical_vol:
            return "high_volatility"
        elif current_vol < 0.5 * historical_vol:
            return "low_volatility"
        else:
            return "normal_volatility"


class DataQualityMonitor:
    """Monitor data quality and generate alerts."""

    def __init__(self):
        self.quality_thresholds = {
            'missing_data_threshold': 0.05,  # 5% missing data
            'outlier_threshold': 0.1,      # 10% outliers
            'stale_data_threshold': 5     # 5 days stale data
        }

    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess overall data quality.

        Returns:
            Dictionary containing quality metrics and alerts
        """
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'missing_data_analysis': self._analyze_missing_data(data),
            'outlier_analysis': self._analyze_outliers(data),
            'stale_data_analysis': self._analyze_stale_data(data),
            'overall_score': 0.0,
            'alerts': []
        }

        # Calculate overall quality score
        score_components = [
            1.0 - quality_report['missing_data_analysis']['missing_percentage'] / 100,
            1.0 - quality_report['outlier_analysis']['outlier_percentage'] / 100,
            1.0 if quality_report['stale_data_analysis']['is_stale'] else 0.0
        ]

        quality_report['overall_score'] = np.mean(score_components)

        # Generate alerts
        quality_report['alerts'] = self._generate_alerts(quality_report)

        return quality_report

    def _analyze_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = data.isnull().sum()
        missing_percentage = (missing_counts / len(data)) * 100

        return {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentage': missing_percentage.max(),
            'consecutive_missing': self._detect_consecutive_missing(data),
            'quality_ok': missing_percentage.max() < self.quality_thresholds['missing_data_threshold']
        }

    def _analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numerical columns."""
        outlier_info = {}
        total_outliers = 0

        for column in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1

            outliers = ((data[column] < (Q1 - 1.5 * IQR)) |
                       (data[column] > (Q3 + 1.5 * IQR)))
            outlier_count = outliers.sum()
            total_outliers += outlier_count

            outlier_info[column] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(data)) * 100
            }

        overall_percentage = (total_outliers / (len(data) * len(outlier_info))) * 100

        return {
            'column_wise': outlier_info,
            'outlier_percentage': overall_percentage,
            'quality_ok': overall_percentage < self.quality_thresholds['outlier_threshold']
        }

    def _analyze_stale_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for stale or outdated data."""
        if len(data) == 0:
            return {'is_stale': True, 'last_update': None}

        last_date = data.index.max()
        current_date = datetime.now()
        days_since_update = (current_date - last_date).days

        return {
            'is_stale': days_since_update > self.quality_thresholds['stale_data_threshold'],
            'last_update': last_date.isoformat(),
            'days_since_update': days_since_update
        }

    def _detect_consecutive_missing(self, data: pd.DataFrame) -> Dict[str, int]:
        """Detect consecutive missing values."""
        consecutive_missing = {}

        for column in data.columns:
            missing_series = data[column].isnull()
            if missing_series.any():
                # Group consecutive missing values
                consecutive_groups = (missing_series.diff() != 0).cumsum()
                max_consecutive = missing_series.groupby(consecutive_groups).sum().max()
                consecutive_missing[column] = max_consecutive
            else:
                consecutive_missing[column] = 0

        return consecutive_missing

    def _generate_alerts(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate quality alerts based on analysis."""
        alerts = []

        # Missing data alerts
        if not quality_report['missing_data_analysis']['quality_ok']:
            alerts.append(f"High missing data: {quality_report['missing_data_analysis']['missing_percentage']:.2f}%")

        # Outlier alerts
        if not quality_report['outlier_analysis']['quality_ok']:
            alerts.append(f"High outlier percentage: {quality_report['outlier_analysis']['outlier_percentage']:.2f}%")

        # Stale data alerts
        if quality_report['stale_data_analysis']['is_stale']:
            alerts.append(f"Stale data: {quality_report['stale_data_analysis']['days_since_update']} days since update")

        # Low overall score
        if quality_report['overall_score'] < 0.7:
            alerts.append(f"Low overall data quality score: {quality_report['overall_score']:.2f}")

        return alerts


class EnhancedDataIngestionService:
    """Main service for enhanced data ingestion with extreme event detection."""

    def __init__(self):
        self.data_sources = {
            'yahoo_finance': YahooFinanceDataSource()
        }
        self.event_detector = ExtremeEventDetector()
        self.quality_monitor = DataQualityMonitor()
        self.cleaning_library = DataCleaningLibrary()
        self.validation_library = DataValidationLibrary()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def ingest_data(self,
                        symbol: str,
                        start_date: str,
                        end_date: str,
                        source: str = 'yahoo_finance') -> Dict[str, Any]:
        """
        Ingest data with enhanced processing and event detection.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source name

        Returns:
            Dictionary containing processed data and analysis results
        """
        try:
            self.logger.info(f"Starting data ingestion for {symbol} from {start_date} to {end_date}")

            # Fetch raw data
            if source not in self.data_sources:
                raise ValueError(f"Unknown data source: {source}")

            data_source = self.data_sources[source]
            raw_data = await data_source.fetch_data(symbol, start_date, end_date)

            # Detect extreme events
            extreme_events = self.event_detector.detect_extreme_events(raw_data)

            # Assess data quality
            quality_report = self.quality_monitor.assess_data_quality(raw_data)

            # Clean and validate data
            cleaned_data = self.cleaning_library.clean_data(raw_data)
            validated_data = self.validation_library.validate_data(cleaned_data)

            # Compile results
            ingestion_result = {
                'symbol': symbol,
                'date_range': {'start': start_date, 'end': end_date},
                'data_source': source,
                'metadata': data_source.get_metadata(),
                'raw_data_stats': {
                    'records': len(raw_data),
                    'date_range': f"{raw_data.index.min()} to {raw_data.index.max()}"
                },
                'extreme_events': extreme_events,
                'data_quality': quality_report,
                'processed_data': validated_data,
                'processing_metadata': {
                    'ingestion_time': datetime.now().isoformat(),
                    'processing_steps': ['fetch', 'event_detection', 'quality_assessment', 'cleaning', 'validation'],
                    'success': True
                }
            }

            self.logger.info(f"Successfully ingested data for {symbol}: {len(validated_data)} records")
            return ingestion_result

        except Exception as e:
            self.logger.error(f"Data ingestion failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'processing_metadata': {
                    'ingestion_time': datetime.now().isoformat(),
                    'success': False
                }
            }

    async def batch_ingest(self,
                          symbols: List[str],
                          start_date: str,
                          end_date: str,
                          max_concurrent: int = 5) -> Dict[str, Any]:
        """
        Ingest data for multiple symbols concurrently.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_concurrent: Maximum concurrent ingestions

        Returns:
            Dictionary containing batch ingestion results
        """
        self.logger.info(f"Starting batch ingestion for {len(symbols)} symbols")

        # Process symbols in batches
        results = []

        for i in range(0, len(symbols), max_concurrent):
            batch = symbols[i:i + max_concurrent]

            # Process batch concurrently
            batch_tasks = [
                self.ingest_data(symbol, start_date, end_date)
                for symbol in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)

        # Compile batch results
        successful_ingestions = [r for r in results if isinstance(r, dict) and r.get('processing_metadata', {}).get('success')]
        failed_ingestions = [r for r in results if not isinstance(r, dict) or not r.get('processing_metadata', {}).get('success')]

        batch_summary = {
            'total_symbols': len(symbols),
            'successful': len(successful_ingestions),
            'failed': len(failed_ingestions),
            'success_rate': len(successful_ingestions) / len(symbols) if symbols else 0,
            'extreme_events_total': sum(len(r['extreme_events']['events']) for r in successful_ingestions),
            'average_quality_score': np.mean([r['data_quality']['overall_score'] for r in successful_ingestions]) if successful_ingestions else 0,
            'results': results
        }

        self.logger.info(f"Batch ingestion completed: {batch_summary['successful']}/{batch_summary['total_symbols']} successful")
        return batch_summary

    async def get_extreme_events_summary(self,
                                       symbol: str,
                                       start_date: str,
                                       end_date: str) -> Dict[str, Any]:
        """
        Get summary of extreme events for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary containing extreme events summary
        """
        ingestion_result = await self.ingest_data(symbol, start_date, end_date)

        if not ingestion_result.get('processing_metadata', {}).get('success'):
            return {'error': 'Failed to fetch data for extreme events analysis'}

        events = ingestion_result['extreme_events']['events']

        return {
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'total_events': len(events),
            'positive_events': len([e for e in events if e['type'] == 'positive']),
            'negative_events': len([e for e in events if e['type'] == 'negative']),
            'avg_magnitude': np.mean([e['magnitude'] for e in events]) if events else 0,
            'max_magnitude': max([e['magnitude'] for e in events]) if events else 0,
            'volatility_regime': ingestion_result['extreme_events']['volatility_regime'],
            'recent_events': sorted(events, key=lambda x: x['date'], reverse=True)[:10]
        }