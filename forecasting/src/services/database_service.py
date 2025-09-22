"""
Time series database integration with regime-aware storage.

Implements database services for storing and retrieving time series data
with regime information, optimized for quantitative financial applications.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import sqlite3
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# Import for regime detection
from forecasting.models.market_regime import MarketRegime


@dataclass
class TimeSeriesData:
    """Time series data with metadata."""
    symbol: str
    timestamps: List[str]
    values: List[float]
    metadata: Dict[str, Any]
    regime_data: Optional[Dict[str, Any]] = None


class DatabaseInterface(ABC):
    """Abstract interface for time series databases."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to database."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from database."""
        pass

    @abstractmethod
    async def store_time_series(self, data: TimeSeriesData) -> bool:
        """Store time series data."""
        pass

    @abstractmethod
    async def retrieve_time_series(self,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str) -> Optional[TimeSeriesData]:
        """Retrieve time series data."""
        pass

    @abstractmethod
    async def store_regime_data(self,
                               symbol: str,
                               regime_data: Dict[str, Any]) -> bool:
        """Store regime classification data."""
        pass

    @abstractmethod
    async def retrieve_regime_data(self,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str) -> Optional[Dict[str, Any]]:
        """Retrieve regime classification data."""
        pass


class SQLiteTimeSeriesDB(DatabaseInterface):
    """SQLite implementation for time series data storage."""

    def __init__(self, db_path: str = "forecasting_db.sqlite"):
        self.db_path = db_path
        self.connection = None
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> bool:
        """Connect to SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            await self._create_tables()
            self.logger.info(f"Connected to SQLite database: {self.db_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from database."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self.logger.info("Disconnected from database")
            return True
        except Exception as e:
            self.logger.error(f"Failed to disconnect: {e}")
            return False

    async def _create_tables(self):
        """Create necessary database tables."""
        cursor = self.connection.cursor()

        # Time series data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_series (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value REAL NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                volume INTEGER,
                regime_id INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        # Regime data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                regime_type TEXT NOT NULL,
                regime_id INTEGER NOT NULL,
                probability REAL,
                transition_matrix TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        """)

        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON time_series(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_regime_symbol_date ON regime_data(symbol, date)")

        self.connection.commit()

    async def store_time_series(self, data: TimeSeriesData) -> bool:
        """Store time series data."""
        try:
            cursor = self.connection.cursor()

            # Prepare data for insertion
            records = []
            for i, (timestamp, value) in enumerate(zip(data.timestamps, data.values)):
                regime_id = None
                if data.regime_data and 'regime_ids' in data.regime_data:
                    regime_id = data.regime_data['regime_ids'][i] if i < len(data.regime_data['regime_ids']) else None

                record = {
                    'symbol': data.symbol,
                    'timestamp': timestamp,
                    'value': value,
                    'regime_id': regime_id,
                    'metadata': json.dumps(data.metadata)
                }

                # Add OHLCV data if available
                if 'ohlc_data' in data.metadata and i < len(data.metadata['ohlc_data']):
                    ohlc = data.metadata['ohlc_data'][i]
                    record.update({
                        'open_price': ohlc.get('open'),
                        'high_price': ohlc.get('high'),
                        'low_price': ohlc.get('low'),
                        'volume': ohlc.get('volume')
                    })

                records.append(record)

            # Insert records
            for record in records:
                cursor.execute("""
                    INSERT OR REPLACE INTO time_series
                    (symbol, timestamp, value, open_price, high_price, low_price, volume, regime_id, metadata)
                    VALUES (:symbol, :timestamp, :value, :open_price, :high_price, :low_price, :volume, :regime_id, :metadata)
                """, record)

            self.connection.commit()
            self.logger.info(f"Stored {len(records)} records for {data.symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store time series data: {e}")
            return False

    async def retrieve_time_series(self,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str) -> Optional[TimeSeriesData]:
        """Retrieve time series data."""
        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                SELECT timestamp, value, open_price, high_price, low_price, volume, regime_id, metadata
                FROM time_series
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (symbol, start_date, end_date))

            rows = cursor.fetchall()

            if not rows:
                self.logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                return None

            # Extract data
            timestamps = [row['timestamp'] for row in rows]
            values = [row['value'] for row in rows]

            # Get regime data if available
            regime_ids = [row['regime_id'] for row in rows if row['regime_id'] is not None]

            # Get metadata from first record
            metadata = json.loads(rows[0]['metadata']) if rows[0]['metadata'] else {}

            # Add OHLC data if available
            ohlc_data = []
            for row in rows:
                if all([row['open_price'], row['high_price'], row['low_price']]):
                    ohlc_data.append({
                        'open': row['open_price'],
                        'high': row['high_price'],
                        'low': row['low_price'],
                        'volume': row['volume']
                    })
                else:
                    ohlc_data.append(None)

            if ohlc_data and any(ohlc_data):
                metadata['ohlc_data'] = ohlc_data

            regime_data = None
            if regime_ids:
                regime_data = {'regime_ids': regime_ids}

            return TimeSeriesData(
                symbol=symbol,
                timestamps=timestamps,
                values=values,
                metadata=metadata,
                regime_data=regime_data
            )

        except Exception as e:
            self.logger.error(f"Failed to retrieve time series data: {e}")
            return None

    async def store_regime_data(self,
                               symbol: str,
                               regime_data: Dict[str, Any]) -> bool:
        """Store regime classification data."""
        try:
            cursor = self.connection.cursor()

            # Store regime classifications
            if 'classifications' in regime_data:
                for date, classification in regime_data['classifications'].items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO regime_data
                        (symbol, date, regime_type, regime_id, probability, transition_matrix, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        date,
                        classification.get('regime_type', 'unknown'),
                        classification.get('regime_id'),
                        classification.get('probability'),
                        json.dumps(regime_data.get('transition_matrix', {})),
                        json.dumps(classification.get('metadata', {}))
                    ))

            self.connection.commit()
            self.logger.info(f"Stored regime data for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store regime data: {e}")
            return False

    async def retrieve_regime_data(self,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str) -> Optional[Dict[str, Any]]:
        """Retrieve regime classification data."""
        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                SELECT date, regime_type, regime_id, probability, transition_matrix, metadata
                FROM regime_data
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, (symbol, start_date, end_date))

            rows = cursor.fetchall()

            if not rows:
                return None

            classifications = {}
            for row in rows:
                classifications[row['date']] = {
                    'regime_type': row['regime_type'],
                    'regime_id': row['regime_id'],
                    'probability': row['probability'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }

            return {
                'classifications': classifications,
                'transition_matrix': json.loads(rows[0]['transition_matrix']) if rows[0]['transition_matrix'] else None
            }

        except Exception as e:
            self.logger.error(f"Failed to retrieve regime data: {e}")
            return None

    async def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM time_series ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get available symbols: {e}")
            return []

    async def get_data_range(self, symbol: str) -> Optional[Dict[str, str]]:
        """Get date range for a symbol."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT MIN(timestamp) as start_date, MAX(timestamp) as end_date
                FROM time_series
                WHERE symbol = ?
            """, (symbol,))

            row = cursor.fetchone()
            if row and row['start_date']:
                return {
                    'start_date': row['start_date'],
                    'end_date': row['end_date']
                }
            return None

        except Exception as e:
            self.logger.error(f"Failed to get data range for {symbol}: {e}")
            return None

    async def delete_data(self,
                         symbol: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> bool:
        """Delete data for a symbol within date range."""
        try:
            cursor = self.connection.cursor()

            if start_date and end_date:
                # Delete specific date range
                cursor.execute("""
                    DELETE FROM time_series
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                """, (symbol, start_date, end_date))
            else:
                # Delete all data for symbol
                cursor.execute("DELETE FROM time_series WHERE symbol = ?", (symbol,))

            deleted_count = cursor.rowcount
            self.connection.commit()

            self.logger.info(f"Deleted {deleted_count} records for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete data for {symbol}: {e}")
            return False


class TimeSeriesDatabaseService:
    """Main service for time series database operations."""

    def __init__(self, db_type: str = "sqlite", **kwargs):
        self.db_type = db_type
        self.kwargs = kwargs
        self.db_interface = None
        self.logger = logging.getLogger(__name__)

        # Initialize database interface
        if db_type == "sqlite":
            db_path = kwargs.get('db_path', 'forecasting_db.sqlite')
            self.db_interface = SQLiteTimeSeriesDB(db_path)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    async def initialize(self) -> bool:
        """Initialize database connection."""
        return await self.db_interface.connect()

    async def shutdown(self) -> bool:
        """Shutdown database connection."""
        return await self.db_interface.disconnect()

    async def store_market_data(self,
                              symbol: str,
                              data: pd.DataFrame,
                              regime_data: Optional[Dict[str, Any]] = None) -> bool:
        """Store market data with regime information."""
        try:
            # Convert DataFrame to TimeSeriesData
            timestamps = data.index.strftime('%Y-%m-%d').tolist()
            values = data['Close'].tolist() if 'Close' in data.columns else data.iloc[:, 0].tolist()

            metadata = {
                'data_source': 'market_data',
                'columns': data.columns.tolist(),
                'records': len(data)
            }

            # Add OHLC data if available
            if all(col in data.columns for col in ['Open', 'High', 'Low']):
                metadata['ohlc_data'] = [
                    {
                        'open': row['Open'],
                        'high': row['High'],
                        'low': row['Low'],
                        'volume': row.get('Volume', 0)
                    }
                    for _, row in data.iterrows()
                ]

            time_series_data = TimeSeriesData(
                symbol=symbol,
                timestamps=timestamps,
                values=values,
                metadata=metadata,
                regime_data=regime_data
            )

            return await self.db_interface.store_time_series(time_series_data)

        except Exception as e:
            self.logger.error(f"Failed to store market data for {symbol}: {e}")
            return False

    async def retrieve_market_data(self,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str) -> Optional[pd.DataFrame]:
        """Retrieve market data as DataFrame."""
        try:
            time_series_data = await self.db_interface.retrieve_time_series(
                symbol, start_date, end_date
            )

            if time_series_data is None:
                return None

            # Convert to DataFrame
            df = pd.DataFrame({
                'Close': time_series_data.values
            }, index=pd.to_datetime(time_series_data.timestamps))

            # Add OHLC data if available
            if (time_series_data.metadata and 'ohlc_data' in time_series_data.metadata and
                time_series_data.metadata['ohlc_data']):

                ohlc_data = time_series_data.metadata['ohlc_data']
                df['Open'] = [ohlc['open'] if ohlc else np.nan for ohlc in ohlc_data]
                df['High'] = [ohlc['high'] if ohlc else np.nan for ohlc in ohlc_data]
                df['Low'] = [ohlc['low'] if ohlc else np.nan for ohlc in ohlc_data]
                df['Volume'] = [ohlc['volume'] if ohlc else 0 for ohlc in ohlc_data]

            return df

        except Exception as e:
            self.logger.error(f"Failed to retrieve market data for {symbol}: {e}")
            return None

    async def store_regime_analysis(self,
                                  symbol: str,
                                  regime_analysis: Dict[str, Any]) -> bool:
        """Store regime analysis results."""
        return await self.db_interface.store_regime_data(symbol, regime_analysis)

    async def retrieve_regime_analysis(self,
                                     symbol: str,
                                     start_date: str,
                                     end_date: str) -> Optional[Dict[str, Any]]:
        """Retrieve regime analysis results."""
        return await self.db_interface.retrieve_regime_data(symbol, start_date, end_date)

    async def get_regime_aware_data(self,
                                   symbol: str,
                                   start_date: str,
                                   end_date: str) -> Optional[Dict[str, Any]]:
        """Retrieve both market data and regime information."""
        try:
            # Get market data
            market_data = await self.retrieve_market_data(symbol, start_date, end_date)
            if market_data is None:
                return None

            # Get regime data
            regime_data = await self.retrieve_regime_analysis(symbol, start_date, end_date)

            return {
                'symbol': symbol,
                'date_range': {'start': start_date, 'end': end_date},
                'market_data': market_data.to_dict('records'),
                'regime_data': regime_data,
                'combined_stats': self._calculate_combined_stats(market_data, regime_data)
            }

        except Exception as e:
            self.logger.error(f"Failed to get regime-aware data for {symbol}: {e}")
            return None

    def _calculate_combined_stats(self,
                                market_data: pd.DataFrame,
                                regime_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate combined statistics for market and regime data."""
        stats = {
            'market_stats': {
                'records': len(market_data),
                'date_range': f"{market_data.index.min()} to {market_data.index.max()}",
                'return_stats': {
                    'mean_return': market_data['Close'].pct_change().mean(),
                    'volatility': market_data['Close'].pct_change().std(),
                    'max_return': market_data['Close'].pct_change().max(),
                    'min_return': market_data['Close'].pct_change().min()
                }
            }
        }

        # Add regime statistics if available
        if regime_data and 'classifications' in regime_data:
            regime_counts = {}
            for classification in regime_data['classifications'].values():
                regime_id = classification.get('regime_id')
                if regime_id is not None:
                    regime_counts[regime_id] = regime_counts.get(regime_id, 0) + 1

            stats['regime_stats'] = {
                'total_regime_points': len(regime_data['classifications']),
                'regime_distribution': regime_counts,
                'unique_regimes': len(regime_counts)
            }

        return stats

    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            symbols = await self.db_interface.get_available_symbols()

            symbol_info = {}
            for symbol in symbols:
                data_range = await self.db_interface.get_data_range(symbol)
                symbol_info[symbol] = {
                    'data_range': data_range,
                    'symbol': symbol
                }

            return {
                'database_type': self.db_type,
                'total_symbols': len(symbols),
                'symbols': symbol_info,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}