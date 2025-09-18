"""
Market data storage system for quantitative trading.

Provides persistent storage for ingested market data with support for
multiple file formats, versioning, and efficient retrieval.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import pickle
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Import data types from feeds
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'feeds'))
from yahoo_finance_ingestion import AssetClass, IngestionResult


class StorageFormat(Enum):
    """Supported storage formats."""
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    FEATHER = "feather"


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    XZ = "xz"
    SNAPPY = "snappy"


@dataclass
class StorageMetadata:
    """Metadata for stored market data."""
    symbol: str
    asset_class: str
    source: str
    start_date: str
    end_date: str
    data_points: int
    file_size: int
    file_format: str
    compression: str
    checksum: str
    created_at: str
    updated_at: str
    version: str = "1.0"
    columns: List[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        if self.columns is None:
            self.columns = []


class MarketDataStorage:
    """Market data storage system with versioning and multiple format support."""

    def __init__(self, base_path: str = "data/storage", default_format: StorageFormat = StorageFormat.PARQUET,
                 compression: CompressionType = CompressionType.NONE, enable_versioning: bool = True):
        """
        Initialize market data storage.

        Args:
            base_path: Base directory for data storage
            default_format: Default storage format
            compression: Compression type
            enable_versioning: Whether to enable data versioning
        """
        self.base_path = Path(base_path)
        self.default_format = default_format
        self.compression = compression
        self.enable_versioning = enable_versioning
        self.logger = logging.getLogger(__name__)

        # Create base directories
        self._create_directory_structure()

        # Metadata storage
        self.metadata_file = self.base_path / "metadata" / "storage_registry.json"
        self.metadata_registry = self._load_metadata_registry()

    def _create_directory_structure(self):
        """Create the directory structure for data storage."""
        directories = [
            self.base_path,
            self.base_path / "raw",
            self.base_path / "processed",
            self.base_path / "metadata",
            self.base_path / "temp"
        ]

        for asset_class in AssetClass:
            asset_dir = self.base_path / "raw" / asset_class.value
            directories.extend([asset_dir])

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_metadata_registry(self) -> Dict[str, StorageMetadata]:
        """Load metadata registry from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return {k: StorageMetadata(**v) for k, v in data.items()}
            except Exception as e:
                self.logger.error(f"Error loading metadata registry: {e}")
                return {}
        return {}

    def _save_metadata_registry(self):
        """Save metadata registry to disk."""
        try:
            # Create directory if it doesn't exist
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert metadata objects to dictionaries
            registry_dict = {k: asdict(v) for k, v in self.metadata_registry.items()}

            with open(self.metadata_file, 'w') as f:
                json.dump(registry_dict, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving metadata registry: {e}")

    def _generate_file_path(self, symbol: str, asset_class: AssetClass,
                           start_date: datetime, end_date: datetime,
                           file_format: StorageFormat) -> Path:
        """Generate file path for data storage."""
        asset_dir = self.base_path / "raw" / asset_class.value

        # Create filename with date range
        date_str = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
        base_filename = f"{symbol}_{date_str}"

        # Add versioning if enabled
        if self.enable_versioning:
            version = 1
            while True:
                version_str = f"_v{version}" if version > 1 else ""
                filename = f"{base_filename}{version_str}.{file_format.value}"
                filepath = asset_dir / filename

                if not filepath.exists():
                    break
                version += 1
        else:
            filename = f"{base_filename}.{file_format.value}"
            filepath = asset_dir / filename

        return filepath

    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate checksum for file integrity verification."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _save_data(self, data: pd.DataFrame, filepath: Path,
                   file_format: StorageFormat, compression: CompressionType) -> bool:
        """Save data to file using specified format."""
        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save based on format
            if file_format == StorageFormat.CSV:
                compression_suffix = f".{compression.value}" if compression != CompressionType.NONE else ""
                data.to_csv(filepath.with_suffix(compression_suffix), index=True)
            elif file_format == StorageFormat.PARQUET:
                data.to_parquet(filepath, engine='pyarrow', compression=compression.value if compression != CompressionType.NONE else None)
            elif file_format == StorageFormat.HDF5:
                data.to_hdf(filepath, key='data', mode='w', complib=compression.value if compression != CompressionType.NONE else None)
            elif file_format == StorageFormat.PICKLE:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            elif file_format == StorageFormat.FEATHER:
                data.to_feather(filepath)

            return True

        except Exception as e:
            self.logger.error(f"Error saving data to {filepath}: {e}")
            return False

    def _load_data(self, filepath: Path, file_format: StorageFormat) -> Optional[pd.DataFrame]:
        """Load data from file."""
        try:
            if file_format == StorageFormat.CSV:
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif file_format == StorageFormat.PARQUET:
                return pd.read_parquet(filepath)
            elif file_format == StorageFormat.HDF5:
                return pd.read_hdf(filepath, key='data')
            elif file_format == StorageFormat.PICKLE:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            elif file_format == StorageFormat.FEATHER:
                return pd.read_feather(filepath)

        except Exception as e:
            self.logger.error(f"Error loading data from {filepath}: {e}")
            return None

    def save_ingestion_result(self, result: IngestionResult,
                            file_format: Optional[StorageFormat] = None,
                            compression: Optional[CompressionType] = None) -> bool:
        """
        Save ingestion result to storage.

        Args:
            result: Ingestion result to save
            file_format: Override default file format
            compression: Override default compression

        Returns:
            True if successful, False otherwise
        """
        if not result.success or result.data is None:
            self.logger.warning(f"Cannot save failed result for {result.metadata.get('symbol', 'unknown')}")
            return False

        try:
            # Use provided format or default
            save_format = file_format or self.default_format
            save_compression = compression or self.compression

            # Extract metadata
            symbol = result.metadata.get('symbol', 'unknown')
            asset_class = AssetClass(result.metadata.get('asset_class', 'equity'))
            start_date = datetime.fromisoformat(result.metadata.get('data_range', {}).get('min_date', '2020-01-01'))
            end_date = datetime.fromisoformat(result.metadata.get('data_range', {}).get('max_date', '2020-01-01'))

            # Generate file path
            filepath = self._generate_file_path(symbol, asset_class, start_date, end_date, save_format)

            # Save data
            success = self._save_data(result.data, filepath, save_format, save_compression)

            if success:
                # Calculate checksum
                checksum = self._calculate_checksum(filepath)

                # Create metadata
                metadata = StorageMetadata(
                    symbol=symbol,
                    asset_class=asset_class.value,
                    source=result.metadata.get('source', 'unknown'),
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    data_points=len(result.data),
                    file_size=filepath.stat().st_size,
                    file_format=save_format.value,
                    compression=save_compression.value,
                    checksum=checksum,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    columns=list(result.data.columns),
                    description=f"Market data for {symbol} ({asset_class.value})"
                )

                # Store metadata
                storage_key = f"{asset_class.value}_{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                self.metadata_registry[storage_key] = metadata
                self._save_metadata_registry()

                self.logger.info(f"Saved {symbol} data to {filepath}")
                return True

        except Exception as e:
            self.logger.error(f"Error saving ingestion result for {symbol}: {e}")

        return False

    def save_multiple_results(self, results: Dict[str, IngestionResult],
                            file_format: Optional[StorageFormat] = None,
                            compression: Optional[CompressionType] = None) -> Dict[str, bool]:
        """Save multiple ingestion results concurrently."""
        save_results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.save_ingestion_result, result, file_format, compression): symbol
                for symbol, result in results.items()
            }

            for future in futures:
                symbol = futures[future]
                try:
                    save_results[symbol] = future.result()
                except Exception as e:
                    self.logger.error(f"Error saving {symbol}: {e}")
                    save_results[symbol] = False

        return save_results

    def load_data(self, symbol: str, asset_class: AssetClass,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Load data for a specific symbol and asset class.

        Args:
            symbol: Asset symbol
            asset_class: Asset class
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with loaded data or None if not found
        """
        try:
            # Find matching files in metadata
            matching_keys = [
                key for key, metadata in self.metadata_registry.items()
                if metadata.symbol == symbol and metadata.asset_class == asset_class.value
            ]

            if not matching_keys:
                self.logger.warning(f"No data found for {symbol} ({asset_class.value})")
                return None

            # Get the most recent data
            latest_key = max(matching_keys, key=lambda k: self.metadata_registry[k].created_at)
            metadata = self.metadata_registry[latest_key]

            # Construct file path
            asset_dir = self.base_path / "raw" / asset_class.value
            filename_pattern = f"{symbol}_*.{metadata.file_format}"
            matching_files = list(asset_dir.glob(filename_pattern))

            if not matching_files:
                self.logger.error(f"Data file not found for {symbol}")
                return None

            # Load the most recent file
            latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
            data = self._load_data(latest_file, StorageFormat(metadata.file_format))

            if data is not None:
                # Apply date filters if specified
                if start_date is not None:
                    data = data[data.index >= start_date]
                if end_date is not None:
                    data = data[data.index <= end_date]

                self.logger.info(f"Loaded {len(data)} data points for {symbol}")
                return data

        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")

        return None

    def get_available_symbols(self, asset_class: Optional[AssetClass] = None) -> List[str]:
        """Get list of available symbols in storage."""
        symbols = set()

        for metadata in self.metadata_registry.values():
            if asset_class is None or metadata.asset_class == asset_class.value:
                symbols.add(metadata.symbol)

        return sorted(list(symbols))

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage system information."""
        total_files = len(self.metadata_registry)
        total_size = sum(metadata.file_size for metadata in self.metadata_registry.values())

        asset_class_counts = {}
        for metadata in self.metadata_registry.values():
            asset_class = metadata.asset_class
            asset_class_counts[asset_class] = asset_class_counts.get(asset_class, 0) + 1

        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'asset_class_distribution': asset_class_counts,
            'base_path': str(self.base_path),
            'default_format': self.default_format.value,
            'compression': self.compression.value,
            'versioning_enabled': self.enable_versioning
        }

    def cleanup_old_versions(self, keep_versions: int = 2):
        """Clean up old versions of data files."""
        # Group by symbol and asset class
        symbol_groups = {}
        for key, metadata in self.metadata_registry.items():
            group_key = (metadata.symbol, metadata.asset_class)
            if group_key not in symbol_groups:
                symbol_groups[group_key] = []
            symbol_groups[group_key].append((key, metadata))

        # For each group, keep only the most recent versions
        keys_to_remove = []
        for (symbol, asset_class), group in symbol_groups.items():
            if len(group) > keep_versions:
                # Sort by creation date and remove oldest
                group.sort(key=lambda x: x[1].created_at, reverse=True)
                for key, _ in group[keep_versions:]:
                    keys_to_remove.append(key)

        # Remove old files and metadata
        for key in keys_to_remove:
            try:
                metadata = self.metadata_registry[key]
                asset_dir = self.base_path / "raw" / metadata.asset_class
                filename_pattern = f"{metadata.symbol}_*.{metadata.file_format}"
                matching_files = list(asset_dir.glob(filename_pattern))

                # Remove files
                for file_path in matching_files:
                    file_path.unlink()

                # Remove from metadata
                del self.metadata_registry[key]

                self.logger.info(f"Removed old version: {key}")

            except Exception as e:
                self.logger.error(f"Error removing old version {key}: {e}")

        # Save updated metadata
        if keys_to_remove:
            self._save_metadata_registry()

    def export_storage_summary(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """Export storage summary as DataFrame."""
        summary_data = []
        for key, metadata in self.metadata_registry.items():
            summary_data.append({
                'symbol': metadata.symbol,
                'asset_class': metadata.asset_class,
                'start_date': metadata.start_date,
                'end_date': metadata.end_date,
                'data_points': metadata.data_points,
                'file_size_mb': metadata.file_size / (1024 * 1024),
                'file_format': metadata.file_format,
                'compression': metadata.compression,
                'created_at': metadata.created_at,
                'updated_at': metadata.updated_at,
                'storage_key': key
            })

        df = pd.DataFrame(summary_data)

        if output_path:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Exported storage summary to {output_path}")

        return df


def create_default_storage() -> MarketDataStorage:
    """Create a default market data storage instance."""
    return MarketDataStorage(
        base_path="data/storage",
        default_format=StorageFormat.PARQUET,
        compression=CompressionType.NONE,
        enable_versioning=True
    )