#!/usr/bin/env python3
"""
Data management utilities for quantitative trading system.

Provides tools for data maintenance, cleanup, quality checks, and
automated data refresh operations.
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.src.storage.market_data_storage import MarketDataStorage, create_default_storage
from data.src.feeds.yahoo_finance_ingestion import YahooFinanceIngestion, AssetClass, create_default_ingestion


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class DataManager:
    """Data management utilities for quantitative trading system."""

    def __init__(self, base_path: str = "data/storage"):
        """Initialize data manager."""
        self.base_path = Path(base_path)
        self.storage = create_default_storage()
        self.ingestion = create_default_ingestion()
        self.logger = logging.getLogger(__name__)

        # Define universes directly
        self.universes = {
            'sp500_large_cap': {
                AssetClass.EQUITY: [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
                    'PG', 'UNH', 'HD', 'MA', 'PYPL', 'DIS', 'NFLX', 'ADBE', 'CRM', 'BAC',
                    'XOM', 'CVX', 'LLY', 'ABBV', 'MRK', 'PEP', 'KO', 'AVGO', 'COST', 'LIN',
                    'TMO', 'ABT', 'ACN', 'DHR', 'WFC', 'MCD', 'NKE', 'TXN', 'NEE', 'PM',
                    'HON', 'UNP', 'RTX', 'LOW', 'INTU', 'QCOM', 'CAT', 'CSCO', 'DE', 'GE',
                    'SCHW', 'BKNG', 'AMT', 'MS', 'BLK', 'GS', 'BA', 'AMGN', 'IBM', 'PLD',
                    'CVS', 'MDT', 'C', 'LMT', 'AXP', 'CI', 'ETN', 'CB', 'T', 'ADP',
                    'MMC', 'ISRG', 'MO', 'PFE', 'COP', 'DUK', 'ORCL', 'SO', 'BSX', 'SYK'
                ],
                AssetClass.ETF: [
                    'SPY', 'VOO', 'IVV', 'VTI', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'XLF',
                    'XLK', 'XLV', 'XLU', 'XLE', 'XLP', 'XLY', 'XLB', 'XLI', 'XLRE', 'XME'
                ],
                AssetClass.FX: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD'],
                AssetClass.BOND: ['US10Y', 'US2Y', 'US5Y', 'US30Y'],
                AssetClass.COMMODITY: ['GOLD', 'SILVER', 'OIL', 'NATGAS', 'COPPER'],
                AssetClass.INDEX: ['^GSPC', '^DJI', '^IXIC', '^VIX']
            }
        }

    def get_storage_summary(self) -> Dict[str, Any]:
        """Get comprehensive storage summary."""
        # Get basic storage info
        storage_info = self.storage.get_storage_info()

        # Get detailed coverage
        coverage = self.storage.get_data_coverage_summary()

        # Calculate age of data
        data_ages = {}
        for key, metadata in self.storage.metadata_registry.items():
            start_date = datetime.fromisoformat(metadata.start_date)
            end_date = datetime.fromisoformat(metadata.end_date)
            data_ages[key] = {
                'start_date': start_date,
                'end_date': end_date,
                'days_span': (end_date - start_date).days,
                'days_since_update': (datetime.now() - end_date).days
            }

        return {
            'storage_info': storage_info,
            'coverage': coverage,
            'data_ages': data_ages,
            'oldest_data': min(data_ages.values(), key=lambda x: x['start_date'])['start_date'],
            'newest_data': max(data_ages.values(), key=lambda x: x['end_date'])['end_date'],
            'average_update_age': sum(d['days_since_update'] for d in data_ages.values()) / len(data_ages) if data_ages else 0
        }

    def check_data_quality(self) -> Dict[str, Any]:
        """Check quality of stored data."""
        quality_issues = []
        quality_stats = {
            'total_symbols': 0,
            'symbols_with_issues': 0,
            'issues_by_type': {}
        }

        for key, metadata in self.storage.metadata_registry.items():
            quality_stats['total_symbols'] += 1

            # Load data for quality check
            data = self.storage.load_data(
                metadata.symbol,
                AssetClass(metadata.asset_class),
                datetime.fromisoformat(metadata.start_date),
                datetime.fromisoformat(metadata.end_date)
            )

            if data is None:
                quality_issues.append({
                    'symbol': metadata.symbol,
                    'asset_class': metadata.asset_class,
                    'issue': 'data_load_failed',
                    'severity': 'high'
                })
                continue

            # Check for missing data
            missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
            if missing_pct > 5:
                quality_issues.append({
                    'symbol': metadata.symbol,
                    'asset_class': metadata.asset_class,
                    'issue': 'high_missing_data',
                    'severity': 'medium' if missing_pct < 20 else 'high',
                    'missing_percentage': missing_pct
                })

            # Check for data gaps
            date_gaps = data.index.to_series().diff()
            max_gap = date_gaps.max()
            if max_gap > timedelta(days=7):
                quality_issues.append({
                    'symbol': metadata.symbol,
                    'asset_class': metadata.asset_class,
                    'issue': 'data_gaps',
                    'severity': 'medium',
                    'max_gap_days': max_gap.days
                })

            # Check for stale data
            days_since_update = (datetime.now() - data.index.max()).days
            if days_since_update > 7:
                quality_issues.append({
                    'symbol': metadata.symbol,
                    'asset_class': metadata.asset_class,
                    'issue': 'stale_data',
                    'severity': 'low' if days_since_update < 30 else 'medium',
                    'days_stale': days_since_update
                })

        # Count issues by type
        for issue in quality_issues:
            issue_type = issue['issue']
            if issue_type not in quality_stats['issues_by_type']:
                quality_stats['issues_by_type'][issue_type] = 0
            quality_stats['issues_by_type'][issue_type] += 1

        quality_stats['symbols_with_issues'] = len(set(issue['symbol'] for issue in quality_issues))

        return {
            'quality_issues': quality_issues,
            'quality_stats': quality_stats,
            'overall_health_score': max(0, 100 - (len(quality_issues) / quality_stats['total_symbols'] * 100)) if quality_stats['total_symbols'] > 0 else 100
        }

    def cleanup_old_data(self, keep_years: int = 5, dry_run: bool = False) -> Dict[str, Any]:
        """Clean up old data files based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=keep_years * 365)

        files_to_remove = []
        total_size_freed = 0

        for key, metadata in self.storage.metadata_registry.items():
            end_date = datetime.fromisoformat(metadata.end_date)
            if end_date < cutoff_date:
                files_to_remove.append({
                    'key': key,
                    'symbol': metadata.symbol,
                    'asset_class': metadata.asset_class,
                    'end_date': end_date,
                    'file_size': metadata.file_size,
                    'age_days': (datetime.now() - end_date).days
                })
                total_size_freed += metadata.file_size

        if dry_run:
            return {
                'dry_run': True,
                'files_to_remove': len(files_to_remove),
                'total_size_freed_bytes': total_size_freed,
                'total_size_freed_mb': total_size_freed / (1024 * 1024),
                'cutoff_date': cutoff_date.isoformat(),
                'files': files_to_remove
            }

        # Actually remove files
        removed_count = 0
        for file_info in files_to_remove:
            try:
                # Remove from metadata registry
                del self.storage.metadata_registry[file_info['key']]

                # Remove actual file
                asset_dir = self.base_path / "raw" / file_info['asset_class']
                filename_pattern = f"{file_info['symbol']}_*.*"
                matching_files = list(asset_dir.glob(filename_pattern))

                for file_path in matching_files:
                    file_path.unlink()

                removed_count += 1
                self.logger.info(f"Removed old data for {file_info['symbol']}")

            except Exception as e:
                self.logger.error(f"Error removing {file_info['symbol']}: {e}")

        # Save updated metadata
        self.storage._save_metadata_registry()

        return {
            'dry_run': False,
            'files_removed': removed_count,
            'total_size_freed_bytes': total_size_freed,
            'total_size_freed_mb': total_size_freed / (1024 * 1024),
            'cutoff_date': cutoff_date.isoformat()
        }

    def update_stale_data(self, days_threshold: int = 7, max_workers: int = 4) -> Dict[str, Any]:
        """Update data that hasn't been updated recently."""
        stale_symbols = {}
        total_updated = 0

        # Find stale data
        for key, metadata in self.storage.metadata_registry.items():
            end_date = datetime.fromisoformat(metadata.end_date)
            days_stale = (datetime.now() - end_date).days

            if days_stale >= days_threshold:
                asset_class = AssetClass(metadata.asset_class)
                if asset_class not in stale_symbols:
                    stale_symbols[asset_class] = []
                stale_symbols[asset_class].append(metadata.symbol)

        if not stale_symbols:
            return {'status': 'no_stale_data', 'symbols_updated': 0}

        # Use existing ingestion instance with updated max_workers
        self.ingestion.max_workers = max_workers

        # Update each asset class
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Get last 30 days to ensure coverage

        for asset_class, symbols in stale_symbols.items():
            self.logger.info(f"Updating {asset_class.value}: {len(symbols)} symbols")

            # Create requests for incremental update
            requests = []
            for symbol in symbols:
                request = ingestion.base_ingestion.create_batch_requests(
                    [symbol], asset_class, start_date, end_date
                )
                requests.extend(request)

            # Fetch updated data
            results = self.ingestion.fetch_multiple_assets(requests)

            # Store successful updates
            successful_results = {
                symbol: result for symbol, result in results.items()
                if result.success and result.data is not None
            }

            storage_results = self.storage.save_multiple_results(successful_results)
            successful_saves = sum(1 for success in storage_results.values() if success)

            total_updated += successful_saves
            self.logger.info(f"Updated {successful_saves}/{len(symbols)} symbols for {asset_class.value}")

        return {
            'status': 'update_complete',
            'symbols_updated': total_updated,
            'stale_symbols_found': sum(len(symbols) for symbols in stale_symbols.values()),
            'update_threshold_days': days_threshold
        }

    def repair_data_issues(self, dry_run: bool = True) -> Dict[str, Any]:
        """Attempt to repair common data issues."""
        quality_report = self.check_data_quality()
        repairs_made = []

        for issue in quality_report['quality_issues']:
            if issue['issue'] == 'stale_data' and issue['severity'] in ['medium', 'high']:
                # Schedule for update
                repairs_made.append({
                    'symbol': issue['symbol'],
                    'asset_class': issue['asset_class'],
                    'action': 'schedule_update',
                    'reason': f"Data is {issue['days_stale']} days old"
                })

            elif issue['issue'] == 'data_gaps' and issue['severity'] == 'high':
                # Schedule for re-download
                repairs_made.append({
                    'symbol': issue['symbol'],
                    'asset_class': issue['asset_class'],
                    'action': 'schedule_redownload',
                    'reason': f"Data gap of {issue['max_gap_days']} days detected"
                })

            elif issue['issue'] == 'high_missing_data' and issue['severity'] == 'high':
                # Schedule for re-download
                repairs_made.append({
                    'symbol': issue['symbol'],
                    'asset_class': issue['asset_class'],
                    'action': 'schedule_redownload',
                    'reason': f"High missing data: {issue['missing_percentage']:.1f}%"
                })

        if not dry_run:
            # Execute repairs (would need implementation)
            pass

        return {
            'dry_run': dry_run,
            'issues_found': len(quality_report['quality_issues']),
            'repairs_scheduled': len(repairs_made),
            'repairs': repairs_made
        }

    def export_data_manifest(self, output_path: str = None) -> str:
        """Export comprehensive data manifest."""
        if output_path is None:
            output_path = self.base_path / "metadata" / "data_manifest.json"

        manifest = {
            'generated_at': datetime.now().isoformat(),
            'storage_summary': self.get_storage_summary(),
            'quality_report': self.check_data_quality(),
            'universes': {
                universe_name: {
                    'description': f'{universe_name} universe',
                    'symbols_by_asset_class': {
                        asset_class.value: len(symbols)
                        for asset_class, symbols in universe_data.items()
                    }
                }
                for universe_name, universe_data in self.universes.items()
            }
        }

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        self.logger.info(f"Data manifest exported to {output_path}")
        return str(output_path)

    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive data health check."""
        summary = self.get_storage_summary()
        quality = self.check_data_quality()

        health_score = quality['overall_health_score']

        # Determine overall health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        else:
            status = "poor"

        return {
            'overall_status': status,
            'health_score': health_score,
            'storage_summary': summary,
            'quality_report': quality,
            'recommendations': self._generate_recommendations(quality),
            'last_check': datetime.now().isoformat()
        }

    def _generate_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality report."""
        recommendations = []

        issue_stats = quality_report['quality_stats']['issues_by_type']

        if 'stale_data' in issue_stats:
            recommendations.append(f"Update {issue_stats['stale_data']} symbols with stale data")

        if 'high_missing_data' in issue_stats:
            recommendations.append(f"Re-download {issue_stats['high_missing_data']} symbols with high missing data")

        if 'data_gaps' in issue_stats:
            recommendations.append(f"Repair {issue_stats['data_gaps']} symbols with data gaps")

        if quality_report['quality_stats']['symbols_with_issues'] > 0:
            issue_percentage = (quality_report['quality_stats']['symbols_with_issues'] / quality_report['quality_stats']['total_symbols']) * 100
            if issue_percentage > 20:
                recommendations.append("Consider running comprehensive data repair")

        return recommendations


def main():
    """Main function for data management CLI."""
    parser = argparse.ArgumentParser(description='Data management utilities')
    parser.add_argument('--action', type=str, required=True,
                       choices=['summary', 'quality', 'cleanup', 'update', 'repair', 'health', 'export'],
                       help='Action to perform')
    parser.add_argument('--base-path', type=str, default='data/storage',
                       help='Base path for data storage')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    parser.add_argument('--keep-years', type=int, default=5,
                       help='Years of data to keep (for cleanup)')
    parser.add_argument('--days-threshold', type=int, default=7,
                       help='Days threshold for stale data (for update)')
    parser.add_argument('--output', type=str,
                       help='Output file path (for export)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Initialize data manager
        manager = DataManager(base_path=args.base_path)

        # Execute requested action
        if args.action == 'summary':
            result = manager.get_storage_summary()
            print(f"\n📊 Storage Summary:")
            print(f"  Total files: {result['storage_info']['total_files']}")
            print(f"  Total size: {result['storage_info']['total_size_mb']:.2f} MB")
            print(f"  Unique symbols: {result['coverage']['total_unique_symbols']}")
            print(f"  Oldest data: {result['oldest_data'].strftime('%Y-%m-%d')}")
            print(f"  Newest data: {result['newest_data'].strftime('%Y-%m-%d')}")
            print(f"  Average update age: {result['average_update_age']:.1f} days")

        elif args.action == 'quality':
            result = manager.check_data_quality()
            print(f"\n🔍 Data Quality Report:")
            print(f"  Overall health score: {result['overall_health_score']:.1f}%")
            print(f"  Total symbols: {result['quality_stats']['total_symbols']}")
            print(f"  Symbols with issues: {result['quality_stats']['symbols_with_issues']}")
            print(f"  Issues by type: {result['quality_stats']['issues_by_type']}")

            if result['quality_issues']:
                print(f"\n  Top issues:")
                for issue in result['quality_issues'][:5]:
                    print(f"    {issue['symbol']} ({issue['asset_class']}): {issue['issue']}")

        elif args.action == 'cleanup':
            result = manager.cleanup_old_data(args.keep_years, args.dry_run)
            if args.dry_run:
                print(f"\n🧹 Cleanup (Dry Run):")
                print(f"  Files to remove: {result['files_to_remove']}")
                print(f"  Space to free: {result['total_size_freed_mb']:.2f} MB")
            else:
                print(f"\n🧹 Cleanup Complete:")
                print(f"  Files removed: {result['files_removed']}")
                print(f"  Space freed: {result['total_size_freed_mb']:.2f} MB")

        elif args.action == 'update':
            result = manager.update_stale_data(args.days_threshold)
            print(f"\n🔄 Data Update:")
            print(f"  Status: {result['status']}")
            print(f"  Symbols updated: {result['symbols_updated']}")
            if result['stale_symbols_found'] > 0:
                print(f"  Stale symbols found: {result['stale_symbols_found']}")

        elif args.action == 'repair':
            result = manager.repair_data_issues(args.dry_run)
            print(f"\n🔧 Data Repair:")
            print(f"  Dry run: {result['dry_run']}")
            print(f"  Issues found: {result['issues_found']}")
            print(f"  Repairs scheduled: {result['repairs_scheduled']}")

        elif args.action == 'health':
            result = manager.run_health_check()
            print(f"\n🏥 Data Health Check:")
            print(f"  Overall status: {result['overall_status']}")
            print(f"  Health score: {result['health_score']:.1f}%")
            if result['recommendations']:
                print(f"\n  Recommendations:")
                for rec in result['recommendations']:
                    print(f"    - {rec}")

        elif args.action == 'export':
            output_path = manager.export_data_manifest(args.output)
            print(f"\n📤 Data manifest exported to: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Error executing {args.action}: {e}")
        return 1


if __name__ == "__main__":
    exit(main())