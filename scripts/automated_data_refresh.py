#!/usr/bin/env python3
"""
Automated data refresh system for quantitative trading.

Provides scheduled data updates, quality monitoring, and maintenance
operations to keep the market data current and healthy.
"""

import sys
import os
import logging
import argparse
import json
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.src.storage.market_data_storage import create_default_storage
from data.src.feeds.yahoo_finance_ingestion import create_default_ingestion, AssetClass
from data_management import DataManager


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    import logging.handlers

    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Add rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10*1024*1024, backupCount=5
        )
        handlers.append(file_handler)

    # Configure logging
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set formatter for all handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)


class AutomatedDataRefresh:
    """Automated data refresh system for quantitative trading."""

    def __init__(self, base_path: str = "data/storage", config_file: str = None):
        """
        Initialize automated data refresh system.

        Args:
            base_path: Base path for data storage
            config_file: Configuration file path
        """
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        self.data_manager = DataManager(base_path=base_path)
        self.storage = self.data_manager.storage
        self.ingestion = self.data_manager.ingestion

        # Load configuration
        self.config = self._load_config(config_file)
        self.running = False
        self.scheduler_thread = None

    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load refresh configuration."""
        default_config = {
            'refresh_schedule': {
                'daily_update': '01:00',  # 1 AM daily
                'weekly_quality_check': 'sunday 02:00',
                'monthly_cleanup': '1st 03:00',
                'health_check': 'monday 04:00'
            },
            'update_settings': {
                'stale_threshold_days': 7,
                'max_workers': 4,
                'batch_size': 25,
                'retry_attempts': 3
            },
            'quality_settings': {
                'max_missing_data_pct': 5,
                'max_gap_days': 7,
                'auto_repair': True
            },
            'retention_settings': {
                'keep_years': 5,
                'cleanup_threshold_days': 30
            },
            'notifications': {
                'enabled': False,
                'email_on_failure': True,
                'log_level': 'INFO'
            }
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with default config
                    for key, value in user_config.items():
                        if key in default_config and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.error(f"Error loading config file {config_file}: {e}")
                self.logger.info("Using default configuration")

        return default_config

    def save_config(self, config_file: str = None):
        """Save current configuration to file."""
        if config_file is None:
            config_file = self.base_path / "metadata" / "refresh_config.json"

        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        self.logger.info(f"Configuration saved to {config_path}")

    def schedule_jobs(self):
        """Schedule all automated jobs."""
        config = self.config['refresh_schedule']

        # Daily market data update
        schedule.every().day.at(config['daily_update']).do(self.daily_update_job)

        # Weekly quality check
        schedule.every().sunday.at(config['weekly_quality_check']).do(self.weekly_quality_check)

        # Monthly cleanup
        schedule.every().day.at(config['monthly_cleanup']).do(self.monthly_cleanup_job)

        # Health check
        schedule.every().monday.at(config['health_check']).do(self.health_check_job)

        self.logger.info("Scheduled automated data refresh jobs:")
        self.logger.info(f"  Daily update: {config['daily_update']}")
        self.logger.info(f"  Weekly quality check: {config['weekly_quality_check']}")
        self.logger.info(f"  Monthly cleanup: {config['monthly_cleanup']}")
        self.logger.info(f"  Health check: {config['health_check']}")

    def daily_update_job(self):
        """Execute daily data update job."""
        self.logger.info("Starting daily data update job")

        try:
            settings = self.config['update_settings']
            result = self.data_manager.update_stale_data(
                days_threshold=settings['stale_threshold_days'],
                max_workers=settings['max_workers']
            )

            self.logger.info(f"Daily update completed: {result['symbols_updated']} symbols updated")

            if self.config['notifications']['enabled'] and result['symbols_updated'] == 0:
                self.logger.info("No symbols required updating")

        except Exception as e:
            self.logger.error(f"Daily update job failed: {e}")
            if self.config['notifications']['email_on_failure']:
                self._send_notification("Daily Update Failed", str(e))

    def weekly_quality_check(self):
        """Execute weekly quality check job."""
        self.logger.info("Starting weekly quality check")

        try:
            quality_report = self.data_manager.check_data_quality()

            self.logger.info(f"Weekly quality check completed:")
            self.logger.info(f"  Overall health score: {quality_report['overall_health_score']:.1f}%")
            self.logger.info(f"  Issues found: {len(quality_report['data_quality_issues'])}")

            if self.config['quality_settings']['auto_repair'] and quality_report['data_quality_issues']:
                self.logger.info("Starting auto-repair of data issues")
                repair_result = self.data_manager.repair_data_issues(dry_run=False)
                self.logger.info(f"Auto-repair completed: {repair_result['repairs_scheduled']} repairs made")

        except Exception as e:
            self.logger.error(f"Weekly quality check failed: {e}")
            if self.config['notifications']['email_on_failure']:
                self._send_notification("Weekly Quality Check Failed", str(e))

    def monthly_cleanup_job(self):
        """Execute monthly cleanup job."""
        # Check if today is the first day of the month
        if datetime.now().day != 1:
            return

        self.logger.info("Starting monthly cleanup job")

        try:
            settings = self.config['retention_settings']
            result = self.storage.cleanup_old_data(
                keep_years=settings['keep_years'],
                dry_run=False
            )

            self.logger.info(f"Monthly cleanup completed:")
            self.logger.info(f"  Files removed: {result['files_removed']}")
            self.logger.info(f"  Space freed: {result['total_size_freed_mb']:.2f} MB")

            # Cleanup old versions
            self.storage.cleanup_old_versions(keep_versions=2)

        except Exception as e:
            self.logger.error(f"Monthly cleanup job failed: {e}")
            if self.config['notifications']['email_on_failure']:
                self._send_notification("Monthly Cleanup Failed", str(e))

    def health_check_job(self):
        """Execute comprehensive health check."""
        self.logger.info("Starting comprehensive health check")

        try:
            health_report = self.data_manager.run_health_check()

            self.logger.info(f"Health check completed:")
            self.logger.info(f"  Overall status: {health_report['overall_status']}")
            self.logger.info(f"  Health score: {health_report['health_score']:.1f}%")

            if health_report['recommendations']:
                self.logger.info("Recommendations:")
                for rec in health_report['recommendations']:
                    self.logger.info(f"  - {rec}")

            # Save health report
            report_file = self.base_path / "metadata" / f"health_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(health_report, f, indent=2, default=str)

            self.logger.info(f"Health report saved to {report_file}")

        except Exception as e:
            self.logger.error(f"Health check job failed: {e}")
            if self.config['notifications']['email_on_failure']:
                self._send_notification("Health Check Failed", str(e))

    def _send_notification(self, subject: str, message: str):
        """Send notification (placeholder implementation)."""
        self.logger.info(f"NOTIFICATION: {subject} - {message}")
        # In a real implementation, this would send email or other notifications
        pass

    def _scheduler_loop(self):
        """Main scheduler loop running in separate thread."""
        self.logger.info("Starting scheduler loop")
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def start(self):
        """Start the automated refresh system."""
        if self.running:
            self.logger.warning("Refresh system is already running")
            return

        self.logger.info("Starting automated data refresh system")
        self.running = True

        # Schedule jobs
        self.schedule_jobs()

        # Start scheduler in separate thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        self.logger.info("Automated data refresh system started")

    def stop(self):
        """Stop the automated refresh system."""
        if not self.running:
            self.logger.warning("Refresh system is not running")
            return

        self.logger.info("Stopping automated data refresh system")
        self.running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        schedule.clear()
        self.logger.info("Automated data refresh system stopped")

    def run_once(self, job_type: str = "daily_update"):
        """Run a specific job once without scheduling."""
        self.logger.info(f"Running {job_type} job once")

        if job_type == "daily_update":
            self.daily_update_job()
        elif job_type == "quality_check":
            self.weekly_quality_check()
        elif job_type == "cleanup":
            self.monthly_cleanup_job()
        elif job_type == "health_check":
            self.health_check_job()
        else:
            self.logger.error(f"Unknown job type: {job_type}")

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        next_runs = {}
        for job in schedule.jobs:
            next_runs[job.job_func.__name__] = job.next_run

        return {
            'running': self.running,
            'scheduled_jobs': len(schedule.jobs),
            'next_runs': next_runs,
            'config': self.config,
            'storage_summary': self.data_manager.get_storage_summary()
        }


def main():
    """Main function for automated data refresh CLI."""
    parser = argparse.ArgumentParser(description='Automated data refresh system')
    parser.add_argument('--action', type=str, default='start',
                       choices=['start', 'stop', 'status', 'run-once'],
                       help='Action to perform')
    parser.add_argument('--job-type', type=str, default='daily_update',
                       choices=['daily_update', 'quality_check', 'cleanup', 'health_check'],
                       help='Job type for run-once action')
    parser.add_argument('--base-path', type=str, default='data/storage',
                       help='Base path for data storage')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str,
                       help='Log file path')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    try:
        # Initialize refresh system
        refresh_system = AutomatedDataRefresh(args.base_path, args.config)

        if args.action == 'start':
            refresh_system.start()
            logger.info("Refresh system started. Press Ctrl+C to stop.")

            # Keep running until interrupted
            try:
                while True:
                    time.sleep(60)
                    # Print status every hour
                    if datetime.now().minute == 0:
                        status = refresh_system.get_status()
                        logger.info(f"System running - {status['scheduled_jobs']} jobs scheduled")

            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                refresh_system.stop()

        elif args.action == 'stop':
            refresh_system.stop()
            logger.info("Refresh system stopped")

        elif args.action == 'status':
            status = refresh_system.get_status()
            print(f"Status: {'Running' if status['running'] else 'Stopped'}")
            print(f"Scheduled jobs: {status['scheduled_jobs']}")
            if status['next_runs']:
                print("Next scheduled runs:")
                for job, next_time in status['next_runs'].items():
                    print(f"  {job}: {next_time}")

        elif args.action == 'run-once':
            refresh_system.run_once(args.job_type)
            logger.info(f"Completed {args.job_type} job")

        return 0

    except Exception as e:
        logger.error(f"Error in automated refresh system: {e}")
        return 1


if __name__ == "__main__":
    exit(main())