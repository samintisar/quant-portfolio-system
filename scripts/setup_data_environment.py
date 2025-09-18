"""
Setup script for data ingestion and storage environment.

This script sets up the necessary directories and validates the
environment for data ingestion and storage operations.
"""

import os
import sys
from pathlib import Path
import subprocess
import logging


def setup_logging():
    """Setup logging for setup operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_directory_structure():
    """Create the necessary directory structure."""
    logger = logging.getLogger(__name__)

    directories = [
        "data/storage/raw/equity",
        "data/storage/raw/etf",
        "data/storage/raw/fx",
        "data/storage/raw/bond",
        "data/storage/raw/commodity",
        "data/storage/raw/index",
        "data/storage/processed",
        "data/storage/metadata",
        "data/storage/temp",
        "data/logs",
        "data/cache",
        "output/results",
        "output/plots",
        "output/reports"
    ]

    logger.info("Creating directory structure...")

    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✓ Created: {directory}")

    return True


def check_python_dependencies():
    """Check if required Python packages are installed."""
    logger = logging.getLogger(__name__)

    required_packages = [
        'pandas',
        'numpy',
        'yfinance',
        'scipy',
        'aiohttp',
        'requests'
    ]

    missing_packages = []

    logger.info("Checking Python dependencies...")

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"  ✗ {package} (missing)")

    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install missing packages with: pip install -r docs/requirements.txt")
        return False

    logger.info("All required packages are installed")
    return True


def check_optional_dependencies():
    """Check optional dependencies for enhanced functionality."""
    logger = logging.getLogger(__name__)

    optional_packages = {
        'pyarrow': 'Parquet file format support',
        'tables': 'HDF5 file format support',
        'pyarrow.parquet': 'Feather file format support',
        'matplotlib': 'Plotting capabilities',
        'plotly': 'Interactive plotting',
        'seaborn': 'Statistical plotting'
    }

    logger.info("Checking optional dependencies...")

    available_optional = []
    missing_optional = []

    for package, description in optional_packages.items():
        try:
            __import__(package)
            available_optional.append(package)
            logger.info(f"  ✓ {package} - {description}")
        except ImportError:
            missing_optional.append(package)
            logger.info(f"  - {package} - {description} (optional)")

    if missing_optional:
        logger.info(f"Optional packages not installed: {missing_optional}")
        logger.info("These are optional - system will work without them")

    return available_optional, missing_optional


def validate_project_structure():
    """Validate that the project structure is correct."""
    logger = logging.getLogger(__name__)

    required_files = [
        "CLAUDE.md",
        "README.md",
        "data/src/feeds/__init__.py",
        "data/src/feeds/yahoo_finance_ingestion.py",
        "data/src/feeds/data_ingestion_interface.py",
        "data/src/storage/__init__.py",
        "data/src/storage/market_data_storage.py",
        "examples/basic_usage.py",
        "scripts/demo_data_ingestion_and_storage.py",
        "docs/requirements.txt"
    ]

    logger.info("Validating project structure...")

    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"  ✓ {file_path}")
        else:
            missing_files.append(file_path)
            logger.warning(f"  ✗ {file_path}")

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False

    logger.info("Project structure is valid")
    return True


def create_environment_config():
    """Create environment configuration file."""
    logger = logging.getLogger(__name__)

    config_content = """
# Data Ingestion Environment Configuration
# This file contains configuration settings for the data ingestion system

[storage]
default_path = data/storage
default_format = parquet
compression = none
enable_versioning = true
max_file_size_mb = 1000

[ingestion]
max_workers = 5
rate_limit_seconds = 0.1
timeout_seconds = 30
retry_attempts = 3

[logging]
level = INFO
log_file = data/logs/ingestion.log
max_log_size_mb = 10
backup_count = 5

[yahoo_finance]
user_agent = mozilla/5.0
proxy_enabled = false
timeout_seconds = 10

[validation]
enable_strict_validation = true
max_missing_data_pct = 10
max_outlier_std = 3
enable_price_validation = true
enable_returns_validation = true
"""

    config_file = Path("config/.env")
    config_file.parent.mkdir(exist_ok=True)

    try:
        with open(config_file, 'w') as f:
            f.write(config_content.strip())
        logger.info(f"Created configuration file: {config_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create config file: {e}")
        return False


def create_init_files():
    """Create __init__.py files for proper Python package structure."""
    logger = logging.getLogger(__name__)

    init_directories = [
        "data",
        "data/src",
        "data/src/feeds",
        "data/src/storage",
        "data/src/processing",
        "data/src/models",
        "scripts",
        "examples",
        "output"
    ]

    logger.info("Creating __init__.py files...")

    for directory in init_directories:
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            try:
                init_file.touch()
                logger.info(f"  ✓ {init_file}")
            except Exception as e:
                logger.warning(f"  ✗ Failed to create {init_file}: {e}")

    return True


def run_validation_tests():
    """Run basic validation tests to ensure system works."""
    logger = logging.getLogger(__name__)

    logger.info("Running validation tests...")

    try:
        # Test imports
        sys.path.append(str(Path.cwd()))

        from data.src.feeds import create_default_ingestion_system, AssetClass
        from data.src.storage import create_default_storage

        logger.info("  ✓ Import tests passed")

        # Test basic functionality
        ingestion = create_default_ingestion_system()
        storage = create_default_storage()

        logger.info("  ✓ System initialization tests passed")

        # Test storage info
        storage_info = storage.get_storage_info()
        logger.info(f"  ✓ Storage system ready at: {storage_info['base_path']}")

        return True

    except Exception as e:
        logger.error(f"Validation tests failed: {e}")
        return False


def main():
    """Main setup function."""
    logger = setup_logging()

    print("Data Ingestion and Storage Environment Setup")
    print("=" * 50)

    success = True

    # Create directory structure
    if not create_directory_structure():
        success = False

    # Check dependencies
    if not check_python_dependencies():
        success = False

    # Check optional dependencies
    check_optional_dependencies()

    # Validate project structure
    if not validate_project_structure():
        success = False

    # Create configuration
    if not create_environment_config():
        success = False

    # Create init files
    if not create_init_files():
        success = False

    # Run validation tests
    if not run_validation_tests():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✅ Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r docs/requirements.txt")
        print("2. Run basic example: python examples/basic_usage.py")
        print("3. Run full demo: python scripts/demo_data_ingestion_and_storage.py")
    else:
        print("❌ Environment setup had issues")
        print("Please check the logs above and fix any problems")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)