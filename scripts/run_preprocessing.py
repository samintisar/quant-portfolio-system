"""
Preprocessing Pipeline Configuration for Financial Data

This script creates and executes a preprocessing pipeline for the financial market data.
"""

import sys
sys.path.append('C:\\Users\\samin\\Desktop\\Github\\quant-portfolio-system')

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Import preprocessing components
from data.src.preprocessing import PreprocessingOrchestrator
from data.src.config.pipeline_config import PipelineConfigManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_equity_preprocessing_config():
    """Create preprocessing configuration for equity data."""

    # Define preprocessing rules for financial data
    rules = [
        {
            'rule_type': 'missing_value_handling',
            'parameters': {
                'method': 'forward_fill',
                'threshold': 0.05,  # 5% threshold for missing values
                'window_size': 5
            },
            'priority': 1
        },
        {
            'rule_type': 'outlier_detection',
            'parameters': {
                'method': 'iqr',
                'threshold': 2.5,  # 2.5 IQR for financial data
                'action': 'winsorize'  # Winsorize outliers instead of removing
            },
            'priority': 2
        },
        {
            'rule_type': 'validation',
            'parameters': {
                'check_negative_prices': True,
                'check_zero_volume': True,
                'check_extreme_returns': True,
                'return_threshold': 0.5  # 50% daily return threshold
            },
            'priority': 3
        },
        {
            'rule_type': 'normalization',
            'parameters': {
                'method': 'robust',  # Robust scaling for financial data
                'preserve_stats': True,
                'columns_to_normalize': ['open', 'high', 'low', 'close', 'volume']
            },
            'priority': 4
        }
    ]

    # Quality thresholds
    quality_thresholds = {
        'completeness': 0.98,      # 98% complete data
        'consistency': 0.95,      # 95% consistent
        'accuracy': 0.90,         # 90% accurate
        'timeliness': 0.99,       # 99% timely
        'uniqueness': 0.99,       # 99% unique records
        'min_data_points': 1000,  # Minimum data points
        'max_outlier_ratio': 0.05 # Maximum 5% outliers
    }

    return {
        'pipeline_id': 'equity_preprocessing_v1',
        'description': 'Comprehensive preprocessing pipeline for equity data with outlier handling and normalization',
        'asset_classes': ['equity', 'etf'],
        'rules': rules,
        'quality_thresholds': quality_thresholds,
        'output_format': 'parquet',
        'version': '1.0.0',
        'metadata': {
            'created_by': 'quant_system',
            'purpose': 'portfolio_optimization',
            'data_sources': ['yahoo_finance'],
            'expected_performance': '<30_seconds_for_1M_records'
        }
    }

def load_sample_equity_data():
    """Load sample equity data for preprocessing."""

    # Paths to sample data files
    data_dir = Path('C:\\Users\\samin\\Desktop\\Github\\quant-portfolio-system\\data\\storage\\raw\\equity')

    # Load a few sample equity files
    sample_files = [
        'AAPL_20200921_to_20250917.parquet',
        'MSFT_20200921_to_20250917.parquet',
        'SPY_20200921_to_20250917.parquet'
    ]

    data_frames = []

    for file_name in sample_files:
        file_path = data_dir / file_name
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                # Add symbol column for identification
                df['symbol'] = file_name.split('_')[0]
                df['asset_class'] = 'equity' if file_name.split('_')[0] in ['AAPL', 'MSFT'] else 'etf'
                # Add timestamp column if not present
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.to_datetime(df.index)
                data_frames.append(df)
                logger.info(f"Loaded {file_name}: {df.shape[0]} rows")
            except Exception as e:
                logger.error(f"Error loading {file_name}: {e}")

    if not data_frames:
        raise ValueError("No data files could be loaded")

    # Combine all data
    combined_df = pd.concat(data_frames, ignore_index=True)
    logger.info(f"Combined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")

    return combined_df

def run_preprocessing_pipeline():
    """Run the complete preprocessing pipeline."""

    try:
        # Initialize orchestrator
        orchestrator = PreprocessingOrchestrator()
        config_manager = PipelineConfigManager()

        logger.info("=== Starting Financial Data Preprocessing ===")

        # Step 1: Create configuration
        logger.info("Step 1: Creating preprocessing configuration...")
        config_data = create_equity_preprocessing_config()

        # Create configuration object directly
        from data.src.config.pipeline_config import PreprocessingConfig
        config = PreprocessingConfig.from_dict(config_data)
        config_manager._configs[config.pipeline_id] = config

        # Save configuration
        config_path = config_manager.save_config(config)
        logger.info(f"Configuration saved to: {config_path}")

        # Step 2: Load data
        logger.info("Step 2: Loading sample equity data...")
        input_data = load_sample_equity_data()

        # Step 3: Run preprocessing
        logger.info("Step 3: Running preprocessing pipeline...")
        output_path = "C:\\Users\\samin\\Desktop\\Github\\quant-portfolio-system\\data\\storage\\processed\\equity_preprocessed"

        results = orchestrator.preprocess_data(
            input_data=input_data,
            pipeline_id='equity_preprocessing_v1',
            output_path=output_path,
            enable_versioning=True
        )

        # Step 4: Display results
        logger.info("=== Preprocessing Results ===")
        logger.info(f"Success: {results['success']}")
        logger.info(f"Session ID: {results['session_id']}")
        logger.info(f"Original shape: {results['original_shape']}")
        logger.info(f"Final shape: {results['final_shape']}")
        logger.info(f"Quality score: {results['quality_score']:.3f}")
        logger.info(f"Execution time: {results['execution_time']:.2f} seconds")
        logger.info(f"Processed data count: {results['processed_data_count']}")

        if results['success']:
            logger.info(f"Output saved to: {output_path}")
            logger.info(f"Input version ID: {results.get('input_version_id', 'N/A')}")
            logger.info(f"Output version ID: {results.get('output_version_id', 'N/A')}")
        else:
            logger.error(f"Preprocessing failed: {results.get('error', 'Unknown error')}")

        # Step 5: Display quality metrics
        if 'quality_report' in results:
            qr = results['quality_report']
            logger.info("=== Quality Metrics ===")
            logger.info(f"Completeness: {qr.get('completeness_score', 0):.3f}")
            logger.info(f"Consistency: {qr.get('consistency_score', 0):.3f}")
            logger.info(f"Accuracy: {qr.get('accuracy_score', 0):.3f}")
            logger.info(f"Overall quality: {qr.get('overall_score', 0):.3f}")

        return results

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Run the preprocessing pipeline
    results = run_preprocessing_pipeline()

    # Print summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
    print(f"Data points processed: {results['original_shape'][0]:,}")
    print(f"Processing time: {results['execution_time']:.2f} seconds")
    print(f"Quality score: {results['quality_score']:.3f}")

    if results['success']:
        print(f"Output location: {results['output_path']}")
        print(f"Version controlled: Yes")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")