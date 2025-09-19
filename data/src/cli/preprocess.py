"""
CLI Preprocess Command

Command-line interface for preprocessing financial market data
with various configuration options and output formats.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from ..preprocessing import PreprocessingOrchestrator
from ..config.pipeline_config import PipelineConfigManager, PreprocessingConfig


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
        log_file: Optional log file path
    """
    level = logging.DEBUG if verbose else logging.INFO
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format,
        handlers=handlers
    )


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Preprocess financial market data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess CSV file with default pipeline
  python -m data.src.cli.preprocess input.csv --pipeline default

  # Create custom equity pipeline
  python -m data.src.cli.preprocess --create-pipeline equity_pipeline \\
    --description "Equity data processing" \\
    --asset-classes equity \\
    --output-dir ./pipelines

  # Preprocess with custom output
  python -m data.src.cli.preprocess input.parquet \\
    --pipeline equity_pipeline \\
    --output-dir ./processed_data \\
    --format parquet \\
    --verbose

  # List available pipelines
  python -m data.src.cli.preprocess --list-pipelines

  # Validate pipeline configuration
  python -m data.src.cli.preprocess --validate-pipeline equity_pipeline
        """
    )

    # Input/Output options
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input data file (CSV, Parquet, or JSON)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./processed_data',
        help='Output directory for processed data (default: ./processed_data)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['parquet', 'csv', 'json'],
        default='parquet',
        help='Output format (default: parquet)'
    )

    # Pipeline options
    parser.add_argument(
        '--pipeline', '-p',
        type=str,
        default='default',
        help='Pipeline configuration to use (default: default)'
    )
    parser.add_argument(
        '--create-pipeline',
        type=str,
        help='Create a new pipeline with the specified ID'
    )
    parser.add_argument(
        '--description',
        type=str,
        help='Description for new pipeline'
    )
    parser.add_argument(
        '--asset-classes',
        nargs='+',
        choices=['equity', 'fx', 'bond', 'commodity', 'crypto', 'all'],
        help='Asset classes for new pipeline'
    )

    # Configuration options
    parser.add_argument(
        '--config-dir',
        type=str,
        help='Directory for pipeline configurations'
    )
    parser.add_argument(
        '--list-pipelines',
        action='store_true',
        help='List available pipelines'
    )
    parser.add_argument(
        '--validate-pipeline',
        type=str,
        help='Validate a specific pipeline configuration'
    )

    # Processing options
    parser.add_argument(
        '--rules-file',
        type=str,
        help='JSON file containing preprocessing rules'
    )
    parser.add_argument(
        '--quality-thresholds',
        type=str,
        help='JSON file containing quality thresholds'
    )

    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only show summary statistics'
    )

    return parser


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        sys.exit(1)


def create_pipeline_config(args) -> PreprocessingConfig:
    """Create a new pipeline configuration.

    Args:
        args: Parsed command-line arguments

    Returns:
        Created PreprocessingConfig
    """
    config_manager = PipelineConfigManager(args.config_dir)

    # Load custom rules if provided
    rules = None
    if args.rules_file:
        rules_data = load_json_file(args.rules_file)
        rules = rules_data.get('rules', [])

    # Load custom quality thresholds if provided
    quality_thresholds = None
    if args.quality_thresholds:
        thresholds_data = load_json_file(args.quality_thresholds)
        quality_thresholds = thresholds_data.get('quality_thresholds')

    # Create pipeline
    config = config_manager.create_default_config(
        pipeline_id=args.create_pipeline,
        description=args.description or f"Pipeline created via CLI",
        asset_classes=args.asset_classes or ['equity'],
        rules=rules,
        quality_thresholds=quality_thresholds
    )

    # Validate and save
    validation = config_manager.validate_config(config)
    if not validation['is_valid']:
        print("Pipeline configuration validation failed:")
        for error in validation['errors']:
            print(f"  - {error}")
        sys.exit(1)

    config_path = config_manager.save_config(config)
    print(f"Pipeline '{args.create_pipeline}' created successfully: {config_path}")

    return config


def list_pipelines(config_manager: PipelineConfigManager):
    """List available pipelines.

    Args:
        config_manager: Configuration manager instance
    """
    pipelines = config_manager.list_pipelines()

    if not pipelines:
        print("No pipelines found.")
        return

    print("Available pipelines:")
    print("-" * 80)
    for pipeline in pipelines:
        print(f"ID: {pipeline['pipeline_id']}")
        print(f"Description: {pipeline['description']}")
        print(f"Asset Classes: {', '.join(pipeline['asset_classes'])}")
        print(f"Rules: {pipeline['rules_count']}")
        print(f"Version: {pipeline['version']}")
        print(f"Created: {pipeline['created_at']}")
        print("-" * 80)


def validate_pipeline(pipeline_id: str, config_manager: PipelineConfigManager):
    """Validate a pipeline configuration.

    Args:
        pipeline_id: Pipeline identifier
        config_manager: Configuration manager instance
    """
    validation = config_manager.validate_config(pipeline_id)

    if validation['is_valid']:
        print(f"Pipeline '{pipeline_id}' is valid ✓")
    else:
        print(f"Pipeline '{pipeline_id}' has validation issues ✗")
        print("Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
        if validation.get('warnings'):
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")


def preprocess_data(args, orchestrator: PreprocessingOrchestrator) -> Dict[str, Any]:
    """Preprocess data file.

    Args:
        args: Parsed command-line arguments
        orchestrator: Preprocessing orchestrator instance

    Returns:
        Processing results
    """
    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    # Validate pipeline exists
    pipeline_summary = None
    try:
        pipeline_summary = orchestrator.config_manager.get_config_summary(args.pipeline)
    except:
        print(f"Pipeline '{args.pipeline}' not found")
        print("Use --list-pipelines to see available pipelines")
        sys.exit(1)

    print(f"Processing with pipeline: {args.pipeline}")
    print(f"Description: {pipeline_summary['description']}")
    print(f"Asset classes: {', '.join(pipeline_summary['asset_classes'])}")

    if args.dry_run:
        print("\nDry run - would process:")
        print(f"  Input: {input_path}")
        print(f"  Output: {args.output_dir}")
        print(f"  Format: {args.format}")
        print(f"  Pipeline: {args.pipeline}")
        return {'dry_run': True}

    # Process the data
    print(f"\nProcessing {input_path}...")
    results = orchestrator.preprocess_from_file(
        input_path=str(input_path),
        pipeline_id=args.pipeline,
        output_path=args.output_dir
    )

    return results


def display_results(results: Dict[str, Any], args):
    """Display processing results.

    Args:
        results: Processing results
        args: Parsed command-line arguments
    """
    if args.dry_run:
        return

    if not results.get('success', False):
        print(f"Processing failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

    print("\n" + "="*60)
    print("PROCESSING RESULTS")
    print("="*60)

    print(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
    print(f"Session ID: {results['session_id']}")
    print(f"Dataset ID: {results['dataset_id']}")

    if 'original_shape' in results:
        print(f"Original shape: {results['original_shape']}")
        print(f"Final shape: {results['final_shape']}")

    print(f"Quality score: {results['quality_score']:.3f}")
    print(f"Processed data count: {results['processed_data_count']}")
    print(f"Execution time: {results['execution_time']:.2f} seconds")

    if results.get('output_path'):
        print(f"Output saved to: {results['output_path']}")

    # Show quality metrics if not summary only
    if not args.summary_only and 'quality_report' in results:
        quality_report = results['quality_report']
        print(f"\nQuality Metrics:")
        print(f"  Overall Score: {quality_report['overall_score']:.3f}")

        for metric in quality_report['metrics']:
            status_icon = "✓" if metric['status'] == 'pass' else "⚠" if metric['status'] == 'warn' else "✗"
            print(f"  {status_icon} {metric['metric_type']}: {metric['value']:.3f} (threshold: {metric['threshold']:.3f})")

    # Show recommendations if any
    if results.get('recommendations'):
        print(f"\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  - {rec}")

    # Show processing summary
    if 'processing_logs' in results and not args.summary_only:
        logs = results['processing_logs']
        successful_ops = sum(1 for log in logs if log['success'])
        print(f"\nProcessing Summary:")
        print(f"  Total operations: {len(logs)}")
        print(f"  Successful: {successful_ops}")
        print(f"  Failed: {len(logs) - successful_ops}")

        # Show failed operations
        failed_ops = [log for log in logs if not log['success']]
        if failed_ops:
            print(f"\nFailed Operations:")
            for op in failed_ops:
                print(f"  - {op['operation']}: {op['error_message']}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose, args.log_file)

    # Initialize components
    config_manager = PipelineConfigManager(args.config_dir)
    orchestrator = PreprocessingOrchestrator(config_manager)

    try:
        # Handle pipeline creation
        if args.create_pipeline:
            create_pipeline_config(args)
            return

        # Handle pipeline listing
        if args.list_pipelines:
            list_pipelines(config_manager)
            return

        # Handle pipeline validation
        if args.validate_pipeline:
            validate_pipeline(args.validate_pipeline, config_manager)
            return

        # Handle data processing
        if args.input_file:
            results = preprocess_data(args, orchestrator)
            display_results(results, args)
        else:
            print("No input file specified. Use --help for usage information.")
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()