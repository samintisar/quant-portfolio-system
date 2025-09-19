"""
CLI Quality Report Command

Command-line interface for generating and viewing data quality reports
with various analysis options and output formats.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

from ..services.quality_service import QualityService
from ..config.pipeline_config import PipelineConfigManager
from ..preprocessing import PreprocessingOrchestrator


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
        description='Generate and analyze data quality reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate quality report for processed data
  python -m data.src.cli.quality_report processed_data.parquet \\
    --dataset-id my_dataset --output-dir ./reports

  # Compare quality metrics over time
  python -m data.src.cli.quality_report --compare-trends \\
    --dataset-id my_dataset --days 30

  # Generate detailed report with all metrics
  python -m data.src.cli.quality_report data.csv \\
    --detailed --format json --output report.json

  # Show quality summary only
  python -m data.src.cli.quality_report processed_data/ \\
    --summary-only

  # Export quality metrics to CSV
  python -m data.src.cli.quality_report data.parquet \\
    --export-csv quality_metrics.csv
        """
    )

    # Input options
    parser.add_argument(
        'input_data',
        nargs='?',
        help='Input data file or directory (CSV, Parquet, JSON)'
    )

    # Dataset identification
    parser.add_argument(
        '--dataset-id', '-d',
        type=str,
        default='default_dataset',
        help='Dataset identifier (default: default_dataset)'
    )

    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./quality_reports',
        help='Output directory for reports (default: ./quality_reports)'
    )
    parser.add_argument(
        '--output-file', '-f',
        type=str,
        help='Output file name (default: auto-generated)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'html', 'text', 'csv'],
        default='text',
        help='Output format (default: text)'
    )

    # Analysis options
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Generate detailed report with all metrics'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary statistics'
    )
    parser.add_argument(
        '--compare-trends',
        action='store_true',
        help='Analyze quality trends over time'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to analyze for trends (default: 30)'
    )

    # Export options
    parser.add_argument(
        '--export-csv',
        type=str,
        help='Export metrics to CSV file'
    )
    parser.add_argument(
        '--export-json',
        type=str,
        help='Export report to JSON file'
    )

    # Filtering options
    parser.add_argument(
        '--metric-types',
        nargs='+',
        choices=['completeness', 'consistency', 'accuracy', 'timeliness', 'uniqueness', 'validity', 'all'],
        default=['all'],
        help='Metric types to include (default: all)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='Custom threshold for quality assessment'
    )

    # Configuration options
    parser.add_argument(
        '--config-dir',
        type=str,
        help='Directory for quality service configuration'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Directory for caching quality metrics'
    )

    # Display options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output except errors'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )

    return parser


def load_input_data(input_path: str) -> pd.DataFrame:
    """Load input data from file.

    Args:
        input_path: Path to input file

    Returns:
        Loaded DataFrame
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        if input_path.is_dir():
            # Look for processed data files
            parquet_files = list(input_path.glob("*.parquet"))
            csv_files = list(input_path.glob("*.csv"))
            json_files = list(input_path.glob("*.json"))

            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
            elif csv_files:
                df = pd.read_csv(csv_files[0])
            elif json_files:
                df = pd.read_json(json_files[0])
            else:
                print(f"No supported data files found in {input_path}")
                sys.exit(1)
        else:
            # Single file
            if input_path.suffix.lower() == '.csv':
                df = pd.read_csv(input_path)
            elif input_path.suffix.lower() in ['.parquet', '.pq']:
                df = pd.read_parquet(input_path)
            elif input_path.suffix.lower() == '.json':
                df = pd.read_json(input_path)
            else:
                print(f"Unsupported file format: {input_path.suffix}")
                sys.exit(1)

        return df

    except Exception as e:
        print(f"Error loading data from {input_path}: {e}")
        sys.exit(1)


def generate_quality_report(df: pd.DataFrame, dataset_id: str, quality_service: QualityService,
                           detailed: bool = False) -> Dict[str, Any]:
    """Generate quality report for dataset.

    Args:
        df: Input DataFrame
        dataset_id: Dataset identifier
        quality_service: Quality service instance
        detailed: Generate detailed report

    Returns:
        Quality report dictionary
    """
    print(f"Generating quality report for dataset: {dataset_id}")

    # Calculate quality metrics
    report = quality_service.calculate_all_metrics(df, dataset_id)

    # Add detailed analysis if requested
    if detailed:
        detailed_analysis = {}

        # Column-wise analysis
        column_analysis = {}
        for col in df.columns:
            col_data = df[col]
            col_analysis[col] = {
                'data_type': str(col_data.dtype),
                'null_count': int(col_data.isnull().sum()),
                'null_percentage': float(col_data.isnull().sum() / len(col_data) * 100),
                'unique_count': int(col_data.nunique()),
                'sample_values': col_data.dropna().head(5).tolist()
            }

            # Numeric columns get additional stats
            if pd.api.types.is_numeric_dtype(col_data):
                col_analysis[col].update({
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median())
                })

        detailed_analysis['column_analysis'] = column_analysis

        # Data shape and size info
        detailed_analysis['dataset_info'] = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'file_size_estimate': len(df.to_json()) if len(df) < 10000 else 'large'
        }

        report.detailed_analysis = detailed_analysis

    return report


def display_text_report(report: Dict[str, Any], args):
    """Display quality report in text format.

    Args:
        report: Quality report dictionary
        args: Command-line arguments
    """
    color = not args.no_color

    def color_text(text, color_code):
        return f"\033[{color_code}m{text}\033[0m" if color else text

    print("\n" + "="*60)
    print(color_text("DATA QUALITY REPORT", "1;36"))
    print("="*60)

    # Basic info
    print(f"\nDataset ID: {report['dataset_id']}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Overall Quality Score: {color_text(f'{report[\"overall_score\"]:.3f}', '1;32' if report['overall_score'] >= 0.8 else '1;33' if report['overall_score'] >= 0.6 else '1;31')}")

    # Metrics summary
    print(f"\nQuality Metrics:")
    for metric in report['metrics']:
        status = metric['status']
        if status == 'pass':
            status_icon = color_text("✓", "32")
        elif status == 'warn':
            status_icon = color_text("⚠", "33")
        else:
            status_icon = color_text("✗", "31")

        value = metric['value']
        threshold = metric['threshold']
        metric_type = metric['metric_type'].title()

        print(f"  {status_icon} {metric_type}: {value:.3f} (threshold: {threshold:.3f})")

    # Show failing metrics
    failing_metrics = [m for m in report['metrics'] if m['status'] == 'fail']
    if failing_metrics:
        print(f"\n{color_text('Failing Metrics:', '1;31')}")
        for metric in failing_metrics:
            print(f"  • {metric['metric_type']}: {metric['value']:.3f} < {metric['threshold']:.3f}")

    # Show warning metrics
    warning_metrics = [m for m in report['metrics'] if m['status'] == 'warn']
    if warning_metrics:
        print(f"\n{color_text('Warning Metrics:', '1;33')}")
        for metric in warning_metrics:
            print(f"  • {metric['metric_type']}: {metric['value']:.3f} ≈ {metric['threshold']:.3f}")

    # Detailed analysis if available
    if hasattr(report, 'detailed_analysis') and not args.summary_only:
        detailed = report.detailed_analysis

        if 'dataset_info' in detailed:
            info = detailed['dataset_info']
            print(f"\nDataset Information:")
            print(f"  Shape: {info['shape']}")
            print(f"  Memory Usage: {info['memory_usage']:,} bytes")

        if 'column_analysis' in detailed:
            print(f"\nColumn Analysis:")
            for col, analysis in detailed['column_analysis'].items():
                print(f"  {col}:")
                print(f"    Type: {analysis['data_type']}")
                print(f"    Null Count: {analysis['null_count']} ({analysis['null_percentage']:.1f}%)")
                print(f"    Unique Values: {analysis['unique_count']}")

                if 'mean' in analysis:
                    print(f"    Mean: {analysis['mean']:.3f}")
                    print(f"    Std: {analysis['std']:.3f}")
                    print(f"    Range: [{analysis['min']:.3f}, {analysis['max']:.3f}]")

    # Recommendations
    print(f"\nRecommendations:")
    recommendations = generate_recommendations(report)
    for rec in recommendations:
        print(f"  • {rec}")


def generate_recommendations(report: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on quality report.

    Args:
        report: Quality report dictionary

    Returns:
        List of recommendations
    """
    recommendations = []

    for metric in report['metrics']:
        metric_type = metric['metric_type']
        value = metric['value']
        status = metric['status']

        if status == 'fail':
            if metric_type == 'completeness':
                recommendations.append("Address missing data - consider imputation or data collection improvements")
            elif metric_type == 'consistency':
                recommendations.append("Review data for consistency issues and outliers")
            elif metric_type == 'accuracy':
                recommendations.append("Validate data sources and accuracy of measurements")
            elif metric_type == 'timeliness':
                recommendations.append("Improve data processing pipeline to reduce latency")
            elif metric_type == 'uniqueness':
                recommendations.append("Remove duplicate records and ensure proper data deduplication")
            elif metric_type == 'validity':
                recommendations.append("Validate data types and ranges for all fields")

        elif status == 'warn':
            if metric_type == 'completeness':
                recommendations.append("Monitor missing data levels and consider preventive measures")
            elif metric_type == 'timeliness':
                recommendations.append("Monitor data latency and optimize processing times")

    if not recommendations:
        recommendations.append("Data quality is good - continue monitoring")

    return recommendations


def export_report_csv(report: Dict[str, Any], output_path: str):
    """Export quality report to CSV.

    Args:
        report: Quality report dictionary
        output_path: Output file path
    """
    metrics_data = []
    for metric in report['metrics']:
        metrics_data.append({
            'metric_id': metric['metric_id'],
            'dataset_id': metric['dataset_id'],
            'metric_type': metric['metric_type'],
            'value': metric['value'],
            'threshold': metric['threshold'],
            'status': metric['status'],
            'timestamp': metric['timestamp']
        })

    df = pd.DataFrame(metrics_data)
    df.to_csv(output_path, index=False)
    print(f"Report exported to CSV: {output_path}")


def export_report_json(report: Dict[str, Any], output_path: str):
    """Export quality report to JSON.

    Args:
        report: Quality report dictionary
        output_path: Output file path
    """
    # Convert to dictionary for JSON serialization
    report_dict = {
        'dataset_id': report['dataset_id'],
        'overall_score': report['overall_score'],
        'generated_at': datetime.now().isoformat(),
        'metrics': []
    }

    for metric in report['metrics']:
        report_dict['metrics'].append({
            'metric_id': metric['metric_id'],
            'dataset_id': metric['dataset_id'],
            'metric_type': metric['metric_type'],
            'value': metric['value'],
            'threshold': metric['threshold'],
            'status': metric['status'],
            'timestamp': metric['timestamp'].isoformat()
        })

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)

    print(f"Report exported to JSON: {output_path}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if not args.quiet:
        setup_logging(args.verbose)

    try:
        # Initialize quality service
        quality_service = QualityService()

        # Handle different input scenarios
        if args.input_data:
            # Load data and generate report
            df = load_input_data(args.input_data)

            # Generate report
            report = generate_quality_report(
                df,
                args.dataset_id,
                quality_service,
                args.detailed
            )

            # Display report
            if not args.quiet:
                if args.format == 'text':
                    display_text_report(report, args)
                elif args.format == 'json':
                    print(json.dumps(report.to_dict(), indent=2, default=str))

            # Export if requested
            if args.export_csv:
                export_report_csv(report, args.export_csv)

            if args.export_json:
                export_report_json(report, args.export_json)

            # Save to output directory if specified
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                output_file = args.output_file or f"quality_report_{args.dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
                output_path = output_dir / output_file

                if args.format == 'json':
                    export_report_json(report, str(output_path))
                elif args.format == 'csv':
                    export_report_csv(report, str(output_path))

        else:
            # No input data - show help
            print("No input data specified. Use --help for usage information.")
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()