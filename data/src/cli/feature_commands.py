"""
CLI Commands for Feature Generation

This module provides command-line interfaces for feature generation operations,
including batch processing, quality assessment, and configuration management.

Key Features:
- Command-line interface for feature generation
- Batch processing capabilities
- Quality reporting and assessment
- Configuration management
- Integration with all preprocessing libraries

Author: Claude Code
Date: 2025-09-19
"""

import click
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from ..services.feature_service import FeatureGenerator, FeatureGenerationConfig
from ..services.validation_service import DataValidator
from ..models.financial_instrument import FinancialInstrument, InstrumentType, Currency
from ..models.price_data import PriceData, Frequency, PriceType
from ..models.feature_set import FeatureSet
from ..storage.data_loader import DataLoader
from ..config.pipeline_config import PipelineConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """CLI for financial feature generation"""
    ctx.ensure_object(dict)

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration if provided
    if config:
        ctx.obj['config'] = PipelineConfig.load(config)
    else:
        ctx.obj['config'] = PipelineConfig()


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input file path containing OHLCV data')
@click.option('--symbol', '-s', required=True, help='Financial instrument symbol')
@click.option('--name', '-n', help='Financial instrument name')
@click.option('--type', '-t', default='stock',
              type=click.Choice(['stock', 'etf', 'bond', 'commodity', 'currency', 'crypto']),
              help='Instrument type')
@click.option('--currency', default='USD', help='Instrument currency')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--config-file', type=click.Path(exists=True), help='Feature generation config file')
@click.option('--quality-threshold', type=float, default=0.7, help='Quality threshold (0-1)')
@click.option('--report', type=click.Path(), help='Quality report output path')
@click.pass_context
def generate_features(ctx, input, symbol, name, type, currency, output, config_file,
                    quality_threshold, report):
    """Generate features from OHLCV data"""
    try:
        # Load input data
        loader = DataLoader()
        price_data = loader.load_ohlcv_data(
            input_path=input,
            symbol=symbol,
            name=name or symbol,
            instrument_type=type,
            currency=currency
        )

        # Load feature generation configuration
        if config_file:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            feature_config = FeatureGenerationConfig(**config_dict)
        else:
            feature_config = FeatureGenerationConfig(quality_threshold=quality_threshold)

        # Generate features
        generator = FeatureGenerator(feature_config)
        feature_set = generator.generate_features(price_data)

        # Save output
        if output:
            feature_set.to_csv(output)
            click.echo(f"Features saved to {output}")
        else:
            # Print summary
            click.echo(f"Generated {len(feature_set.features.columns)} features for {symbol}")
            click.echo(f"Quality score: {feature_set.quality_metrics.get('completeness_score', 0):.3f}")

        # Generate quality report
        if report:
            quality_report = _generate_quality_report(feature_set)
            with open(report, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            click.echo(f"Quality report saved to {report}")

    except Exception as e:
        click.echo(f"Error generating features: {str(e)}", err=True)
        logger.error(f"Feature generation failed for {symbol}: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True),
              help='Input directory containing OHLCV files')
@click.option('--output-dir', '-o', required=True, type=click.Path(),
              help='Output directory for feature files')
@click.option('--pattern', '-p', default='*.csv', help='File pattern to match')
@click.option('--config-file', type=click.Path(exists=True), help='Feature generation config file')
@click.option('--parallel', is_flag=True, help='Enable parallel processing')
@click.option('--max-workers', type=int, default=4, help='Maximum parallel workers')
@click.option('--quality-report', type=click.Path(), help='Batch quality report path')
@click.pass_context
def batch_generate(ctx, input_dir, output_dir, pattern, config_file, parallel,
                  max_workers, quality_report):
    """Generate features for multiple files in batch"""
    try:
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            feature_config = FeatureGenerationConfig(**config_dict)
        else:
            feature_config = FeatureGenerationConfig()

        # Find input files
        input_path = Path(input_dir)
        input_files = list(input_path.glob(pattern))

        if not input_files:
            click.echo(f"No files found matching pattern {pattern} in {input_dir}")
            return

        click.echo(f"Found {len(input_files)} files to process")

        # Process files
        generator = FeatureGenerator(feature_config)
        loader = DataLoader()
        results = []

        if parallel and len(input_files) > 1:
            import concurrent.futures
            from functools import partial

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                process_func = partial(_process_single_file, generator, loader, output_dir)
                futures = [executor.submit(process_func, file_path) for file_path in input_files]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing file: {str(e)}")
        else:
            for file_path in input_files:
                result = _process_single_file(generator, loader, output_dir, file_path)
                if result:
                    results.append(result)

        # Generate batch quality report
        if quality_report:
            batch_report = _generate_batch_quality_report(results)
            with open(quality_report, 'w') as f:
                json.dump(batch_report, f, indent=2, default=str)
            click.echo(f"Batch quality report saved to {quality_report}")

        click.echo(f"Successfully processed {len(results)} files")

    except Exception as e:
        click.echo(f"Error in batch processing: {str(e)}", err=True)
        raise click.Abort()


def _process_single_file(generator, loader, output_dir, file_path):
    """Process a single file for feature generation"""
    try:
        # Extract symbol from filename
        symbol = file_path.stem

        # Load data
        price_data = loader.load_ohlcv_data(
            input_path=str(file_path),
            symbol=symbol,
            name=symbol,
            instrument_type='stock'
        )

        # Generate features
        feature_set = generator.generate_features(price_data)

        # Save output
        output_path = Path(output_dir) / f"{symbol}_features.csv"
        feature_set.to_csv(output_path)

        logger.info(f"Processed {symbol}: {len(feature_set.features.columns)} features")

        return {
            'symbol': symbol,
            'feature_count': len(feature_set.features.columns),
            'quality_score': feature_set.quality_metrics.get('completeness_score', 0),
            'output_path': str(output_path)
        }

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input file containing OHLCV data')
@click.option('--config-file', type=click.Path(exists=True), help='Feature definitions config file')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def custom_features(ctx, input, config_file, output):
    """Generate custom features based on configuration"""
    try:
        # Load feature definitions
        if not config_file:
            click.echo("Error: Config file is required for custom features")
            raise click.Abort()

        with open(config_file, 'r') as f:
            feature_definitions = json.load(f)

        # Load input data
        loader = DataLoader()
        symbol = Path(input).stem
        price_data = loader.load_ohlcv_data(
            input_path=input,
            symbol=symbol,
            name=symbol,
            instrument_type='stock'
        )

        # Generate custom features
        generator = FeatureGenerator()
        feature_set = generator.generate_custom_features(price_data, feature_definitions)

        # Save output
        if output:
            feature_set.to_csv(output)
            click.echo(f"Custom features saved to {output}")
        else:
            click.echo(f"Generated {len(feature_set.features.columns)} custom features")

    except Exception as e:
        click.echo(f"Error generating custom features: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input file containing feature data')
@click.option('--quality-threshold', type=float, default=0.7, help='Quality threshold (0-1)')
@click.option('--output', '-o', type=click.Path(), help='Output file for cleaned data')
@click.pass_context
def validate_features(ctx, input, quality_threshold, output):
    """Validate feature quality and clean if necessary"""
    try:
        # Load feature data
        features_df = pd.read_csv(input, index_col=0, parse_dates=True)

        # Validate quality
        validator = DataValidator()
        validation_result = validator.validate_data(features_df)

        click.echo(f"Validation Results for {Path(input).stem}:")
        click.echo(f"  Valid: {validation_result.is_valid}")
        click.echo(f"  Quality Score: {validation_result.quality_score:.3f}")
        click.echo(f"  Issues Found: {len(validation_result.issues)}")

        if validation_result.issues:
            click.echo("\nTop Issues:")
            for issue in validation_result.issues[:5]:
                click.echo(f"  - {issue}")

        # Clean data if quality is below threshold
        if validation_result.quality_score < quality_threshold:
            click.echo(f"\nQuality below threshold ({quality_threshold}). Cleaning data...")

            # Simple cleaning: forward fill, then backward fill
            cleaned_features = features_df.fillna(method='ffill').fillna(method='bfill')

            if output:
                cleaned_features.to_csv(output)
                click.echo(f"Cleaned features saved to {output}")
            else:
                click.echo("Data cleaned but no output file specified")

    except Exception as e:
        click.echo(f"Error validating features: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def create_config(ctx, output):
    """Create a default feature generation configuration file"""
    try:
        default_config = {
            "return_periods": [1, 5, 21, 63],
            "volatility_windows": [21, 63, 252],
            "volatility_methods": ["rolling", "ewma"],
            "momentum_periods": [14, 21, 63],
            "momentum_indicators": ["rsi", "macd", "stochastic", "williams_r", "cci", "roc", "mfi"],
            "quality_threshold": 0.7,
            "max_missing_ratio": 0.1,
            "enable_garch": True,
            "enable_multifeature": True
        }

        if output:
            with open(output, 'w') as f:
                json.dump(default_config, f, indent=2)
            click.echo(f"Configuration saved to {output}")
        else:
            click.echo(json.dumps(default_config, indent=2))

    except Exception as e:
        click.echo(f"Error creating configuration: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def create_custom_config(ctx, output):
    """Create a sample custom feature configuration file"""
    try:
        custom_config = [
            {
                "name": "daily_return",
                "type": "returns",
                "params": {
                    "period": 1,
                    "return_type": "simple"
                }
            },
            {
                "name": "weekly_volatility",
                "type": "volatility",
                "params": {
                    "method": "rolling",
                    "window": 5
                }
            },
            {
                "name": "momentum_rsi",
                "type": "momentum",
                "params": {
                    "indicator": "rsi",
                    "period": 14
                }
            }
        ]

        if output:
            with open(output, 'w') as f:
                json.dump(custom_config, f, indent=2)
            click.echo(f"Custom configuration saved to {output}")
        else:
            click.echo(json.dumps(custom_config, indent=2))

    except Exception as e:
        click.echo(f"Error creating custom configuration: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input file containing OHLCV data')
@click.option('--symbol', '-s', required=True, help='Financial instrument symbol')
@click.option('--clean', is_flag=True, help='Apply data cleaning before feature generation')
@click.option('--validate', is_flag=True, help='Apply validation and quality assessment')
@click.option('--normalize', is_flag=True, help='Apply normalization to features')
@click.option('--method', default='z_score',
              type=click.Choice(['z_score', 'min_max', 'robust']),
              help='Normalization method')
@click.option('--quality-threshold', type=float, default=0.7, help='Quality threshold (0-1)')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def pipeline_features(ctx, input, symbol, clean, validate, normalize, method,
                    quality_threshold, output):
    """Generate features with integrated preprocessing pipeline"""
    try:
        from ..lib.cleaning import DataCleaner
        from ..lib.normalization import DataNormalizer

        # Load input data
        loader = DataLoader()
        price_data = loader.load_ohlcv_data(
            input_path=input,
            symbol=symbol,
            name=symbol,
            instrument_type='stock'
        )

        click.echo(f"Processing {symbol} with {len(price_data.prices)} data points")

        # Step 1: Clean data if requested
        if clean:
            click.echo("  Cleaning data...")
            cleaner = DataCleaner()
            price_data.prices = cleaner.clean_data(price_data.prices)
            click.echo(f"  Data cleaned: {len(price_data.prices)} valid points remaining")

        # Step 2: Validate data if requested
        if validate:
            click.echo("  Validating data...")
            validator = DataValidator()
            validation_result = validator.validate_financial_data(
                price_data.prices, price_data.instrument
            )
            click.echo(f"  Validation: Score={validation_result.quality_score:.3f}, "
                      f"Valid={validation_result.is_valid}")

            if not validation_result.is_valid:
                click.echo(f"  Validation issues: {len(validation_result.issues)}")
                if validation_result.quality_score < quality_threshold:
                    click.echo(f"  Quality below threshold, applying auto-cleaning...")
                    price_data.prices = cleaner.clean_data(price_data.prices)

        # Step 3: Generate features
        click.echo("  Generating features...")
        feature_config = FeatureGenerationConfig(quality_threshold=quality_threshold)
        generator = FeatureGenerator(feature_config)
        feature_set = generator.generate_features(price_data)

        # Step 4: Normalize features if requested
        if normalize:
            click.echo(f"  Normalizing features with {method}...")
            normalizer = DataNormalizer(method=method)
            feature_data = feature_set.features.select_dtypes(include=[np.number])
            normalized_features = normalizer.normalize_data(feature_data)

            # Replace numeric features with normalized versions
            numeric_cols = feature_data.columns
            feature_set.features[numeric_cols] = normalized_features

        # Step 5: Output results
        if output:
            feature_set.to_csv(output)
            click.echo(f"Features saved to {output}")
        else:
            click.echo(f"Generated {len(feature_set.features.columns)} features")
            click.echo(f"Quality score: {feature_set.quality_metrics.get('completeness_score', 0):.3f}")

        # Summary
        click.echo(f"\nPipeline Summary:")
        click.echo(f"  Data points: {len(feature_set.features)}")
        click.echo(f"  Features: {len(feature_set.features.columns)}")
        click.echo(f"  Date range: {feature_set.features.index[0]} to {feature_set.features.index[-1]}")

    except Exception as e:
        click.echo(f"Error in pipeline feature generation: {str(e)}", err=True)
        logger.error(f"Pipeline feature generation failed for {symbol}: {str(e)}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input file containing feature data')
@click.pass_context
def feature_info(ctx, input):
    """Display information about features in a file"""
    try:
        # Load feature data
        features_df = pd.read_csv(input, index_col=0, parse_dates=True)

        click.echo(f"Feature Information for {Path(input).stem}:")
        click.echo(f"  Total Features: {len(features_df.columns)}")
        click.echo(f"  Data Points: {len(features_df)}")
        click.echo(f"  Date Range: {features_df.index[0]} to {features_df.index[-1]}")
        click.echo(f"  Missing Values: {features_df.isnull().sum().sum()}")
        click.echo(f"  Completeness: {(1 - features_df.isnull().sum().sum() / features_df.size):.3f}")

        # Feature categories (simple detection)
        return_features = [col for col in features_df.columns if 'return' in col.lower()]
        volatility_features = [col for col in features_df.columns if 'vol' in col.lower()]
        momentum_features = [col for col in features_df.columns if any(indicator in col.lower()
                          for indicator in ['rsi', 'macd', 'stochastic', 'williams', 'cci', 'roc'])]

        click.echo(f"\nFeature Categories:")
        click.echo(f"  Returns: {len(return_features)}")
        click.echo(f"  Volatility: {len(volatility_features)}")
        click.echo(f"  Momentum: {len(momentum_features)}")

        if click.confirm("\nShow feature names?"):
            click.echo("\nFeature Names:")
            for col in features_df.columns:
                click.echo(f"  - {col}")

    except Exception as e:
        click.echo(f"Error getting feature info: {str(e)}", err=True)
        raise click.Abort()


def _generate_quality_report(feature_set: FeatureSet) -> Dict[str, Any]:
    """Generate a detailed quality report"""
    return {
        'instrument': feature_set.instrument.symbol,
        'generation_timestamp': feature_set.created_at.isoformat(),
        'total_features': len(feature_set.features.columns),
        'data_points': len(feature_set.features),
        'date_range': {
            'start': str(feature_set.features.index[0]),
            'end': str(feature_set.features.index[-1])
        },
        'quality_metrics': feature_set.quality_metrics,
        'feature_types': feature_set.feature_types,
        'missing_value_analysis': {
            'total_missing': int(feature_set.features.isnull().sum().sum()),
            'missing_by_feature': feature_set.features.isnull().sum().to_dict()
        }
    }


def _generate_batch_quality_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a batch processing quality report"""
    if not results:
        return {'error': 'No results to report'}

    total_features = sum(r['feature_count'] for r in results)
    avg_quality = sum(r['quality_score'] for r in results) / len(results)

    return {
        'batch_summary': {
            'total_files': len(results),
            'successful_files': len(results),
            'total_features_generated': total_features,
            'average_quality_score': avg_quality,
            'generation_timestamp': datetime.now().isoformat()
        },
        'file_results': results,
        'quality_distribution': {
            'high_quality': len([r for r in results if r['quality_score'] >= 0.9]),
            'medium_quality': len([r for r in results if 0.7 <= r['quality_score'] < 0.9]),
            'low_quality': len([r for r in results if r['quality_score'] < 0.7])
        }
    }


if __name__ == '__main__':
    cli()