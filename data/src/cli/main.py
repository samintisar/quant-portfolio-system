"""
Main CLI Entry Point

Simple main function that serves as the entry point for CLI operations.
"""

import sys
import argparse
from typing import Optional


def cli_main():
    """
    Main CLI entry point function.

    This function provides a simple command-line interface that delegates
    to the existing Click-based CLI in feature_commands.py.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Quantitative Portfolio System CLI',
        prog='quant-cli'
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate features command
    generate_parser = subparsers.add_parser('generate', help='Generate features')
    generate_parser.add_argument('--input', '-i', required=True, help='Input file path')
    generate_parser.add_argument('--symbol', '-s', required=True, help='Financial instrument symbol')
    generate_parser.add_argument('--output', '-o', help='Output file path')

    # Validate features command
    validate_parser = subparsers.add_parser('validate', help='Validate features')
    validate_parser.add_argument('--input', '-i', required=True, help='Input file path')

    # List features command
    list_parser = subparsers.add_parser('list', help='List features')
    list_parser.add_argument('--category', default='all', help='Feature category')

    # Configure command
    config_parser = subparsers.add_parser('configure', help='Configure pipeline')
    config_parser.add_argument('--output', '-o', help='Output config file')

    # Help command
    subparsers.add_parser('help', help='Show help')

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == 'generate':
        print(f"Generating features for {args.symbol} from {args.input}")
        # Delegate to existing Click CLI
        sys.argv = ['feature-cli', 'generate-features',
                   '--input', args.input, '--symbol', args.symbol]
        if args.output:
            sys.argv.extend(['--output', args.output])

        # Import and run the existing CLI
        from .feature_commands import cli
        cli()

    elif args.command == 'validate':
        print(f"Validating features from {args.input}")
        sys.argv = ['feature-cli', 'validate-features', '--input', args.input]

        from .feature_commands import cli
        cli()

    elif args.command == 'list':
        print(f"Listing features in category: {args.category}")
        print("Available features: returns, volatility, momentum, trend, volume")

    elif args.command == 'configure':
        print("Creating configuration")
        sys.argv = ['feature-cli', 'create-config']
        if args.output:
            sys.argv.extend(['--output', args.output])

        from .feature_commands import cli
        cli()

    elif args.command == 'help' or args.command is None:
        parser.print_help()

    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


def main() -> None:
    """Backwards-compatible entry point expected by contract tests."""

    cli_main()


if __name__ == '__main__':
    cli_main()
