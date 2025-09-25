#!/usr/bin/env python3
"""
Data configuration validation script.

This script validates that the data storage follows the configured policies
and identifies any files that violate the data management rules.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.data_settings import (
    ESSENTIAL_TIME_PERIODS, CORE_SYMBOLS, BENCHMARK_SYMBOLS,
    DATA_DIRECTORIES, FILE_NAMING, validate_config,
    get_restricted_periods
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def find_violation_files():
    """Find files that violate data management policies."""
    violations = {
        'restricted_periods': [],
        'report_files': [],
        'unknown_symbols': [],
        'duplicate_data': [],
        ' misplaced_files': []
    }

    restricted_periods = get_restricted_periods()
    all_allowed_symbols = set(CORE_SYMBOLS + BENCHMARK_SYMBOLS)

    # Scan data directories
    for directory in [DATA_DIRECTORIES['raw'], DATA_DIRECTORIES['processed']]:
        if not os.path.exists(directory):
            continue

        for file_path in Path(directory).glob('*.csv'):
            filename = file_path.stem
            filepath = str(file_path)

            # Check for restricted periods
            for period in restricted_periods:
                if f'_{period}_' in filename:
                    violations['restricted_periods'].append(filepath)

            # Check for report files
            if filename.endswith('_report'):
                violations['report_files'].append(filepath)

            # Check for unknown symbols
            parts = filename.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                if symbol not in all_allowed_symbols and not symbol.startswith('^'):
                    violations['unknown_symbols'].append(filepath)

    return violations

def check_data_completeness():
    """Check if essential data is complete."""
    completeness = {
        'missing_files': [],
        'found_files': []
    }

    # Check for essential files
    essential_files = []

    # Core symbol files (processed)
    for symbol in CORE_SYMBOLS:
        for period in ESSENTIAL_TIME_PERIODS:
            essential_files.append(os.path.join(
                DATA_DIRECTORIES['processed'],
                FILE_NAMING['processed'].format(symbol=symbol, period=period)
            ))

    # Benchmark files (processed)
    for symbol in BENCHMARK_SYMBOLS:
        for period in ESSENTIAL_TIME_PERIODS:
            essential_files.append(os.path.join(
                DATA_DIRECTORIES['processed'],
                FILE_NAMING['processed'].format(symbol=symbol, period=period)
            ))

    # Combined files
    for period in ESSENTIAL_TIME_PERIODS:
        essential_files.append(os.path.join(
            DATA_DIRECTORIES['processed'],
            FILE_NAMING['combined'].format(period=period)
        ))

    # Check which files exist
    for file_path in essential_files:
        if os.path.exists(file_path):
            completeness['found_files'].append(file_path)
        else:
            completeness['missing_files'].append(file_path)

    return completeness

def generate_report():
    """Generate a comprehensive data validation report."""
    print("=== Data Configuration Validation ===\n")

    # Validate configuration
    print("1. Configuration Validation:")
    config_validation = validate_config()
    if config_validation['valid']:
        print("   OK: Configuration is valid")
    else:
        print("   ISSUE: Configuration has problems:")
        for issue in config_validation['issues']:
            print(f"      - {issue}")
    print()

    # Check for policy violations
    print("2. Policy Violations:")
    violations = find_violation_files()

    if violations['restricted_periods']:
        print(f"   ISSUE: Found {len(violations['restricted_periods'])} files with restricted periods:")
        for file_path in violations['restricted_periods'][:5]:  # Show first 5
            print(f"      - {file_path}")
        if len(violations['restricted_periods']) > 5:
            print(f"      ... and {len(violations['restricted_periods']) - 5} more")

    if violations['report_files']:
        print(f"   ISSUE: Found {len(violations['report_files'])} report files (should be removed):")
        for file_path in violations['report_files'][:5]:
            print(f"      - {file_path}")
        if len(violations['report_files']) > 5:
            print(f"      ... and {len(violations['report_files']) - 5} more")

    if violations['unknown_symbols']:
        print(f"   ISSUE: Found {len(violations['unknown_symbols'])} files with unknown symbols:")
        for file_path in violations['unknown_symbols']:
            print(f"      - {file_path}")

    if not any(violations.values()):
        print("   OK: No policy violations found")
    print()

    # Check data completeness
    print("3. Data Completeness:")
    completeness = check_data_completeness()

    essential_count = len(completeness['found_files']) + len(completeness['missing_files'])
    completeness_pct = len(completeness['found_files']) / essential_count * 100 if essential_count > 0 else 0

    print(f"   Essential files: {len(completeness['found_files'])}/{essential_count} ({completeness_pct:.1f}%)")

    if completeness['missing_files']:
        print(f"   ISSUE: Missing {len(completeness['missing_files'])} essential files:")
        for file_path in completeness['missing_files'][:5]:
            print(f"      - {file_path}")
        if len(completeness['missing_files']) > 5:
            print(f"      ... and {len(completeness['missing_files']) - 5} more")
    else:
        print("   OK: All essential files present")
    print()

    # Summary
    total_violations = sum(len(v) for v in violations.values())
    print("4. Summary:")
    config_status = "Valid" if config_validation['valid'] else "Invalid"
    violation_status = "None" if total_violations == 0 else f"{total_violations} found"
    complete_status = "Complete" if not completeness['missing_files'] else f"{len(completeness['missing_files'])} missing"

    print(f"   Configuration: {'OK' if config_validation['valid'] else 'ISSUE'} ({config_status})")
    print(f"   Policy violations: {'OK' if total_violations == 0 else 'ISSUE'} ({violation_status})")
    print(f"   Data completeness: {'OK' if not completeness['missing_files'] else 'ISSUE'} ({complete_status})")

    return {
        'config_valid': config_validation['valid'],
        'violations': violations,
        'completeness': completeness,
        'total_violations': total_violations
    }

def auto_cleanup(violations):
    """Automatically clean up policy violations."""
    if not any(violations.values()):
        print("No cleanup needed - no violations found.")
        return True

    print("\n=== Auto-Cleanup ===")
    cleaned_count = 0

    # Clean report files
    for file_path in violations['report_files']:
        try:
            os.remove(file_path)
            print(f"OK: Removed report file: {file_path}")
            cleaned_count += 1
        except Exception as e:
            print(f"ERROR: Failed to remove {file_path}: {e}")

    # Clean restricted period files (ask for confirmation)
    if violations['restricted_periods']:
        print(f"\nFound {len(violations['restricted_periods'])} files with restricted periods.")
        response = input("Remove these files? (y/N): ").strip().lower()
        if response == 'y':
            for file_path in violations['restricted_periods']:
                try:
                    os.remove(file_path)
                    print(f"OK: Removed {file_path}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"ERROR: Failed to remove {file_path}: {e}")

    print(f"\nCleanup complete: {cleaned_count} files removed.")
    return cleaned_count > 0

if __name__ == "__main__":
    print("Data Configuration Validation Script")
    print("=" * 50)

    try:
        # Generate validation report
        report = generate_report()

        # Ask if user wants to cleanup
        if report['total_violations'] > 0:
            print(f"\nFound {report['total_violations']} policy violations.")
            response = input("Run automatic cleanup? (y/N): ").strip().lower()
            if response == 'y':
                auto_cleanup(report['violations'])
                print("\nRe-running validation after cleanup...")
                generate_report()
        else:
            print("\n✅ Data storage is compliant with configuration!")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Validation failed: {e}")

    print("\nScript completed.")