#!/usr/bin/env python3
"""
Environment setup script to ensure proper conda environment usage.
This script checks if we're in the correct conda environment and helps with setup.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_conda_environment():
    """Check if we're running in the correct conda environment."""
    expected_env = "quant-portfolio-system"
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")

    print(f"Current conda environment: {current_env}")
    print(f"Expected environment: {expected_env}")

    if current_env != expected_env:
        print(f"\n❌ Not running in the correct conda environment!")
        print(f"Please run: conda activate {expected_env}")
        return False

    print("✅ Running in correct conda environment!")
    return True

def install_dependencies():
    """Install dependencies using conda environment."""
    try:
        print("Installing/updating dependencies...")
        subprocess.run(["conda", "env", "update", "-f", "environment.yml", "--prune"],
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies updated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to update dependencies: {e}")
        return False

def verify_installation():
    """Verify that all required packages can be imported."""
    required_packages = [
        "pandas", "numpy", "yfinance", "cvxpy", "sklearn",
        "fastapi", "uvicorn", "pydantic", "matplotlib", "pytest"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        return False

    print("\n✅ All packages imported successfully!")
    return True

def verify_project_imports():
    """Verify that project modules can be imported."""
    try:
        import portfolio
        print("✅ Portfolio package imported successfully!")

        # Test specific modules
        from portfolio.ml import RandomForestPredictor
        print("✅ ML predictor imported successfully!")

        from portfolio.optimizer.optimizer import SimplePortfolioOptimizer
        print("✅ Portfolio optimizer imported successfully!")

        return True
    except ImportError as e:
        print(f"❌ Failed to import project modules: {e}")
        return False

def main():
    """Main setup function."""
    print("=== Quant Portfolio System Environment Setup ===\n")

    # Check conda environment
    if not check_conda_environment():
        print("\nPlease activate the correct conda environment first:")
        print("conda activate quant-portfolio-system")
        return False

    # Install/update dependencies
    if not install_dependencies():
        return False

    # Verify installation
    if not verify_installation():
        return False

    # Verify project imports
    if not verify_project_imports():
        return False

    print("\n🎉 Environment setup completed successfully!")
    print("You can now run the project commands.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)