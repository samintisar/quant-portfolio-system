#!/usr/bin/env python3
"""
GPU Environment Setup Script for RTX 3090
==========================================

This script sets up the GPU environment for optimal performance on RTX 3090.
It validates installation, configures settings, and provides diagnostics.

Author: Claude Code
Version: 1.0.0
"""

import os
import sys
import subprocess
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.src.services.gpu_accelerator import GPUAccelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUEnvironmentSetup:
    """GPU environment setup and validation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_accelerator = None
        self.installation_results = {}

    def setup_environment(self) -> Dict[str, Any]:
        """Complete GPU environment setup"""
        results = {
            'timestamp': time.time(),
            'setup_phase': 'gpu_optimization',
            'components': {},
            'success': False,
            'recommendations': []
        }

        try:
            # Step 1: Check CUDA installation
            self.logger.info("Checking CUDA installation...")
            results['components']['cuda'] = self._check_cuda_installation()

            # Step 2: Check GPU drivers
            self.logger.info("Checking GPU drivers...")
            results['components']['drivers'] = self._check_gpu_drivers()

            # Step 3: Validate Python packages
            self.logger.info("Validating Python packages...")
            results['components']['packages'] = self._validate_python_packages()

            # Step 4: Initialize GPU accelerator
            self.logger.info("Initializing GPU accelerator...")
            self.gpu_accelerator = GPUAccelerator()
            results['components']['accelerator'] = self._test_gpu_accelerator()

            # Step 5: Test GPU performance
            self.logger.info("Testing GPU performance...")
            results['components']['performance'] = self._test_gpu_performance()

            # Step 6: Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)

            # Check overall success
            results['success'] = all(
                component.get('success', False)
                for component in results['components'].values()
            )

            if results['success']:
                self.logger.info("✅ GPU environment setup completed successfully!")
            else:
                self.logger.warning("⚠️ GPU environment setup completed with issues")

            return results

        except Exception as e:
            self.logger.error(f"GPU environment setup failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    def _check_cuda_installation(self) -> Dict[str, Any]:
        """Check CUDA installation and version"""
        result = {'success': False, 'details': {}}

        try:
            # Check nvcc (CUDA compiler)
            try:
                nvcc_version = subprocess.run(
                    ['nvcc', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if nvcc_version.returncode == 0:
                    result['details']['nvcc'] = nvcc_version.stdout
                    result['success'] = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                result['details']['nvcc'] = 'Not found'

            # Check CUDA runtime
            try:
                cuda_version = subprocess.run(
                    ['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if cuda_version.returncode == 0:
                    result['details']['cuda_runtime'] = cuda_version.stdout.strip()
                    result['success'] = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                result['details']['cuda_runtime'] = 'Not found'

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    def _check_gpu_drivers(self) -> Dict[str, Any]:
        """Check GPU drivers and device information"""
        result = {'success': False, 'details': {}}

        try:
            # Check nvidia-smi
            try:
                driver_info = subprocess.run(
                    ['nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if driver_info.returncode == 0:
                    lines = driver_info.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            result['details'][f'gpu_{i}'] = {
                                'driver_version': parts[0],
                                'name': parts[1],
                                'memory_total': parts[2]
                            }
                    result['success'] = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                result['details']['nvidia_smi'] = 'Not found'

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    def _validate_python_packages(self) -> Dict[str, Any]:
        """Validate required Python packages"""
        result = {'success': False, 'packages': {}}

        required_packages = [
            'torch', 'torchvision', 'torchaudio',
            'tensorflow', 'cupy', 'cudf', 'cuml',
            'nvidia_ml_py3', 'psutil', 'numpy'
        ]

        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                result['packages'][package] = {'installed': True, 'version': 'unknown'}

                # Get version if possible
                try:
                    module = sys.modules[package.replace('-', '_')]
                    if hasattr(module, '__version__'):
                        result['packages'][package]['version'] = module.__version__
                except:
                    pass

            except ImportError:
                result['packages'][package] = {'installed': False, 'version': 'N/A'}

        # Check if critical packages are installed
        critical_packages = ['torch', 'numpy']
        result['success'] = all(
            result['packages'][pkg]['installed']
            for pkg in critical_packages
        )

        return result

    def _test_gpu_accelerator(self) -> Dict[str, Any]:
        """Test GPU accelerator functionality"""
        result = {'success': False, 'tests': {}}

        if not self.gpu_accelerator:
            result['error'] = 'GPU accelerator not initialized'
            return result

        try:
            # Test device detection
            device = self.gpu_accelerator.get_device()
            result['tests']['device_detection'] = {
                'success': device is not None,
                'device': str(device) if device else 'None'
            }

            # Test memory info
            try:
                memory_info = self.gpu_accelerator.get_memory_info()
                result['tests']['memory_info'] = {
                    'success': True,
                    'allocated_mb': memory_info.allocated // (1024 * 1024),
                    'utilization_percent': memory_info.utilization_percent
                }
            except Exception as e:
                result['tests']['memory_info'] = {'success': False, 'error': str(e)}

            # Test batch size calculation
            try:
                batch_size = self.gpu_accelerator.get_optimal_batch_size()
                result['tests']['batch_size'] = {
                    'success': True,
                    'optimal_batch_size': batch_size
                }
            except Exception as e:
                result['tests']['batch_size'] = {'success': False, 'error': str(e)}

            # Test GPU info
            try:
                gpu_info = self.gpu_accelerator.gpu_info
                result['tests']['gpu_info'] = {
                    'success': gpu_info is not None,
                    'name': gpu_info.name if gpu_info else 'Unknown',
                    'memory_total_gb': gpu_info.memory_total // (1024**3) if gpu_info else 0
                }
            except Exception as e:
                result['tests']['gpu_info'] = {'success': False, 'error': str(e)}

            # Test performance metrics
            try:
                metrics = self.gpu_accelerator.get_performance_metrics()
                result['tests']['performance_metrics'] = {
                    'success': True,
                    'gpu_available': metrics.get('gpu_available', False)
                }
            except Exception as e:
                result['tests']['performance_metrics'] = {'success': False, 'error': str(e)}

            # Overall success
            result['success'] = all(
                test.get('success', False)
                for test in result['tests'].values()
            )

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    def _test_gpu_performance(self) -> Dict[str, Any]:
        """Test GPU performance with sample workloads"""
        result = {'success': False, 'benchmarks': {}}

        if not self.gpu_accelerator:
            result['error'] = 'GPU accelerator not initialized'
            return result

        try:
            # Test matrix multiplication (PyTorch)
            if 'torch' in sys.modules:
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Matrix multiplication benchmark
                        device = torch.device('cuda')
                        size = 2048

                        # Warmup
                        a = torch.randn(size, size, device=device)
                        b = torch.randn(size, size, device=device)
                        torch.cuda.synchronize()

                        # Benchmark
                        start_time = time.time()
                        for _ in range(10):
                            c = torch.matmul(a, b)
                        torch.cuda.synchronize()
                        end_time = time.time()

                        result['benchmarks']['matrix_mult'] = {
                            'success': True,
                            'time_seconds': (end_time - start_time) / 10,
                            'size': size
                        }
                except Exception as e:
                    result['benchmarks']['matrix_mult'] = {'success': False, 'error': str(e)}

            # Test large array operations (NumPy/SciPy)
            if 'numpy' in sys.modules:
                try:
                    import numpy as np

                    # Large array operation
                    size = 1000000
                    start_time = time.time()
                    for _ in range(10):
                        arr = np.random.randn(size)
                        result_arr = np.fft.fft(arr)
                    end_time = time.time()

                    result['benchmarks']['array_ops'] = {
                        'success': True,
                        'time_seconds': (end_time - start_time) / 10,
                        'size': size
                    }
                except Exception as e:
                    result['benchmarks']['array_ops'] = {'success': False, 'error': str(e)}

            # Test data processing (Pandas)
            if 'pandas' in sys.modules:
                try:
                    import pandas as pd

                    # Large DataFrame operations
                    df_size = 100000
                    start_time = time.time()
                    for _ in range(5):
                        df = pd.DataFrame({
                            'col1': np.random.randn(df_size),
                            'col2': np.random.randn(df_size),
                            'col3': np.random.choice(['A', 'B', 'C'], df_size)
                        })
                        result_df = df.groupby('col3').agg(['mean', 'std'])
                    end_time = time.time()

                    result['benchmarks']['data_processing'] = {
                        'success': True,
                        'time_seconds': (end_time - start_time) / 5,
                        'df_size': df_size
                    }
                except Exception as e:
                    result['benchmarks']['data_processing'] = {'success': False, 'error': str(e)}

            # Overall success
            successful_benchmarks = sum(
                1 for benchmark in result['benchmarks'].values()
                if benchmark.get('success', False)
            )
            result['success'] = successful_benchmarks > 0

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate setup recommendations based on results"""
        recommendations = []

        # CUDA recommendations
        if not results['components'].get('cuda', {}).get('success', False):
            recommendations.append("Install CUDA 12.1 toolkit from NVIDIA website")
            recommendations.append("Add CUDA to system PATH environment variable")

        # Driver recommendations
        if not results['components'].get('drivers', {}).get('success', False):
            recommendations.append("Install latest NVIDIA drivers for RTX 3090")
            recommendations.append("Restart system after driver installation")

        # Package recommendations
        packages = results['components'].get('packages', {}).get('packages', {})
        missing_packages = [pkg for pkg, info in packages.items() if not info.get('installed', False)]

        if missing_packages:
            recommendations.append(f"Install missing packages: {', '.join(missing_packages)}")
            recommendations.append("Run: pip install -r docs/requirements.txt")

        # Performance recommendations
        if results['components'].get('accelerator', {}).get('success', False):
            accelerator_tests = results['components']['accelerator'].get('tests', {})
            if not accelerator_tests.get('device_detection', {}).get('success', False):
                recommendations.append("Check GPU device detection and configuration")

            if not accelerator_tests.get('memory_info', {}).get('success', False):
                recommendations.append("Verify GPU memory management setup")

        # Environment setup recommendations
        recommendations.append("Set up environment variables: source .env.gpu")
        recommendations.append("Configure GPU settings in config/gpu_config.toml")
        recommendations.append("Monitor GPU performance with built-in monitoring tools")

        return recommendations

    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Save setup results to file"""
        if output_path is None:
            output_path = Path(__file__).parent / 'gpu_setup_results.json'

        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Setup results saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

def main():
    """Main setup function"""
    print("🚀 Starting GPU Environment Setup for RTX 3090...")
    print("=" * 50)

    setup = GPUEnvironmentSetup()
    results = setup.setup_environment()

    # Print summary
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY")
    print("=" * 50)

    for component_name, component_result in results['components'].items():
        status = "✅" if component_result.get('success', False) else "❌"
        print(f"{status} {component_name.upper()}: ", end="")
        if component_result.get('success', False):
            print("Completed successfully")
        else:
            print("Failed - see details")

    print(f"\n🎯 Overall Status: {'✅ SUCCESS' if results['success'] else '❌ NEEDS ATTENTION'}")

    if results['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"   {i}. {rec}")

    # Save results
    setup.save_results(results)

    return 0 if results['success'] else 1

if __name__ == "__main__":
    sys.exit(main())