#!/usr/bin/env python3
"""
GPU optimization test script for RTX 3060.

This script tests and validates GPU optimizations for financial ML models.
Run this to verify GPU setup and performance.
"""

import sys
import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.gpu_config import GPUOptimizer, GPUConfiguration, get_gpu_optimizer
from services.gpu_optimization_service import GPUOptimizationService, get_gpu_optimization_service


def setup_logging():
    """Setup logging for GPU testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gpu_test.log')
        ]
    )
    return logging.getLogger(__name__)


def test_basic_gpu_availability():
    """Test basic GPU availability and configuration."""
    logger = logging.getLogger(__name__)
    logger.info("Testing basic GPU availability...")

    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        logger.info(f"GPU device count: {device_count}")

        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            logger.info(f"GPU {i}: {device_name}")

        # Test current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current GPU device: {current_device}")

        # Test device properties
        props = torch.cuda.get_device_properties(current_device)
        logger.info(f"Compute capability: {props.major}.{props.minor}")
        logger.info(f"Total memory: {props.total_memory / 1e9:.1f} GB")

        return True
    else:
        logger.error("CUDA not available")
        return False


def test_gpu_optimizer():
    """Test GPU optimizer setup."""
    logger = logging.getLogger(__name__)
    logger.info("Testing GPU optimizer setup...")

    try:
        # Create GPU optimizer
        optimizer = GPUOptimizer()
        logger.info(f"GPU optimizer created: {optimizer.is_available}")

        if optimizer.is_available:
            # Test memory info
            memory_info = optimizer.get_memory_info()
            logger.info(f"Memory info: {memory_info}")

            # Test device setup
            logger.info(f"Device: {optimizer.device}")

            return True
        else:
            logger.warning("GPU optimizer not available")
            return False

    except Exception as e:
        logger.error(f"GPU optimizer test failed: {e}")
        return False


def test_tensor_operations():
    """Test basic tensor operations on GPU."""
    logger = logging.getLogger(__name__)
    logger.info("Testing tensor operations on GPU...")

    if not torch.cuda.is_available():
        logger.error("CUDA not available for tensor operations")
        return False

    try:
        # Create test tensors
        size = 1000
        x = torch.randn(size, size, device='cuda')
        y = torch.randn(size, size, device='cuda')

        # Test basic operations
        z = x + y
        w = torch.matmul(x, y)

        # Test memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1e6
        logger.info(f"Memory allocated: {memory_allocated:.1f} MB")

        # Clear cache
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")

        return True

    except Exception as e:
        logger.error(f"Tensor operations test failed: {e}")
        return False


def test_mixed_precision():
    """Test mixed precision training setup."""
    logger = logging.getLogger(__name__)
    logger.info("Testing mixed precision setup...")

    if not torch.cuda.is_available():
        logger.error("CUDA not available for mixed precision")
        return False

    try:
        # Test half precision
        x = torch.randn(1000, 1000, device='cuda', dtype=torch.float16)
        y = torch.randn(1000, 1000, device='cuda', dtype=torch.float16)

        # Test operations
        z = x + y
        logger.info(f"Half precision tensor shape: {z.shape}")

        # Test GradScaler
        from torch.amp import GradScaler
        scaler = GradScaler('cuda')
        logger.info("GradScaler created successfully")

        return True

    except Exception as e:
        logger.error(f"Mixed precision test failed: {e}")
        return False


def test_model_optimization():
    """Test model optimization for GPU."""
    logger = logging.getLogger(__name__)
    logger.info("Testing model optimization...")

    if not torch.cuda.is_available():
        logger.error("CUDA not available for model optimization")
        return False

    try:
        # Create test model
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ).to('cuda')

        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # Test GPU optimization service
        gpu_service = get_gpu_optimization_service()
        if gpu_service.optimizer.is_available:
            optimized_model = gpu_service.optimize_pytorch_model(model)
            logger.info("Model optimized successfully")

            # Test inference
            with torch.no_grad():
                input_data = torch.randn(32, 1000, device='cuda')
                # Convert input data to match model precision
                if next(optimized_model.parameters()).dtype == torch.float16:
                    input_data = input_data.half()
                output = optimized_model(input_data)
                logger.info(f"Model output shape: {output.shape}")

            return True
        else:
            logger.warning("GPU optimization service not available")
            return False

    except Exception as e:
        logger.error(f"Model optimization test failed: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring."""
    logger = logging.getLogger(__name__)
    logger.info("Testing performance monitoring...")

    try:
        gpu_service = get_gpu_optimization_service()

        if gpu_service.optimizer.is_available:
            # Test performance metrics
            metrics = gpu_service.monitor_performance()
            logger.info(f"Performance metrics: {metrics}")

            # Test performance summary
            summary = gpu_service.get_performance_summary()
            logger.info(f"Performance summary: {summary}")

            return True
        else:
            logger.warning("Performance monitoring not available")
            return False

    except Exception as e:
        logger.error(f"Performance monitoring test failed: {e}")
        return False


def test_batch_size_optimization():
    """Test batch size optimization."""
    logger = logging.getLogger(__name__)
    logger.info("Testing batch size optimization...")

    if not torch.cuda.is_available():
        logger.error("CUDA not available for batch size optimization")
        return False

    try:
        # Create test model
        model = torch.nn.Sequential(
            torch.nn.Linear(500, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        ).to('cuda')

        # Test batch size optimization
        optimizer = get_gpu_optimizer()
        if optimizer.is_available:
            optimal_batch_size = optimizer.get_optimal_batch_size(
                model, input_size=(500,), max_trials=3
            )
            logger.info(f"Optimal batch size: {optimal_batch_size}")

            return True
        else:
            logger.warning("Batch size optimization not available")
            return False

    except Exception as e:
        logger.error(f"Batch size optimization test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive GPU optimization test."""
    logger = setup_logging()
    logger.info("Starting comprehensive GPU optimization test...")

    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # Run all tests
    tests = [
        ("basic_gpu_availability", test_basic_gpu_availability),
        ("gpu_optimizer", test_gpu_optimizer),
        ("tensor_operations", test_tensor_operations),
        ("mixed_precision", test_mixed_precision),
        ("model_optimization", test_model_optimization),
        ("performance_monitoring", test_performance_monitoring),
        ("batch_size_optimization", test_batch_size_optimization)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        try:
            logger.info(f"Running {test_name}...")
            result = test_func()
            test_results["tests"][test_name] = {
                "passed": result,
                "error": None
            }
            if result:
                passed_tests += 1
                logger.info(f"PASS {test_name} passed")
            else:
                logger.error(f"FAIL {test_name} failed")
        except Exception as e:
            test_results["tests"][test_name] = {
                "passed": False,
                "error": str(e)
            }
            logger.error(f"FAIL {test_name} failed with exception: {e}")

    # Summary
    test_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    }

    logger.info("\n=== Test Summary ===")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success rate: {test_results['summary']['success_rate']:.1f}%")

    # Save results
    output_file = "gpu_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Test results saved to {output_file}")

    return test_results


if __name__ == "__main__":
    results = run_comprehensive_test()
    sys.exit(0 if results["summary"]["success_rate"] >= 80 else 1)