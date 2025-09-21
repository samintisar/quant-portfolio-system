#!/usr/bin/env python3
"""
GPU performance benchmarking script for RTX 3060.

This script benchmarks GPU performance for financial ML models.
"""

import sys
import torch
import numpy as np
import time
import logging
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.gpu_optimization_service import GPUOptimizationService, get_gpu_optimization_service

def setup_logging():
    """Setup logging for benchmarking."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def benchmark_matrix_operations():
    """Benchmark matrix operations on GPU vs CPU."""
    logger = setup_logging()
    logger.info("Benchmarking matrix operations...")

    # Test sizes
    sizes = [1000, 2000, 5000]
    results = []

    for size in sizes:
        logger.info(f"Testing size: {size}x{size}")

        # CPU benchmark
        cpu_start = time.time()
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - cpu_start

        # GPU benchmark
        if torch.cuda.is_available():
            gpu_start = time.time()
            x_gpu = torch.randn(size, size, device='cuda')
            y_gpu = torch.randn(size, size, device='cuda')
            torch.cuda.synchronize()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - gpu_start

            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        else:
            gpu_time = None
            speedup = None

        result = {
            "size": size,
            "cpu_time_ms": cpu_time * 1000,
            "gpu_time_ms": gpu_time * 1000 if gpu_time else None,
            "speedup": speedup
        }
        results.append(result)

        gpu_time_str = f"{gpu_time*1000:.2f}ms" if gpu_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        logger.info(f"CPU time: {cpu_time*1000:.2f}ms, GPU time: {gpu_time_str}, Speedup: {speedup_str}")

    return results

def benchmark_model_inference():
    """Benchmark model inference on GPU."""
    logger = setup_logging()
    logger.info("Benchmarking model inference...")

    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return []

    # Create test models
    models = {
        "small": torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1)
        ),
        "medium": torch.nn.Sequential(
            torch.nn.Linear(500, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        ),
        "large": torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    }

    results = []
    gpu_service = get_gpu_optimization_service()

    for model_name, model in models.items():
        logger.info(f"Benchmarking {model_name} model...")

        # Optimize model for GPU
        optimized_model = gpu_service.optimize_pytorch_model(model)

        # Test different batch sizes
        batch_sizes = [16, 32, 64, 128]

        for batch_size in batch_sizes:
            input_size = optimized_model[0].in_features
            input_shape = (batch_size, input_size)

            # Run benchmark
            benchmark_result = gpu_service.benchmark_model(
                optimized_model, input_shape, num_iterations=50
            )

            if "error" not in benchmark_result:
                result = {
                    "model": model_name,
                    "batch_size": batch_size,
                    "parameters": sum(p.numel() for p in optimized_model.parameters()),
                    "avg_inference_time_ms": benchmark_result["avg_inference_time_ms"],
                    "throughput_ips": benchmark_result["throughput_ips"],
                    "memory_used_gb": benchmark_result["memory_used_gb"]
                }
                results.append(result)

                logger.info(f"Batch {batch_size}: {benchmark_result['avg_inference_time_ms']:.2f}ms, "
                          f"{benchmark_result['throughput_ips']:.1f} inferences/sec")

    return results

def run_comprehensive_benchmark():
    """Run comprehensive GPU performance benchmark."""
    logger = setup_logging()
    logger.info("Starting comprehensive GPU performance benchmark...")

    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": {},
        "matrix_operations": {},
        "model_inference": []
    }

    # Get GPU info
    if torch.cuda.is_available():
        gpu_service = get_gpu_optimization_service()
        benchmark_results["gpu_info"] = gpu_service.optimizer.get_gpu_metrics()
        logger.info(f"GPU: {benchmark_results['gpu_info'].get('name', 'Unknown')}")

    # Run matrix benchmarks
    matrix_results = benchmark_matrix_operations()
    benchmark_results["matrix_operations"] = matrix_results

    # Run model inference benchmarks
    model_results = benchmark_model_inference()
    benchmark_results["model_inference"] = model_results

    # Save results
    output_file = "gpu_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info(f"Benchmark results saved to {output_file}")

    # Print summary
    logger.info("\n=== Benchmark Summary ===")
    if matrix_results:
        avg_speedup = np.mean([r["speedup"] for r in matrix_results if r["speedup"]])
        logger.info(f"Average matrix operation speedup: {avg_speedup:.2f}x")

    if model_results:
        fastest_inference = min(model_results, key=lambda x: x["avg_inference_time_ms"])
        logger.info(f"Fastest inference: {fastest_inference['model']} model "
                   f"(batch {fastest_inference['batch_size']}) - "
                   f"{fastest_inference['avg_inference_time_ms']:.2f}ms")

    return benchmark_results

if __name__ == "__main__":
    results = run_comprehensive_benchmark()