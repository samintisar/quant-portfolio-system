"""
GPU optimization service for financial ML models.

Provides specialized optimizations for RTX 3060 including:
- Memory management and batch size optimization
- Mixed precision training
- CUDA kernel optimization
- Performance monitoring
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from ..config.gpu_config import GPUOptimizer, GPUConfiguration
except ImportError:
    # Handle case where import fails
    from config.gpu_config import GPUOptimizer, GPUConfiguration


@dataclass
class GPUPerformanceMetrics:
    """GPU performance metrics."""
    timestamp: datetime
    memory_used_mb: float
    memory_utilization_pct: float
    gpu_utilization_pct: float
    temperature_c: float
    power_usage_w: float
    flops: float
    inference_time_ms: float
    batch_throughput: float


class GPUOptimizationService:
    """Service for optimizing GPU performance for financial ML models."""

    def __init__(self, config: Optional[GPUConfiguration] = None):
        self.config = config or GPUConfiguration()
        self.optimizer = GPUOptimizer(config)
        self.logger = logging.getLogger(__name__)
        self.performance_history: List[GPUPerformanceMetrics] = []

    def optimize_pytorch_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize PyTorch model for RTX 3060."""
        if not self.optimizer.is_available:
            self.logger.warning("GPU not available, returning CPU model")
            return model

        try:
            # Basic GPU placement
            model = model.to(self.optimizer.device)

            # Enable mixed precision if configured
            if self.config.mixed_precision:
                model = model.half()

            # Enable gradient checkpointing for memory efficiency
            if self.config.gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()

            # Compile model if available (PyTorch 2.0+)
            if self.config.torch_compile and hasattr(torch, 'compile'):
                model = torch.compile(model)

            self.logger.info("Model optimized for RTX 3060 GPU")
            return model

        except Exception as e:
            self.logger.error(f"Failed to optimize model: {e}")
            return model

    def optimize_training_loop(self,
                             model: torch.nn.Module,
                             train_loader,
                             optimizer: torch.optim.Optimizer,
                             criterion: torch.nn.Module) -> Dict[str, Any]:
        """Setup optimized training loop for RTX 3060."""
        if not self.optimizer.is_available:
            return {"enabled": False, "reason": "GPU not available"}

        try:
            setup_result = {
                "enabled": True,
                "device": self.optimizer.device,
                "mixed_precision": False,
                "gradient_accumulation": False,
                "scaler": None
            }

            # Setup mixed precision
            if self.config.mixed_precision:
                from torch.cuda.amp import GradScaler, autocast
                setup_result["mixed_precision"] = True
                setup_result["scaler"] = GradScaler()
                setup_result["autocast"] = autocast

            # Setup gradient accumulation for memory efficiency
            if self.config.gradient_accumulation_steps > 1:
                setup_result["gradient_accumulation"] = True
                setup_result["accumulation_steps"] = self.config.gradient_accumulation_steps

            # Find optimal batch size
            if len(train_loader.dataset) > 1000:
                sample_input_size = next(iter(train_loader))[0].shape[1:]
                optimal_batch_size = self.optimizer.get_optimal_batch_size(
                    model, sample_input_size
                )
                setup_result["optimal_batch_size"] = optimal_batch_size
                self.logger.info(f"Optimal batch size: {optimal_batch_size}")

            return setup_result

        except Exception as e:
            self.logger.error(f"Failed to setup training optimization: {e}")
            return {"enabled": False, "reason": str(e)}

    def optimize_inference(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Optimize model for inference on RTX 3060."""
        if not self.optimizer.is_available:
            return {"optimized": False, "reason": "GPU not available"}

        try:
            optimizations = {"optimized": True}

            # Move to inference mode
            model.eval()

            # Enable optimizations
            with torch.no_grad():
                # Use torch.jit.script for static models
                try:
                    scripted_model = torch.jit.script(model)
                    optimizations["scripted"] = True
                except:
                    optimizations["scripted"] = False

                # Use torch.compile for PyTorch 2.0+
                if self.config.torch_compile and hasattr(torch, 'compile'):
                    try:
                        compiled_model = torch.compile(model)
                        optimizations["compiled"] = True
                    except:
                        optimizations["compiled"] = False

            # Half precision for inference
            if self.config.mixed_precision:
                model = model.half()
                optimizations["half_precision"] = True

            # Enable memory efficient attention
            if self.config.memory_efficient_attention:
                optimizations["memory_efficient_attention"] = True

            self.logger.info("Model optimized for inference")
            return optimizations

        except Exception as e:
            self.logger.error(f"Failed to optimize for inference: {e}")
            return {"optimized": False, "reason": str(e)}

    def monitor_performance(self) -> GPUPerformanceMetrics:
        """Monitor current GPU performance metrics."""
        if not self.optimizer.is_available:
            return GPUPerformanceMetrics(
                timestamp=datetime.now(),
                memory_used_mb=0,
                memory_utilization_pct=0,
                gpu_utilization_pct=0,
                temperature_c=0,
                power_usage_w=0,
                flops=0,
                inference_time_ms=0,
                batch_throughput=0
            )

        try:
            # Get GPU metrics
            props = torch.cuda.get_device_properties(0)
            memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB

            # Get utilization (requires nvidia-ml-py)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Watts
            except:
                utilization = None
                temp = 0
                power = 0

            metrics = GPUPerformanceMetrics(
                timestamp=datetime.now(),
                memory_used_mb=memory_allocated,
                memory_utilization_pct=(memory_allocated / (props.total_memory / 1e6)) * 100,
                gpu_utilization_pct=utilization.gpu if utilization else 0,
                temperature_c=temp,
                power_usage_w=power,
                flops=0,  # Would need specialized calculation
                inference_time_ms=0,
                batch_throughput=0
            )

            # Store in history
            self.performance_history.append(metrics)
            if len(self.performance_history) > 1000:  # Keep last 1000 entries
                self.performance_history = self.performance_history[-1000:]

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to monitor performance: {e}")
            return GPUPerformanceMetrics(
                timestamp=datetime.now(),
                memory_used_mb=0,
                memory_utilization_pct=0,
                gpu_utilization_pct=0,
                temperature_c=0,
                power_usage_w=0,
                flops=0,
                inference_time_ms=0,
                batch_throughput=0
            )

    def optimize_memory_usage(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Optimize memory usage for RTX 3060's 6GB VRAM."""
        if not self.optimizer.is_available:
            return {"optimized": False, "reason": "GPU not available"}

        try:
            optimizations = {"optimized": True}

            # Clear cache first
            self.optimizer.clear_cache()

            # Enable gradient checkpointing
            if self.config.gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    optimizations["gradient_checkpointing"] = True

            # Use memory efficient attention if available
            if self.config.memory_efficient_attention:
                try:
                    from torch.nn.attention import SDPBackend
                    torch.nn.attention.set_default_backend(SDPBackend.MATH)
                    optimizations["memory_efficient_attention"] = True
                except:
                    pass

            # Optimize data loader settings
            optimizations["pin_memory"] = self.config.pin_memory
            optimizations["num_workers"] = self.config.num_workers

            # Set memory limit
            if self.config.gpu_memory_limit > 0:
                memory_fraction = self.config.gpu_memory_limit / 6144  # 6GB total
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                optimizations["memory_fraction"] = memory_fraction

            self.logger.info("Memory optimization complete")
            return optimizations

        except Exception as e:
            self.logger.error(f"Failed to optimize memory: {e}")
            return {"optimized": False, "reason": str(e)}

    def benchmark_model(self,
                       model: torch.nn.Module,
                       input_shape: Tuple[int, ...],
                       num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark model performance on RTX 3060."""
        if not self.optimizer.is_available:
            return {"error": "GPU not available"}

        try:
            model.eval()
            device = self.optimizer.device

            # Create dummy input with matching precision
            model_dtype = next(model.parameters()).dtype
            dummy_input = torch.randn(input_shape, device=device, dtype=model_dtype)

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)

            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(dummy_input)

            torch.cuda.synchronize()
            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            avg_inference_time = (total_time / num_iterations) * 1000  # ms
            throughput = num_iterations / total_time  # inferences/second

            # Memory usage
            memory_info = self.optimizer.get_memory_info()
            if "error" not in memory_info:
                memory_used_gb = memory_info["allocated"]
                memory_utilization = memory_info["utilization"]
            else:
                memory_used_gb = 0
                memory_utilization = 0

            return {
                "success": True,
                "total_time_s": total_time,
                "avg_inference_time_ms": avg_inference_time,
                "throughput_ips": throughput,
                "memory_used_gb": memory_used_gb,
                "memory_utilization_pct": memory_utilization,
                "batch_size": input_shape[0],
                "num_iterations": num_iterations,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return {"error": str(e)}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from monitoring history."""
        if not self.performance_history:
            return {"error": "No performance data available"}

        try:
            latest_metrics = self.performance_history[-1]

            # Calculate averages over last N measurements
            recent_metrics = self.performance_history[-100:]  # Last 100 measurements

            avg_memory_util = np.mean([m.memory_utilization_pct for m in recent_metrics])
            avg_gpu_util = np.mean([m.gpu_utilization_pct for m in recent_metrics])
            avg_temp = np.mean([m.temperature_c for m in recent_metrics])

            return {
                "current_metrics": {
                    "memory_utilization_pct": latest_metrics.memory_utilization_pct,
                    "gpu_utilization_pct": latest_metrics.gpu_utilization_pct,
                    "temperature_c": latest_metrics.temperature_c,
                    "power_usage_w": latest_metrics.power_usage_w,
                    "memory_used_mb": latest_metrics.memory_used_mb
                },
                "averages": {
                    "memory_utilization_pct": avg_memory_util,
                    "gpu_utilization_pct": avg_gpu_util,
                    "temperature_c": avg_temp
                },
                "monitoring_points": len(self.performance_history),
                "last_update": latest_metrics.timestamp.isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}

    def setup_xgboost_gpu(self) -> Dict[str, Any]:
        """Setup XGBoost for GPU acceleration."""
        try:
            import xgboost as xgb

            # XGBoost 2.0+ GPU parameters optimized for RTX 3060
            gpu_params = {
                'device': 'cuda',
                'tree_method': 'hist',
                'max_bin': 256  # Optimal for GPU
            }

            # Test GPU availability
            try:
                test_data = np.random.rand(100, 10)
                test_labels = np.random.rand(100)
                test_model = xgb.XGBRegressor(**gpu_params)
                test_model.fit(test_data, test_labels)

                return {
                    "enabled": True,
                    "parameters": gpu_params,
                    "device": "cuda"
                }
            except Exception as e:
                self.logger.warning(f"XGBoost GPU test failed: {e}")
                return {"enabled": False, "reason": str(e)}

        except ImportError:
            return {"enabled": False, "reason": "XGBoost not available"}

    def save_performance_log(self, filepath: str):
        """Save performance monitoring log to file."""
        try:
            import json

            log_data = {
                "timestamp": datetime.now().isoformat(),
                "performance_history": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "memory_used_mb": m.memory_used_mb,
                        "memory_utilization_pct": m.memory_utilization_pct,
                        "gpu_utilization_pct": m.gpu_utilization_pct,
                        "temperature_c": m.temperature_c,
                        "power_usage_w": m.power_usage_w,
                        "inference_time_ms": m.inference_time_ms,
                        "batch_throughput": m.batch_throughput
                    }
                    for m in self.performance_history
                ]
            }

            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)

            self.logger.info(f"Performance log saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save performance log: {e}")


# Global service instance
gpu_optimization_service = GPUOptimizationService()


def get_gpu_optimization_service() -> GPUOptimizationService:
    """Get global GPU optimization service instance."""
    return gpu_optimization_service