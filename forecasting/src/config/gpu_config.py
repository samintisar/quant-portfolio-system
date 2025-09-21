"""
GPU configuration and optimization settings for RTX 3060.

Provides specialized configurations for:
- GPU memory management (6GB VRAM optimization)
- CUDA kernel optimizations
- Mixed precision training
- Batch size optimization
- Multi-GPU settings
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class GPUConfiguration:
    """GPU configuration optimized for RTX 3060."""

    # Memory settings (RTX 3060 has 6GB VRAM)
    gpu_memory_limit: int = 5500  # MB (leave some buffer)
    max_batch_size: int = 64
    gradient_accumulation_steps: int = 2

    # CUDA optimization
    cuda_version: str = "12.8"
    compute_capability: str = "8.6"  # RTX 3060

    # Mixed precision training
    mixed_precision: bool = True
    fp16_opt_level: str = "O1"

    # Performance optimization
    torch_compile: bool = False  # Disabled on Windows due to Triton compatibility
    deterministic: bool = False

    # Memory optimization
    gradient_checkpointing: bool = True
    memory_efficient_attention: bool = True

    # Multi-GPU (if available)
    use_multi_gpu: bool = False
    num_gpus: int = 1

    # Model optimization
    model_parallel: bool = False
    pipeline_parallel: bool = False

    # Tensor optimization
    allow_tf32: bool = True  # TF32 is faster on Ampere architecture
    cudnn_benchmark: bool = True

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

    # Custom kernels
    use_flash_attention: bool = True
    use_fused_ops: bool = True


class GPUOptimizer:
    """GPU optimization utilities for RTX 3060."""

    def __init__(self, config: Optional[GPUConfiguration] = None):
        self.config = config or GPUConfiguration()
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.is_available = False

        self._setup_gpu_environment()

    def _setup_gpu_environment(self):
        """Setup GPU environment and check availability."""
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, using CPU")
                return

            self.is_available = True

            # Set device
            self.device = torch.device("cuda")

            # Set default tensor type
            torch.set_default_dtype(torch.float32)

            # Enable TF32 (Ampere architecture optimization)
            if self.config.allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Enable cuDNN benchmark
            if self.config.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True

            # Disable deterministic mode for performance
            if not self.config.deterministic:
                torch.backends.cudnn.deterministic = False

            # Set memory fraction if needed
            if self.config.gpu_memory_limit > 0:
                memory_fraction = self.config.gpu_memory_limit / 6144  # 6GB total
                torch.cuda.set_per_process_memory_fraction(memory_fraction)

            # Enable torch compilation if available
            if self.config.torch_compile and hasattr(torch, 'compile'):
                torch.set_float32_matmul_precision('high')

            self.logger.info(f"GPU setup complete. Device: {self.device}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        except Exception as e:
            self.logger.error(f"Failed to setup GPU environment: {e}")
            self.is_available = False

    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information."""
        if not self.is_available:
            return {"error": "GPU not available"}

        try:
            return {
                "total": torch.cuda.get_device_properties(0).total_memory / 1e9,  # GB
                "allocated": torch.cuda.memory_allocated() / 1e9,  # GB
                "cached": torch.cuda.memory_reserved() / 1e9,  # GB
                "free": (torch.cuda.get_device_properties(0).total_memory -
                        torch.cuda.memory_allocated()) / 1e9,  # GB
                "utilization": torch.cuda.memory_allocated() /
                             torch.cuda.get_device_properties(0).total_memory * 100  # %
            }
        except Exception as e:
            return {"error": str(e)}

    def clear_cache(self):
        """Clear GPU cache."""
        if self.is_available:
            torch.cuda.empty_cache()
            self.logger.info("GPU cache cleared")

    def optimize_model_for_gpu(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize PyTorch model for GPU execution."""
        if not self.is_available:
            return model

        try:
            # Move model to GPU
            model = model.to(self.device)

            # Enable mixed precision if configured
            if self.config.mixed_precision:
                model = model.half()  # Convert to FP16

            # Enable gradient checkpointing for memory efficiency
            if self.config.gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()

            # Compile model if available
            if self.config.torch_compile and hasattr(torch, 'compile'):
                model = torch.compile(model)

            self.logger.info("Model optimized for GPU execution")
            return model

        except Exception as e:
            self.logger.error(f"Failed to optimize model for GPU: {e}")
            return model

    def get_optimal_batch_size(self,
                             model: torch.nn.Module,
                             input_size: tuple,
                             max_trials: int = 5) -> int:
        """Find optimal batch size for GPU memory constraints."""
        if not self.is_available:
            return self.config.max_batch_size

        try:
            model = model.to(self.device)
            model.eval()

            # Start with max batch size
            batch_size = self.config.max_batch_size

            for _ in range(max_trials):
                try:
                    # Test memory usage
                    dummy_input = torch.randn((batch_size,) + input_size,
                                            device=self.device)
                    with torch.no_grad():
                        output = model(dummy_input)

                    # Check if we have enough memory
                    mem_info = self.get_memory_info()
                    if "error" not in mem_info:
                        utilization = mem_info["utilization"]
                        if utilization < 80:  # Leave some headroom
                            return batch_size

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        batch_size = batch_size // 2
                        continue
                    else:
                        raise e

                batch_size = batch_size // 2

            return max(1, batch_size)

        except Exception as e:
            self.logger.error(f"Failed to find optimal batch size: {e}")
            return 32  # Safe default

    def setup_mixed_precision(self) -> Dict[str, Any]:
        """Setup mixed precision training."""
        if not self.is_available or not self.config.mixed_precision:
            return {"enabled": False}

        try:
            from torch.cuda.amp import GradScaler, autocast

            scaler = GradScaler()

            return {
                "enabled": True,
                "scaler": scaler,
                "autocast": autocast,
                "dtype": torch.float16
            }

        except Exception as e:
            self.logger.error(f"Failed to setup mixed precision: {e}")
            return {"enabled": False}

    def save_config(self, filepath: str):
        """Save GPU configuration to file."""
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith('_')
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        self.logger.info(f"GPU configuration saved to {filepath}")

    def load_config(self, filepath: str):
        """Load GPU configuration from file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)

            self.config = GPUConfiguration(**config_dict)
            self._setup_gpu_environment()

            self.logger.info(f"GPU configuration loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get comprehensive GPU metrics."""
        if not self.is_available:
            return {"error": "GPU not available"}

        try:
            props = torch.cuda.get_device_properties(0)

            return {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory": props.total_memory / 1e9,
                "memory_allocated": torch.cuda.memory_allocated() / 1e9,
                "memory_reserved": torch.cuda.memory_reserved() / 1e9,
                "memory_utilization": torch.cuda.memory_allocated() / props.total_memory * 100,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "is_available": self.is_available,
                "mixed_precision_enabled": self.config.mixed_precision,
                "torch_compile_enabled": self.config.torch_compile
            }

        except Exception as e:
            return {"error": str(e)}


# Global GPU optimizer instance
gpu_optimizer = GPUOptimizer()


def setup_gpu_environment() -> GPUOptimizer:
    """Setup global GPU environment."""
    return gpu_optimizer


def get_gpu_optimizer() -> GPUOptimizer:
    """Get global GPU optimizer instance."""
    return gpu_optimizer