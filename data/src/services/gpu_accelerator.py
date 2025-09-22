"""
GPU Acceleration Service for Quantitative Trading System
======================================================

Provides GPU acceleration capabilities for RTX 3090 optimization.
Handles device management, memory optimization, and performance monitoring.

Author: Claude Code
Version: 1.0.0
"""

import logging
import os
import gc
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
from pathlib import Path
import time

# Core imports
try:
    import torch
    import torch.nn as nn
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import cudf
    import cuml
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Local imports
from .performance_monitor import PerformanceMonitor

@dataclass
class GPUInfo:
    """GPU device information"""
    name: str
    memory_total: int
    memory_free: int
    compute_capability: str
    device_index: int
    is_available: bool

@dataclass
class GPUMemoryInfo:
    """GPU memory usage information"""
    allocated: int
    cached: int
    max_allocated: int
    utilization_percent: float

class GPUAccelerator:
    """
    GPU acceleration service for RTX 3090 optimization
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize GPU accelerator

        Args:
            config_path: Path to GPU configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.device = None
        self.memory_pool = None
        self.gpu_info = None
        self.performance_monitor = PerformanceMonitor()

        # Initialize GPU
        self._initialize_gpu()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load GPU configuration"""
        default_config = {
            'gpu': {
                'enabled': True,
                'device': 'cuda:0',
                'memory_limit_gb': 24,
                'compute_capability': '8.6',
                'auto_detect': True
            },
            'optimization': {
                'mixed_precision': True,
                'memory_pool': True,
                'kernel_optimization': True
            },
            'batching': {
                'optimal_batch_size': 512,
                'max_batch_size': 2048,
                'dynamic_batching': True
            },
            'memory': {
                'preallocation': True,
                'memory_pool_size_gb': 20,
                'cleanup_threshold_percent': 85
            }
        }

        # Try to load from TOML file if provided
        if config_path and os.path.exists(config_path):
            try:
                import toml
                with open(config_path, 'r') as f:
                    file_config = toml.load(f)
                    # Merge with defaults
                    for key, value in file_config.items():
                        if key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except ImportError:
                self.logger.warning("TOML library not available, using default config")
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}")

        return default_config

    def _initialize_gpu(self):
        """Initialize GPU device and settings"""
        if not self._is_gpu_available():
            self.logger.warning("GPU not available, falling back to CPU")
            self.device = torch.device('cpu') if TORCH_AVAILABLE else None
            return

        try:
            # Initialize PyTorch
            if TORCH_AVAILABLE:
                self._initialize_pytorch()

            # Initialize TensorFlow
            if TF_AVAILABLE:
                self._initialize_tensorflow()

            # Initialize RAPIDS
            if RAPIDS_AVAILABLE:
                self._initialize_rapids()

            # Get GPU info
            self._get_gpu_info()

            # Setup memory management
            self._setup_memory_management()

            # Setup optimization settings
            self._setup_optimization()

            self.logger.info(f"GPU initialized successfully: {self.gpu_info}")

        except Exception as e:
            self.logger.error(f"Failed to initialize GPU: {e}")
            self.device = torch.device('cpu') if TORCH_AVAILABLE else None

    def _is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        if not self.config['gpu']['enabled']:
            return False

        if TORCH_AVAILABLE:
            return torch.cuda.is_available()
        elif TF_AVAILABLE:
            return len(tf.config.list_physical_devices('GPU')) > 0
        elif CUPY_AVAILABLE:
            return cp.cuda.is_available()

        return False

    def _initialize_pytorch(self):
        """Initialize PyTorch GPU settings"""
        self.device = torch.device(self.config['gpu']['device'])

        # Set default device
        torch.cuda.set_device(self.device)

        # Enable optimizations
        if self.config['optimization']['kernel_optimization']:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Enable TF32 for Ampere architecture
        if self.config['optimization'].get('enable_tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _initialize_tensorflow(self):
        """Initialize TensorFlow GPU settings"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Enable mixed precision
                if self.config['optimization']['mixed_precision']:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)

            except RuntimeError as e:
                self.logger.warning(f"TensorFlow GPU initialization failed: {e}")

    def _initialize_rapids(self):
        """Initialize RAPIDS libraries"""
        # CuPy initialization
        if CUPY_AVAILABLE:
            try:
                # Set memory pool
                if self.config['memory']['memory_pool']:
                    self.memory_pool = cp.get_default_memory_pool()
                    self.memory_pool.set_limit(size=self.config['memory']['memory_pool_size_gb'] * 1024**3)
            except Exception as e:
                self.logger.warning(f"CuPy memory pool initialization failed: {e}")

    def _get_gpu_info(self):
        """Get GPU device information"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self.device)
                self.gpu_info = GPUInfo(
                    name=props.name,
                    memory_total=props.total_memory,
                    memory_free=torch.cuda.memory_available(self.device),
                    compute_capability=f"{props.major}.{props.minor}",
                    device_index=self.device.index,
                    is_available=True
                )
            elif CUPY_AVAILABLE and cp.cuda.is_available():
                device = cp.cuda.Device()
                self.gpu_info = GPUInfo(
                    name=device.name,
                    memory_total=device.mem_info[1],
                    memory_free=device.mem_info[0],
                    compute_capability="8.6",  # RTX 3090
                    device_index=0,
                    is_available=True
                )
            elif NVML_AVAILABLE:
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')

                self.gpu_info = GPUInfo(
                    name=name,
                    memory_total=info.total,
                    memory_free=info.free,
                    compute_capability="8.6",
                    device_index=0,
                    is_available=True
                )
                nvml.nvmlShutdown()
        except Exception as e:
            self.logger.error(f"Failed to get GPU info: {e}")
            self.gpu_info = GPUInfo(
                name="Unknown",
                memory_total=0,
                memory_free=0,
                compute_capability="0.0",
                device_index=0,
                is_available=False
            )

    def _setup_memory_management(self):
        """Setup GPU memory management"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return

        try:
            # Enable memory pooling
            if self.config['memory']['memory_pool']:
                torch.cuda.memory.pool.enabled = True

            # Set memory fraction
            memory_limit = self.config['memory'].get('memory_limit_gb', 20) * 1024**3
            if memory_limit > 0:
                torch.cuda.set_per_process_memory_fraction(
                    memory_limit / self.gpu_info.memory_total
                )

        except Exception as e:
            self.logger.warning(f"Memory management setup failed: {e}")

    def _setup_optimization(self):
        """Setup optimization settings"""
        # Set environment variables
        env_vars = {
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
            'CUDA_LAUNCH_BLOCKING': '0',
            'OMP_NUM_THREADS': '8',
            'MKL_NUM_THREADS': '8'
        }

        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value

    def get_device(self) -> Any:
        """Get the configured device"""
        return self.device

    def optimize_model_for_gpu(self, model: Any) -> Any:
        """
        Optimize a model for GPU execution

        Args:
            model: PyTorch or TensorFlow model

        Returns:
            GPU-optimized model
        """
        if not self._is_gpu_available():
            return model

        try:
            if TORCH_AVAILABLE and hasattr(model, 'to'):
                # Move model to GPU
                model = model.to(self.device)

                # Enable mixed precision
                if self.config['optimization']['mixed_precision']:
                    if torch.cuda.is_bf16_supported():
                        model = model.to(torch.bfloat16)
                    else:
                        model = model.to(torch.float16)

                # Enable gradient checkpointing for memory efficiency
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()

            elif TF_AVAILABLE and hasattr(model, 'compile'):
                # TensorFlow model optimization
                if self.config['optimization']['mixed_precision']:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)

                # Compile with XLA if enabled
                if self.config['optimization'].get('enable_xla', False):
                    model.compile(run_eagerly=False)

            return model

        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return model

    def get_optimal_batch_size(self, model_size_mb: float = 100) -> int:
        """
        Calculate optimal batch size based on available memory

        Args:
            model_size_mb: Model size in MB

        Returns:
            Optimal batch size
        """
        if not self._is_gpu_available():
            return 32

        try:
            available_memory_mb = self.get_memory_info().free // (1024 * 1024)

            # Reserve 20% memory for safety
            safe_memory_mb = available_memory_mb * 0.8

            # Calculate based on model size and available memory
            memory_per_sample_mb = model_size_mb * 1.5  # Estimate with activations
            max_batch = int(safe_memory_mb / memory_per_sample_mb)

            # Apply constraints from config
            optimal_batch = self.config['batching']['optimal_batch_size']
            max_batch_config = self.config['batching']['max_batch_size']

            return min(max_batch, max_batch_config, optimal_batch)

        except Exception as e:
            self.logger.error(f"Batch size calculation failed: {e}")
            return self.config['batching']['optimal_batch_size']

    def get_memory_info(self) -> GPUMemoryInfo:
        """Get current GPU memory information"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return GPUMemoryInfo(
                    allocated=torch.cuda.memory_allocated(self.device),
                    cached=torch.cuda.memory_reserved(self.device),
                    max_allocated=torch.cuda.max_memory_allocated(self.device),
                    utilization_percent=torch.cuda.utilization()
                )
            elif CUPY_AVAILABLE and cp.cuda.is_available():
                mempool = cp.get_default_memory_pool()
                return GPUMemoryInfo(
                    allocated=mempool.used_bytes(),
                    cached=mempool.total_bytes(),
                    max_allocated=mempool.max_used_bytes(),
                    utilization_percent=0.0  # CuPy doesn't provide utilization
                )
            elif NVML_AVAILABLE:
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                nvml.nvmlShutdown()

                return GPUMemoryInfo(
                    allocated=info.used,
                    cached=info.used,  # CuPy doesn't distinguish
                    max_allocated=info.used,
                    utilization_percent=utilization.gpu
                )
        except Exception as e:
            self.logger.error(f"Memory info retrieval failed: {e}")

        return GPUMemoryInfo(0, 0, 0, 0.0)

    def cleanup_memory(self):
        """Clean up GPU memory"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            if CUPY_AVAILABLE and cp.cuda.is_available():
                cp.get_default_memory_pool().free_all_blocks()

            # Force garbage collection
            gc.collect()

            self.logger.info("GPU memory cleanup completed")

        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get GPU performance metrics"""
        metrics = {
            'gpu_available': self._is_gpu_available(),
            'gpu_info': self.gpu_info.__dict__ if self.gpu_info else None,
            'memory_info': self.get_memory_info().__dict__,
            'timestamp': time.time()
        }

        # Add framework-specific metrics
        if TORCH_AVAILABLE and torch.cuda.is_available():
            metrics.update({
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'device_name': torch.cuda.get_device_name(self.device)
            })

        return metrics

    def shutdown(self):
        """Shutdown GPU accelerator"""
        try:
            self.cleanup_memory()

            if NVML_AVAILABLE:
                nvml.nvmlShutdown()

            self.logger.info("GPU accelerator shutdown completed")

        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")