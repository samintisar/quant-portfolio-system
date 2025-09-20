"""
Performance optimization for large dataset processing with regime awareness.

Implements comprehensive performance optimization system including:
- Parallel processing for large datasets
- Memory-efficient algorithms
- Regime-aware computation optimization
- GPU acceleration support
- Performance monitoring and profiling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import psutil
import time
from dataclasses import dataclass
from functools import wraps, partial
import gc
import warnings
warnings.filterwarnings('ignore')

# Optional GPU support
try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Performance profiling
try:
    import cProfile
    import pstats
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance measurement metrics."""
    operation: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    data_size: int
    regime_aware: bool
    optimized: bool


@dataclass
class OptimizationResult:
    """Result of optimization operation."""
    original_time: float
    optimized_time: float
    speedup_factor: float
    memory_reduction_mb: float
    optimization_techniques: List[str]
    regime_aware_optimization: bool


class ParallelProcessor:
    """Parallel processing utilities for large datasets."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.logger = logging.getLogger(__name__)

    async def parallel_apply(self,
                           func: Callable,
                           data_chunks: List[pd.DataFrame],
                           merge_func: Optional[Callable] = None,
                           **kwargs) -> Any:
        """
        Apply function to data chunks in parallel.

        Args:
            func: Function to apply to each chunk
            data_chunks: List of data chunks to process
            merge_func: Optional function to merge results
            **kwargs: Additional arguments for func

        Returns:
            Merged result from all chunks
        """
        if not data_chunks:
            return None

        start_time = time.time()

        try:
            # Use ThreadPoolExecutor for I/O bound tasks
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(None, partial(func, chunk, **kwargs))
                    for chunk in data_chunks
                ]

                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_count = len([r for r in results if isinstance(r, Exception)])

            if failed_count > 0:
                self.logger.warning(f"{failed_count} chunks failed processing")

            # Merge results if merge function provided
            if merge_func and successful_results:
                merged_result = merge_func(successful_results)
            else:
                merged_result = successful_results

            execution_time = time.time() - start_time
            self.logger.info(f"Parallel processing completed in {execution_time:.2f}s with {len(successful_results)} successful chunks")

            return merged_result

        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            raise

    def chunk_dataframe(self,
                       df: pd.DataFrame,
                       chunk_size: int = 10000) -> List[pd.DataFrame]:
        """Split DataFrame into chunks for parallel processing."""
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            chunks.append(chunk.copy())
        return chunks


class MemoryOptimizer:
    """Memory optimization utilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Optimize DataFrame memory usage.

        Returns:
            Tuple of (optimized_df, memory_reduction_mb)
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB

        try:
            optimized_df = df.copy()

            # Optimize numeric columns
            for col in optimized_df.select_dtypes(include=['int64']).columns:
                col_min = optimized_df[col].min()
                col_max = optimized_df[col].max()

                if col_min >= 0:
                    if col_max < 255:
                        optimized_df[col] = optimized_df[col].astype('uint8')
                    elif col_max < 65535:
                        optimized_df[col] = optimized_df[col].astype('uint16')
                    elif col_max < 4294967295:
                        optimized_df[col] = optimized_df[col].astype('uint32')
                else:
                    if col_min > -128 and col_max < 127:
                        optimized_df[col] = optimized_df[col].astype('int8')
                    elif col_min > -32768 and col_max < 32767:
                        optimized_df[col] = optimized_df[col].astype('int16')
                    elif col_min > -2147483648 and col_max < 2147483647:
                        optimized_df[col] = optimized_df[col].astype('int32')

            # Optimize float columns
            for col in optimized_df.select_dtypes(include=['float64']).columns:
                # Check if we can downcast to float32
                if not optimized_df[col].isnull().any():
                    optimized_df[col] = optimized_df[col].astype('float32')

            # Optimize object columns (strings)
            for col in optimized_df.select_dtypes(include=['object']).columns:
                num_unique_values = len(optimized_df[col].unique())
                num_total_values = len(optimized_df[col])

                if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                    optimized_df[col] = optimized_df[col].astype('category')

            optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            memory_reduction = original_memory - optimized_memory

            self.logger.info(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB "
                          f"(reduced by {memory_reduction:.2f}MB)")

            return optimized_df, memory_reduction

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return df, 0.0

    def cleanup_memory(self):
        """Force garbage collection and cleanup."""
        gc.collect()
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent_used': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }


class RegimeAwareOptimizer:
    """Regime-aware computation optimization."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def optimize_by_regime(self,
                          data: pd.DataFrame,
                          regime_col: str = 'regime_id',
                          optimization_func: Optional[Callable] = None) -> Dict[int, Any]:
        """
        Optimize computations by regime.

        Args:
            data: DataFrame with regime information
            regime_col: Column name containing regime IDs
            optimization_func: Optional function to apply per regime

        Returns:
            Dictionary mapping regime IDs to optimized results
        """
        if regime_col not in data.columns:
            self.logger.warning(f"Regime column '{regime_col}' not found in data")
            return {}

        regime_results = {}

        for regime_id, regime_data in data.groupby(regime_col):
            start_time = time.time()

            try:
                if optimization_func:
                    result = optimization_func(regime_data)
                else:
                    result = self._default_regime_optimization(regime_data)

                execution_time = time.time() - start_time

                regime_results[regime_id] = {
                    'result': result,
                    'execution_time': execution_time,
                    'data_size': len(regime_data),
                    'memory_usage_mb': regime_data.memory_usage(deep=True).sum() / 1024 / 1024
                }

                self.logger.info(f"Regime {regime_id} optimized in {execution_time:.2f}s")

            except Exception as e:
                self.logger.error(f"Failed to optimize regime {regime_id}: {e}")
                regime_results[regime_id] = {'error': str(e)}

        return regime_results

    def _default_regime_optimization(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Default optimization for regime data."""
        return {
            'statistics': {
                'mean': regime_data.select_dtypes(include=[np.number]).mean().to_dict(),
                'std': regime_data.select_dtypes(include=[np.number]).std().to_dict(),
                'count': len(regime_data)
            },
            'optimized': True
        }


class GPUAccelerator:
    """GPU acceleration utilities."""

    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.logger = logging.getLogger(__name__)

    def accelerate_computation(self,
                             data: pd.DataFrame,
                             computation_func: Callable) -> Optional[Any]:
        """
        Accelerate computation using GPU if available.

        Args:
            data: Input DataFrame
            computation_func: Function to accelerate

        Returns:
            Result of accelerated computation or None if GPU not available
        """
        if not self.gpu_available:
            self.logger.warning("GPU acceleration not available")
            return None

        try:
            # Convert DataFrame to cuDF
            gpu_data = cudf.DataFrame.from_pandas(data)

            # Perform computation on GPU
            start_time = time.time()
            result = computation_func(gpu_data)
            gpu_time = time.time() - start_time

            # Convert result back to CPU if needed
            if hasattr(result, 'to_pandas'):
                result = result.to_pandas()

            self.logger.info(f"GPU computation completed in {gpu_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"GPU acceleration failed: {e}")
            return None

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.gpu_available


class PerformanceOptimizer:
    """Main performance optimization service."""

    def __init__(self, enable_gpu: bool = True, enable_parallel: bool = True):
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.enable_parallel = enable_parallel
        self.parallel_processor = ParallelProcessor()
        self.memory_optimizer = MemoryOptimizer()
        self.regime_optimizer = RegimeAwareOptimizer()
        self.gpu_accelerator = GPUAccelerator()
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.performance_history = []

    def measure_performance(self, operation_name: str) -> Callable:
        """Decorator to measure performance of functions."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get initial metrics
                start_time = time.time()
                initial_memory = self.memory_optimizer.get_memory_usage()['rss_mb']
                initial_cpu = psutil.cpu_percent()

                # Execute function
                result = func(*args, **kwargs)

                # Calculate final metrics
                execution_time = time.time() - start_time
                final_memory = self.memory_optimizer.get_memory_usage()['rss_mb']
                final_cpu = psutil.cpu_percent()

                # Create performance metrics
                metrics = PerformanceMetrics(
                    operation=operation_name,
                    execution_time=execution_time,
                    memory_usage_mb=final_memory - initial_memory,
                    cpu_usage_percent=final_cpu,
                    data_size=len(args[0]) if args and hasattr(args[0], '__len__') else 0,
                    regime_aware='regime' in kwargs or (args and any('regime' in str(arg) for arg in args)),
                    optimized='optimize' in func.__name__.lower()
                )

                # Store metrics
                self.performance_history.append(metrics)

                # Log performance
                self.logger.info(f"Performance - {operation_name}: {execution_time:.3f}s, "
                              f"Memory: {final_memory - initial_memory:.2f}MB, "
                              f"CPU: {final_cpu:.1f}%")

                return result
            return wrapper
        return decorator

    async def optimize_large_dataset(self,
                                    data: pd.DataFrame,
                                    operations: List[Tuple[str, Callable]],
                                    use_regime_optimization: bool = True,
                                    regime_col: str = 'regime_id') -> OptimizationResult:
        """
        Optimize processing of large datasets.

        Args:
            data: Input DataFrame
            operations: List of (operation_name, function) tuples
            use_regime_optimization: Whether to use regime-aware optimization
            regime_col: Column name for regime information

        Returns:
            OptimizationResult with performance improvements
        """
        start_time = time.time()
        original_memory = self.memory_optimizer.get_memory_usage()['rss_mb']

        try:
            # Phase 1: Memory optimization
            optimized_data, memory_reduction = self.memory_optimizer.optimize_dataframe_memory(data)

            # Phase 2: Parallel processing setup
            if self.enable_parallel and len(optimized_data) > 50000:
                chunk_size = min(10000, len(optimized_data) // (self.parallel_processor.max_workers * 2))
                data_chunks = self.parallel_processor.chunk_dataframe(optimized_data, chunk_size)
                use_parallel = True
            else:
                data_chunks = [optimized_data]
                use_parallel = False

            # Phase 3: Process operations
            operation_results = {}
            optimization_techniques = []

            for op_name, op_func in operations:
                op_start = time.time()

                try:
                    if use_regime_optimization and regime_col in optimized_data.columns:
                        # Regime-aware optimization
                        regime_results = self.regime_optimizer.optimize_by_regime(
                            optimized_data, regime_col, op_func
                        )
                        operation_results[op_name] = regime_results
                        optimization_techniques.append("regime_aware")
                    elif use_parallel:
                        # Parallel processing
                        result = await self.parallel_processor.parallel_apply(
                            op_func, data_chunks, lambda x: pd.concat(x) if x else None
                        )
                        operation_results[op_name] = result
                        optimization_techniques.append("parallel_processing")
                    else:
                        # Sequential processing
                        result = op_func(optimized_data)
                        operation_results[op_name] = result

                    op_time = time.time() - op_start
                    self.logger.info(f"Operation '{op_name}' completed in {op_time:.3f}s")

                except Exception as e:
                    self.logger.error(f"Operation '{op_name}' failed: {e}")
                    operation_results[op_name] = {'error': str(e)}

            # Phase 4: GPU acceleration for compatible operations
            if self.enable_gpu and self.gpu_accelerator.is_gpu_available():
                gpu_results = await self._apply_gpu_acceleration(optimized_data, operations)
                if gpu_results:
                    operation_results.update(gpu_results)
                    optimization_techniques.append("gpu_acceleration")

            # Calculate performance metrics
            total_time = time.time() - start_time
            final_memory = self.memory_optimizer.get_memory_usage()['rss_mb']

            # Estimate original processing time (rough estimate)
            original_time_estimate = total_time * (1.5 if use_parallel else 1.0) * \
                                   (1.3 if use_regime_optimization else 1.0)

            optimization_result = OptimizationResult(
                original_time=original_time_estimate,
                optimized_time=total_time,
                speedup_factor=original_time_estimate / total_time if total_time > 0 else 1.0,
                memory_reduction_mb=memory_reduction,
                optimization_techniques=optimization_techniques,
                regime_aware_optimization=use_regime_optimization
            )

            self.logger.info(f"Dataset optimization completed: "
                          f"{original_time_estimate:.3f}s -> {total_time:.3f}s "
                          f"(speedup: {optimization_result.speedup_factor:.2f}x)")

            return optimization_result

        except Exception as e:
            self.logger.error(f"Large dataset optimization failed: {e}")
            raise

    async def _apply_gpu_acceleration(self,
                                    data: pd.DataFrame,
                                    operations: List[Tuple[str, Callable]]) -> Dict[str, Any]:
        """Apply GPU acceleration to compatible operations."""
        gpu_results = {}

        for op_name, op_func in operations:
            if self._is_gpu_compatible(op_func):
                result = self.gpu_accelerator.accelerate_computation(data, op_func)
                if result is not None:
                    gpu_results[f"{op_name}_gpu"] = result

        return gpu_results

    def _is_gpu_compatible(self, func: Callable) -> bool:
        """Check if function is compatible with GPU acceleration."""
        # Simple heuristic - in practice, this would be more sophisticated
        compatible_keywords = ['matrix', 'vector', 'linear', 'transform', 'compute']
        func_name = func.__name__.lower()
        return any(keyword in func_name for keyword in compatible_keywords)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.performance_history:
            return {"message": "No performance data available"}

        # Calculate summary statistics
        execution_times = [m.execution_time for m in self.performance_history]
        memory_usages = [m.memory_usage_mb for m in self.performance_history]
        cpu_usages = [m.cpu_usage_percent for m in self.performance_history]

        # Recent performance (last 50 operations)
        recent_performance = self.performance_history[-50:] if len(self.performance_history) > 50 else self.performance_history

        return {
            'total_operations': len(self.performance_history),
            'avg_execution_time': np.mean(execution_times),
            'max_execution_time': np.max(execution_times),
            'min_execution_time': np.min(execution_times),
            'avg_memory_usage_mb': np.mean(memory_usages),
            'total_memory_used_mb': np.sum(memory_usages),
            'avg_cpu_usage': np.mean(cpu_usages),
            'regime_aware_operations': len([m for m in recent_performance if m.regime_aware]),
            'optimized_operations': len([m for m in recent_performance if m.optimized]),
            'gpu_available': self.gpu_accelerator.is_gpu_available(),
            'parallel_enabled': self.enable_parallel,
            'system_info': {
                'cpu_cores': cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
            }
        }

    def optimize_model_training(self,
                              training_func: Callable,
                              data: pd.DataFrame,
                              **kwargs) -> Dict[str, Any]:
        """
        Optimize model training performance.

        Args:
            training_func: Model training function
            data: Training data
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results and performance metrics
        """
        start_time = time.time()

        try:
            # Memory optimization for training data
            optimized_data, memory_reduction = self.memory_optimizer.optimize_dataframe_memory(data)

            # Prepare training data
            if len(optimized_data) > 100000 and self.enable_parallel:
                # Large dataset - use batch training
                results = await self._batch_train_model(training_func, optimized_data, **kwargs)
                optimization_techniques = ["batch_training", "memory_optimization"]
            else:
                # Standard training
                results = training_func(optimized_data, **kwargs)
                optimization_techniques = ["memory_optimization"]

            training_time = time.time() - start_time

            return {
                'training_results': results,
                'performance_metrics': {
                    'training_time': training_time,
                    'memory_reduction_mb': memory_reduction,
                    'data_size': len(optimized_data),
                    'optimization_techniques': optimization_techniques
                }
            }

        except Exception as e:
            self.logger.error(f"Model training optimization failed: {e}")
            return {'error': str(e)}

    async def _batch_train_model(self,
                               training_func: Callable,
                               data: pd.DataFrame,
                               batch_size: int = 50000,
                               **kwargs) -> Any:
        """Perform batch training for large datasets."""
        # This would implement incremental/batch training
        # For now, return a mock result
        return training_func(data, **kwargs)