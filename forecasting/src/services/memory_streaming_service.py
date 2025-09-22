"""
Enhanced memory usage optimization with streaming for large datasets.

Implements comprehensive memory management and streaming processing system:
- Chunked streaming for large datasets
- Memory-efficient algorithms
- Real-time memory monitoring
- Automatic memory cleanup
- Adaptive batch size optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Iterator, Callable, Union
import asyncio
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import gc
import psutil
import os
from pathlib import Path
import tempfile
import json
from collections import deque

# Optional GPU support
try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class StreamChunk:
    """Individual chunk of streaming data."""
    chunk_id: int
    data: pd.DataFrame
    metadata: Dict[str, Any]
    timestamp: str
    memory_size_mb: float


@dataclass
class StreamConfig:
    """Configuration for streaming operations."""
    chunk_size: int = 10000
    max_memory_usage_mb: float = 4096.0
    enable_compression: bool = True
    enable_gpu: bool = True
    cleanup_interval: int = 100
    adaptive_chunking: bool = True
    temp_storage_dir: Optional[str] = None


class MemoryMonitor:
    """Real-time memory usage monitoring."""

    def __init__(self, alert_threshold_mb: float = 2048.0):
        self.alert_threshold_mb = alert_threshold_mb
        self.memory_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)

    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()

        usage_stats = {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent_used': process.memory_percent(),
            'system_available_mb': virtual_memory.available / 1024 / 1024,
            'system_total_mb': virtual_memory.total / 1024 / 1024,
            'timestamp': datetime.now().isoformat()
        }

        # Store in history
        self.memory_history.append(usage_stats)

        return usage_stats

    def check_memory_alerts(self) -> List[str]:
        """Check for memory usage alerts."""
        alerts = []
        current_usage = self.get_current_usage()

        if current_usage['rss_mb'] > self.alert_threshold_mb:
            alerts.append(f"High memory usage: {current_usage['rss_mb']:.1f}MB > {self.alert_threshold_mb:.1f}MB")

        if current_usage['system_available_mb'] < 1024:  # Less than 1GB available
            alerts.append(f"Low system memory: {current_usage['system_available_mb']:.1f}MB available")

        if current_usage['percent_used'] > 80:
            alerts.append(f"High process memory percentage: {current_usage['percent_used']:.1f}%")

        return alerts

    def get_memory_trends(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Analyze memory usage trends."""
        if len(self.memory_history) < 10:
            return {"message": "Insufficient data for trend analysis"}

        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_data = [m for m in self.memory_history
                      if datetime.fromisoformat(m['timestamp']) > cutoff_time]

        if not recent_data:
            return {"message": "No recent memory data"}

        rss_values = [m['rss_mb'] for m in recent_data]
        percent_values = [m['percent_used'] for m in recent_data]

        return {
            'analysis_period_minutes': window_minutes,
            'data_points': len(recent_data),
            'current_rss_mb': recent_data[-1]['rss_mb'],
            'avg_rss_mb': np.mean(rss_values),
            'max_rss_mb': np.max(rss_values),
            'min_rss_mb': np.min(rss_values),
            'rss_trend': 'increasing' if rss_values[-1] > rss_values[0] else 'decreasing',
            'avg_cpu_percent': np.mean(percent_values),
            'memory_efficiency': 1.0 - (np.std(rss_values) / np.mean(rss_values)) if np.mean(rss_values) > 0 else 0
        }


class DataStreamer(ABC):
    """Abstract base class for data streaming."""

    @abstractmethod
    def create_stream(self, data_source: Any, config: StreamConfig) -> Iterator[StreamChunk]:
        """Create data stream from source."""
        pass

    @abstractmethod
    def process_chunk(self, chunk: StreamChunk, processing_func: Callable) -> StreamChunk:
        """Process a single chunk."""
        pass


class CSVStreamer(DataStreamer):
    """CSV file streaming implementation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_stream(self, file_path: str, config: StreamConfig) -> Iterator[StreamChunk]:
        """Create stream from CSV file."""
        try:
            # Get file size for progress tracking
            file_size_mb = os.path.getsize(file_path) / 1024 / 1024
            chunk_id = 0

            # Stream CSV in chunks
            for chunk in pd.read_csv(file_path, chunksize=config.chunk_size):
                chunk_memory = chunk.memory_usage(deep=True).sum() / 1024 / 1024

                stream_chunk = StreamChunk(
                    chunk_id=chunk_id,
                    data=chunk,
                    metadata={
                        'file_path': file_path,
                        'file_size_mb': file_size_mb,
                        'chunk_memory_mb': chunk_memory,
                        'rows_processed': len(chunk)
                    },
                    timestamp=datetime.now().isoformat(),
                    memory_size_mb=chunk_memory
                )

                yield stream_chunk
                chunk_id += 1

        except Exception as e:
            self.logger.error(f"CSV streaming failed for {file_path}: {e}")
            raise

    def process_chunk(self, chunk: StreamChunk, processing_func: Callable) -> StreamChunk:
        """Process CSV chunk."""
        try:
            processed_data = processing_func(chunk.data)
            new_memory_size = processed_data.memory_usage(deep=True).sum() / 1024 / 1024

            return StreamChunk(
                chunk_id=chunk.chunk_id,
                data=processed_data,
                metadata={
                    **chunk.metadata,
                    'processed': True,
                    'original_memory_mb': chunk.memory_size_mb,
                    'processed_memory_mb': new_memory_size
                },
                timestamp=datetime.now().isoformat(),
                memory_size_mb=new_memory_size
            )

        except Exception as e:
            self.logger.error(f"Chunk processing failed: {e}")
            raise


class DataFrameStreamer(DataStreamer):
    """DataFrame streaming implementation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_stream(self, df: pd.DataFrame, config: StreamConfig) -> Iterator[StreamChunk]:
        """Create stream from DataFrame."""
        try:
            total_rows = len(df)
            chunk_id = 0

            # Calculate adaptive chunk size if enabled
            if config.adaptive_chunking:
                chunk_size = self._calculate_adaptive_chunk_size(df, config)
            else:
                chunk_size = config.chunk_size

            for i in range(0, total_rows, chunk_size):
                chunk_data = df.iloc[i:i + chunk_size].copy()
                chunk_memory = chunk_data.memory_usage(deep=True).sum() / 1024 / 1024

                stream_chunk = StreamChunk(
                    chunk_id=chunk_id,
                    data=chunk_data,
                    metadata={
                        'total_rows': total_rows,
                        'chunk_start': i,
                        'chunk_end': min(i + chunk_size, total_rows),
                        'chunk_memory_mb': chunk_memory
                    },
                    timestamp=datetime.now().isoformat(),
                    memory_size_mb=chunk_memory
                )

                yield stream_chunk
                chunk_id += 1

        except Exception as e:
            self.logger.error(f"DataFrame streaming failed: {e}")
            raise

    def _calculate_adaptive_chunk_size(self, df: pd.DataFrame, config: StreamConfig) -> int:
        """Calculate adaptive chunk size based on available memory."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        available_memory = config.max_memory_usage_mb - current_memory

        # Estimate memory per row
        sample_rows = min(1000, len(df))
        sample_memory = df.head(sample_rows).memory_usage(deep=True).sum() / 1024 / 1024
        memory_per_row = sample_memory / sample_rows

        # Calculate safe chunk size with safety margin
        safe_memory = available_memory * 0.7  # 70% safety margin
        adaptive_chunk_size = max(100, int(safe_memory / memory_per_row))

        return min(adaptive_chunk_size, config.chunk_size * 2)

    def process_chunk(self, chunk: StreamChunk, processing_func: Callable) -> StreamChunk:
        """Process DataFrame chunk."""
        try:
            processed_data = processing_func(chunk.data)
            new_memory_size = processed_data.memory_usage(deep=True).sum() / 1024 / 1024

            return StreamChunk(
                chunk_id=chunk.chunk_id,
                data=processed_data,
                metadata={
                    **chunk.metadata,
                    'processed': True,
                    'original_memory_mb': chunk.memory_size_mb,
                    'processed_memory_mb': new_memory_size
                },
                timestamp=datetime.now().isoformat(),
                memory_size_mb=new_memory_size
            )

        except Exception as e:
            self.logger.error(f"Chunk processing failed: {e}")
            raise


class MemoryOptimizer:
    """Advanced memory optimization utilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def optimize_chunk_memory(self, chunk: StreamChunk) -> StreamChunk:
        """Optimize memory usage for a chunk."""
        try:
            original_memory = chunk.memory_size_mb

            # Apply memory optimization
            optimized_data = self._optimize_dataframe_memory(chunk.data)
            new_memory_size = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024

            return StreamChunk(
                chunk_id=chunk.chunk_id,
                data=optimized_data,
                metadata={
                    **chunk.metadata,
                    'memory_optimized': True,
                    'original_memory_mb': original_memory,
                    'optimized_memory_mb': new_memory_size,
                    'memory_reduction_mb': original_memory - new_memory_size
                },
                timestamp=chunk.timestamp,
                memory_size_mb=new_memory_size
            )

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return chunk

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        optimized_df = df.copy()

        # Downcast numeric columns
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

        # Downcast float columns
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = optimized_df[col].astype('float32')

        # Convert object columns to category if appropriate
        for col in optimized_df.select_dtypes(include=['object']).columns:
            num_unique = len(optimized_df[col].unique())
            if num_unique / len(optimized_df[col]) < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')

        return optimized_df


class StreamingProcessor:
    """Main streaming processing service."""

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.memory_monitor = MemoryMonitor(self.config.max_memory_usage_mb * 0.8)
        self.memory_optimizer = MemoryOptimizer()
        self.csv_streamer = CSVStreamer()
        self.df_streamer = DataFrameStreamer()
        self.logger = logging.getLogger(__name__)

        # Processing statistics
        self.processing_stats = {
            'total_chunks_processed': 0,
            'total_memory_used_mb': 0,
            'total_processing_time': 0,
            'failed_chunks': 0
        }

    async def process_streaming_data(self,
                                   data_source: Union[str, pd.DataFrame],
                                   processing_func: Callable,
                                   streamer_type: str = "dataframe") -> Dict[str, Any]:
        """
        Process data using streaming approach.

        Args:
            data_source: Path to file or DataFrame
            processing_func: Function to apply to each chunk
            streamer_type: Type of streamer ('csv' or 'dataframe')

        Returns:
            Processing results and statistics
        """
        start_time = time.time()
        processed_chunks = []
        processing_errors = []

        try:
            # Select appropriate streamer
            if streamer_type == "csv" and isinstance(data_source, str):
                streamer = self.csv_streamer
                stream = streamer.create_stream(data_source, self.config)
            elif streamer_type == "dataframe" and isinstance(data_source, pd.DataFrame):
                streamer = self.df_streamer
                stream = streamer.create_stream(data_source, self.config)
            else:
                raise ValueError("Invalid data source or streamer type combination")

            # Process stream
            async for chunk in self._async_stream_generator(stream):
                try:
                    # Check memory usage
                    memory_alerts = self.memory_monitor.check_memory_alerts()
                    if memory_alerts:
                        self.logger.warning(f"Memory alerts: {memory_alerts}")
                        await self._perform_memory_cleanup()

                    # Optimize chunk memory
                    optimized_chunk = self.memory_optimizer.optimize_chunk_memory(chunk)

                    # Process chunk
                    processed_chunk = streamer.process_chunk(optimized_chunk, processing_func)
                    processed_chunks.append(processed_chunk)

                    # Update statistics
                    self.processing_stats['total_chunks_processed'] += 1
                    self.processing_stats['total_memory_used_mb'] += processed_chunk.memory_size_mb

                    # Periodic cleanup
                    if (self.processing_stats['total_chunks_processed'] %
                        self.config.cleanup_interval == 0):
                        await self._perform_memory_cleanup()

                except Exception as e:
                    self.logger.error(f"Chunk {chunk.chunk_id} processing failed: {e}")
                    processing_errors.append({
                        'chunk_id': chunk.chunk_id,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.processing_stats['failed_chunks'] += 1

            # Compile results
            total_time = time.time() - start_time
            self.processing_stats['total_processing_time'] = total_time

            results = {
                'processing_successful': len(processed_chunks) > 0,
                'total_chunks': len(processed_chunks),
                'failed_chunks': len(processing_errors),
                'processing_time': total_time,
                'average_chunk_time': total_time / len(processed_chunks) if processed_chunks else 0,
                'total_memory_used_mb': self.processing_stats['total_memory_used_mb'],
                'average_memory_per_chunk_mb': (
                    self.processing_stats['total_memory_used_mb'] / len(processed_chunks)
                    if processed_chunks else 0
                ),
                'processing_errors': processing_errors,
                'final_memory_usage': self.memory_monitor.get_current_usage(),
                'memory_trends': self.memory_monitor.get_memory_trends()
            }

            # Combine processed chunks if possible
            if processed_chunks and all(isinstance(c.data, pd.DataFrame) for c in processed_chunks):
                combined_data = pd.concat([c.data for c in processed_chunks], ignore_index=True)
                results['combined_result'] = combined_data
                results['combined_memory_mb'] = combined_data.memory_usage(deep=True).sum() / 1024 / 1024

            self.logger.info(f"Streaming processing completed: {len(processed_chunks)} chunks "
                          f"processed in {total_time:.2f}s")

            return results

        except Exception as e:
            self.logger.error(f"Streaming processing failed: {e}")
            return {'error': str(e), 'processing_errors': processing_errors}

    async def _async_stream_generator(self, stream: Iterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        """Convert synchronous stream to async generator."""
        for chunk in stream:
            yield chunk
            # Small delay to allow other async operations
            await asyncio.sleep(0.001)

    async def _perform_memory_cleanup(self):
        """Perform memory cleanup operations."""
        try:
            # Force garbage collection
            gc.collect()

            # Clean up GPU memory if available
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()

            # Log cleanup
            current_memory = self.memory_monitor.get_current_usage()
            self.logger.debug(f"Memory cleanup performed. Current usage: {current_memory['rss_mb']:.1f}MB")

        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")

    async def stream_model_training(self,
                                  training_data: pd.DataFrame,
                                  training_func: Callable,
                                  model_type: str = "incremental") -> Dict[str, Any]:
        """
        Perform streaming model training.

        Args:
            training_data: Training dataset
            training_func: Model training function
            model_type: Type of training ('incremental', 'batch', 'online')

        Returns:
            Training results and performance metrics
        """
        start_time = time.time()

        try:
            if model_type == "incremental":
                # Incremental learning with chunks
                model_results = await self._incremental_training(training_data, training_func)
            elif model_type == "batch":
                # Batch training with memory optimization
                model_results = await self._batch_training_with_memory_opt(training_data, training_func)
            elif model_type == "online":
                # Online learning
                model_results = await self._online_learning_training(training_data, training_func)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            total_time = time.time() - start_time
            final_memory = self.memory_monitor.get_current_usage()

            return {
                'model_type': model_type,
                'training_successful': True,
                'training_time': total_time,
                'final_memory_usage_mb': final_memory['rss_mb'],
                'model_results': model_results,
                'performance_metrics': {
                    'chunks_processed': self.processing_stats['total_chunks_processed'],
                    'memory_efficiency': (
                        training_data.memory_usage(deep=True).sum() / 1024 / 1024 / final_memory['rss_mb']
                    )
                }
            }

        except Exception as e:
            self.logger.error(f"Streaming model training failed: {e}")
            return {'error': str(e)}

    async def _incremental_training(self, data: pd.DataFrame, training_func: Callable):
        """Perform incremental model training."""
        model = None
        stream = self.df_streamer.create_stream(data, self.config)

        async for chunk in stream:
            if model is None:
                # Initial training
                model = training_func(chunk.data, initial=True)
            else:
                # Incremental update
                model = training_func(chunk.data, model=model, update=True)

        return model

    async def _batch_training_with_memory_opt(self, data: pd.DataFrame, training_func: Callable):
        """Perform batch training with memory optimization."""
        optimized_data = self.memory_optimizer._optimize_dataframe_memory(data)
        return training_func(optimized_data)

    async def _online_learning_training(self, data: pd.DataFrame, training_func: Callable):
        """Perform online learning training."""
        stream = self.df_streamer.create_stream(data, self.config)
        model = None

        async for chunk in stream:
            model = training_func(chunk.data, model=model, online=True)

        return model

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary."""
        return {
            'processing_statistics': self.processing_stats,
            'memory_status': self.memory_monitor.get_current_usage(),
            'memory_trends': self.memory_monitor.get_memory_trends(),
            'config_status': {
                'chunk_size': self.config.chunk_size,
                'max_memory_mb': self.config.max_memory_usage_mb,
                'gpu_enabled': self.config.enable_gpu and GPU_AVAILABLE,
                'adaptive_chunking': self.config.adaptive_chunking
            }
        }