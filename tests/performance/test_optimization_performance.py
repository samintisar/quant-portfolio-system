#!/usr/bin/env python3
"""
Performance tests for portfolio optimization algorithms.

Tests the performance, scalability, and memory usage of optimization methods.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
import time
import psutil
import tracemalloc
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.optimizer.optimizer import SimplePortfolioOptimizer


class TestOptimizationPerformance:
    """Performance tests for optimization algorithms."""

    def setup_method(self):
        """Set up test environment."""
        self.optimizer = SimplePortfolioOptimizer()
        self.process = psutil.Process()

    def _generate_test_data(self, n_assets: int, n_samples: int = 252) -> pd.DataFrame:
        """Generate synthetic return data for testing."""
        np.random.seed(42)

        # Generate correlated returns
        mean_returns = np.random.normal(0.001, 0.005, n_assets)

        # Create correlation matrix
        corr_matrix = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr_matrix[i, j] = corr_matrix[j, i] = np.random.uniform(-0.3, 0.7)

        # Generate covariance matrix
        vols = np.random.uniform(0.15, 0.35, n_assets)
        cov_matrix = np.diag(vols) @ corr_matrix @ np.diag(vols)

        # Generate returns
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_samples)

        # Convert to DataFrame with proper index
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='B')
        symbols = [f'Asset_{i}' for i in range(n_assets)]

        return pd.DataFrame(returns, index=dates, columns=symbols)

    def test_mean_variance_optimization_performance(self):
        """Test performance of mean-variance optimization."""
        # Test with different portfolio sizes
        portfolio_sizes = [5, 10, 20, 50]

        for n_assets in portfolio_sizes:
            returns = self._generate_test_data(n_assets)

            # Measure memory usage
            tracemalloc.start()
            start_time = time.time()

            # Run optimization
            result = self.optimizer.mean_variance_optimize(returns)

            # Measure performance metrics
            execution_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Validate results
            assert result is not None
            assert 'weights' in result
            assert len(result['weights']) == n_assets

            # Check performance requirements
            assert execution_time < 5.0, f"Mean-variance optimization took {execution_time:.2f}s for {n_assets} assets"
            assert peak < 50 * 1024 * 1024, f"Peak memory usage {peak / 1024 / 1024:.1f}MB too high for {n_assets} assets"

            # Check weight constraints
            weights = list(result['weights'].values())
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights should sum to 1"
            assert all(w >= 0 for w in weights), "Weights should be non-negative"

            print(f"Mean-variance optimization for {n_assets} assets: {execution_time:.3f}s, {peak / 1024 / 1024:.1f}MB")

    def test_cvar_optimization_performance(self):
        """Test performance of CVaR optimization."""
        portfolio_sizes = [5, 10, 20]

        for n_assets in portfolio_sizes:
            returns = self._generate_test_data(n_assets, 500)  # More samples for CVaR

            tracemalloc.start()
            start_time = time.time()

            result = self.optimizer.cvar_optimize(returns, alpha=0.05)

            execution_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            assert result is not None
            assert 'weights' in result
            assert len(result['weights']) == n_assets

            # CVaR optimization can be slower, so allow more time
            assert execution_time < 10.0, f"CVaR optimization took {execution_time:.2f}s for {n_assets} assets"
            assert peak < 100 * 1024 * 1024, f"Peak memory usage {peak / 1024 / 1024:.1f}MB too high for {n_assets} assets"

            print(f"CVaR optimization for {n_assets} assets: {execution_time:.3f}s, {peak / 1024 / 1024:.1f}MB")

    def test_black_litterman_optimization_performance(self):
        """Test performance of Black-Litterman optimization."""
        portfolio_sizes = [5, 10, 20]

        for n_assets in portfolio_sizes:
            returns = self._generate_test_data(n_assets)

            # Create some market views
            from portfolio.models.views import MarketViewCollection, MarketView
            views = []
            for i in range(min(3, n_assets)):
                views.append(MarketView(
                    asset_symbol=f'Asset_{i}',
                    expected_return=np.random.uniform(0.08, 0.15),
                    confidence=np.random.uniform(0.5, 0.9)
                ))
            market_views = MarketViewCollection(views)

            tracemalloc.start()
            start_time = time.time()

            result = self.optimizer.black_litterman_optimize(returns, market_views)

            execution_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            assert result is not None
            assert 'weights' in result
            assert len(result['weights']) == n_assets

            assert execution_time < 8.0, f"Black-Litterman optimization took {execution_time:.2f}s for {n_assets} assets"
            assert peak < 75 * 1024 * 1024, f"Peak memory usage {peak / 1024 / 1024:.1f}MB too high for {n_assets} assets"

            print(f"Black-Litterman optimization for {n_assets} assets: {execution_time:.3f}s, {peak / 1024 / 1024:.1f}MB")

    def test_efficient_frontier_performance(self):
        """Test performance of efficient frontier calculation."""
        portfolio_sizes = [5, 10]

        for n_assets in portfolio_sizes:
            returns = self._generate_test_data(n_assets)

            tracemalloc.start()
            start_time = time.time()

            frontier = self.optimizer.get_efficient_frontier([f'Asset_{i}' for i in range(n_assets)], n_points=10)

            execution_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            assert len(frontier) == 10
            assert all('return' in point and 'volatility' in point for point in frontier)

            # Efficient frontier can be slower due to multiple optimizations
            assert execution_time < 15.0, f"Efficient frontier took {execution_time:.2f}s for {n_assets} assets"
            assert peak < 100 * 1024 * 1024, f"Peak memory usage {peak / 1024 / 1024:.1f}MB too high for {n_assets} assets"

            print(f"Efficient frontier for {n_assets} assets: {execution_time:.3f}s, {peak / 1024 / 1024:.1f}MB")

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Test with 100 assets and 1000 data points
        returns = self._generate_test_data(100, 1000)

        start_time = time.time()
        start_memory = self.process.memory_info().rss

        result = self.optimizer.mean_variance_optimize(returns)

        execution_time = time.time() - start_time
        end_memory = self.process.memory_info().rss
        memory_increase = (end_memory - start_memory) / 1024 / 1024  # MB

        assert result is not None
        assert len(result['weights']) == 100
        assert execution_time < 30.0, f"Large dataset optimization took {execution_time:.2f}s"
        assert memory_increase < 500, f"Memory increase {memory_increase:.1f}MB too high"

        print(f"Large dataset (100 assets, 1000 samples): {execution_time:.3f}s, {memory_increase:.1f}MB memory increase")

    def test_concurrent_optimization(self):
        """Test performance under concurrent load."""
        import threading
        import queue

        def worker(returns, results_queue, worker_id):
            try:
                result = self.optimizer.mean_variance_optimize(returns)
                results_queue.put(('success', worker_id, result))
            except Exception as e:
                results_queue.put(('error', worker_id, str(e)))

        # Generate test data
        returns = self._generate_test_data(10)

        # Run multiple optimizations concurrently
        results_queue = queue.Queue()
        threads = []

        start_time = time.time()

        for i in range(5):
            thread = threading.Thread(target=worker, args=(returns, results_queue, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        execution_time = time.time() - start_time

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == 5
        assert all(result[0] == 'success' for result in results)
        assert execution_time < 10.0, f"Concurrent optimizations took {execution_time:.2f}s"

        print(f"Concurrent optimizations (5 workers): {execution_time:.3f}s")

    def test_optimization_accuracy(self):
        """Test mathematical accuracy of optimization results."""
        returns = self._generate_test_data(5, 1000)

        # Test that optimization produces consistent results
        results = []
        for _ in range(10):
            result = self.optimizer.mean_variance_optimize(returns)
            results.append(result)

        # Check that all results are valid
        assert all(r is not None for r in results)
        assert all('weights' in r for r in results)

        # Check consistency (with same seed, results should be identical)
        weights_sets = [tuple(r['weights'].values()) for r in results]
        assert len(set(weights_sets)) == 1, "Optimization should be deterministic with fixed seed"

        # Test mathematical properties
        weights = results[0]['weights']
        assert abs(sum(weights.values()) - 1.0) < 1e-10, "Weights should sum to 1"
        assert all(w >= -1e-10 for w in weights.values()), "Weights should be non-negative"

        print(f"Optimization accuracy test passed: deterministic results with proper constraints")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])