"""
API test client for portfolio optimization system.

Provides a simple Python client for testing the API endpoints.
Simple, clean implementation avoiding overengineering for resume projects.
"""

import requests
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# API Configuration
BASE_URL = "http://localhost:8000/api/v1"
API_TIMEOUT = 30  # seconds


class PortfolioAPIClient:
    """Simple API client for portfolio optimization endpoints."""

    def __init__(self, base_url: str = BASE_URL, timeout: int = API_TIMEOUT):
        """Initialize API client."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Portfolio-Optimization-Client/1.0'
        })

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request and return response."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            start_time = time.time()
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                timeout=self.timeout
            )
            response_time = time.time() - start_time

            result = {
                'status_code': response.status_code,
                'response_time': response_time,
                'success': response.status_code < 400,
                'data': response.json() if response.content else None,
                'headers': dict(response.headers)
            }

            if not result['success']:
                result['error'] = response.text

            return result

        except requests.exceptions.RequestException as e:
            return {
                'status_code': 0,
                'response_time': 0,
                'success': False,
                'error': str(e),
                'data': None
            }

    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        return self._make_request('GET', '/health')

    def optimize_portfolio(self, assets: List[Dict], constraints: Optional[Dict] = None,
                          method: str = "mean_variance", objective: str = "sharpe") -> Dict[str, Any]:
        """Optimize portfolio."""
        data = {
            'assets': assets,
            'method': method,
            'objective': objective
        }

        if constraints:
            data['constraints'] = constraints

        return self._make_request('POST', '/portfolio/optimize', data)

    def analyze_portfolio(self, weights: Dict[str, float], assets: List[Dict],
                         benchmark_symbol: Optional[str] = None) -> Dict[str, Any]:
        """Analyze portfolio."""
        data = {
            'weights': weights,
            'assets': assets
        }

        if benchmark_symbol:
            data['benchmark_symbol'] = benchmark_symbol

        return self._make_request('POST', '/portfolio/analyze', data)

    def get_asset_details(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Get asset details."""
        return self._make_request('GET', f'/data/assets/{symbol}?period={period}')

    def search_assets(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for assets."""
        return self._make_request('GET', f'/data/assets/search?query={query}&limit={limit}')

    def get_market_overview(self, symbols: str = "SPY,AAPL,MSFT,GOOGL,AMZN") -> Dict[str, Any]:
        """Get market overview."""
        return self._make_request('GET', f'/data/assets/market-overview?symbols={symbols}')

    def get_optimization_methods(self) -> Dict[str, Any]:
        """Get available optimization methods."""
        return self._make_request('GET', '/portfolio/methods')

    def backtest_portfolio(self, weights: Dict[str, float], assets: List[Dict],
                          start_date: Optional[str] = None, end_date: Optional[str] = None,
                          benchmark_symbol: Optional[str] = None) -> Dict[str, Any]:
        """Backtest portfolio."""
        import json

        endpoint = '/portfolio/analyze/backtest'
        params = []

        params.append(f'weights={json.dumps(weights)}')
        params.append(f'assets={json.dumps(assets)}')

        if start_date:
            params.append(f'start_date={start_date}')
        if end_date:
            params.append(f'end_date={end_date}')
        if benchmark_symbol:
            params.append(f'benchmark_symbol={benchmark_symbol}')

        url = f"{endpoint}?{'&'.join(params)}"
        return self._make_request('GET', url)


def print_test_results(test_name: str, result: Dict[str, Any]):
    """Print test results in a formatted way."""
    print(f"\n=== {test_name} ===")
    print(f"Status Code: {result['status_code']}")
    print(f"Response Time: {result['response_time']:.3f}s")
    print(f"Success: {result['success']}")

    if result['success']:
        if 'data' in result and result['data']:
            print("Response Data:")
            print(json.dumps(result['data'], indent=2))
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


def run_api_tests():
    """Run comprehensive API tests."""
    print("Starting Portfolio Optimization API Tests")
    print("=" * 50)

    client = PortfolioAPIClient()

    # Test 1: Health Check
    result = client.health_check()
    print_test_results("Health Check", result)

    if not result['success']:
        print("❌ API is not healthy. Stopping tests.")
        return

    # Test 2: Get Optimization Methods
    result = client.get_optimization_methods()
    print_test_results("Get Optimization Methods", result)

    # Test 3: Asset Search
    result = client.search_assets("AAPL", limit=5)
    print_test_results("Search Assets (AAPL)", result)

    # Test 4: Get Asset Details
    result = client.get_asset_details("SPY", period="6mo")
    print_test_results("Get SPY Details", result)

    # Test 5: Market Overview
    result = client.get_market_overview()
    print_test_results("Market Overview", result)

    # Test 6: Portfolio Optimization
    assets = [
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "asset_type": "etf"},
        {"symbol": "AAPL", "name": "Apple Inc.", "asset_type": "stock"},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "asset_type": "stock"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "asset_type": "stock"}
    ]

    constraints = {
        "max_position_size": 0.4,
        "min_position_size": 0.05,
        "max_volatility": 0.2,
        "risk_free_rate": 0.02
    }

    result = client.optimize_portfolio(
        assets=assets,
        constraints=constraints,
        method="mean_variance",
        objective="sharpe"
    )
    print_test_results("Portfolio Optimization", result)

    # Test 7: Portfolio Analysis (if optimization was successful)
    if result['success'] and 'data' in result and result['data'].get('optimal_weights'):
        weights = result['data']['optimal_weights']
        result = client.analyze_portfolio(
            weights=weights,
            assets=assets,
            benchmark_symbol="SPY"
        )
        print_test_results("Portfolio Analysis", result)

    # Test 8: Backtest Portfolio
    weights = {"SPY": 0.4, "AAPL": 0.3, "MSFT": 0.2, "GOOGL": 0.1}
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    result = client.backtest_portfolio(
        weights=weights,
        assets=assets,
        start_date=start_date,
        end_date=end_date,
        benchmark_symbol="SPY"
    )
    print_test_results("Portfolio Backtest", result)

    print("\n" + "=" * 50)
    print("✅ API Tests Completed!")


if __name__ == "__main__":
    run_api_tests()