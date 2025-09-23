"""
Simple API for portfolio optimization with basic endpoints only.

Provides essential functionality without overengineered abstractions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import logging
import pandas as pd
from fastapi.responses import JSONResponse

from portfolio.optimizer.optimizer import SimplePortfolioOptimizer
from portfolio.performance.calculator import SimplePerformanceCalculator
from portfolio.data.yahoo_service import YahooFinanceService
from portfolio.models.asset import Asset
from portfolio.models.constraints import PortfolioConstraints
from portfolio.config import get_config
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
optimizer = SimplePortfolioOptimizer()
performance_calc = SimplePerformanceCalculator()
data_service = YahooFinanceService()
config = get_config()

# Create FastAPI app
app = FastAPI(
    title="Simple Portfolio Optimization API",
    description="Basic portfolio optimization without overengineering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AssetModel(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None

class ConstraintModel(BaseModel):
    max_position_size: Optional[float] = 0.1
    max_sector_concentration: Optional[float] = 0.3
    max_volatility: Optional[float] = 0.3
    risk_free_rate: Optional[float] = 0.02

class MarketViewModel(BaseModel):
    symbol: str
    view_type: str = "absolute"
    confidence: float = 0.5
    expected_return: float

class OptimizeRequest(BaseModel):
    assets: List[AssetModel]
    method: Literal["mean_variance", "cvar", "black_litterman"] = "mean_variance"
    objective: str = "sharpe"
    constraints: Optional[ConstraintModel] = None
    market_views: Optional[List[MarketViewModel]] = None
    lookback_period: Optional[int] = 252

class AnalyzeRequest(BaseModel):
    assets: List[AssetModel]
    weights: Dict[str, float]
    benchmark_symbol: Optional[str] = None
    risk_free_rate: Optional[float] = 0.02
    lookback_period: Optional[int] = 252

class OptimizeResponse(BaseModel):
    success: bool
    method: Optional[str] = None
    objective: Optional[str] = None
    optimal_weights: Optional[Dict[str, float]] = None
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None

class AnalyzeResponse(BaseModel):
    success: bool
    weights: Optional[Dict[str, float]] = None
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    report: Optional[str] = None
    error: Optional[str] = None

class AssetsRequest(BaseModel):
    symbols: Optional[List[str]] = None
    period: Optional[str] = "5y"

class AssetData(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    period: str
    data_points: int
    prices: Optional[Dict[str, float]] = None
    returns: Optional[Dict[str, float]] = None
    metrics: Optional[Dict[str, float]] = None

# Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Simple Portfolio Optimization API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "optimize": "/optimize",
            "analyze": "/analyze",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "optimizer": "healthy",
            "performance": "healthy",
            "config": "healthy"
        }
    }

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_portfolio(request: OptimizeRequest):
    """Optimize portfolio for given assets."""
    try:
        symbols = [asset.symbol for asset in request.assets]

        if len(symbols) < 2:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "At least 2 symbols are required"
                }
            )

        # Convert assets to Asset objects for optimizer
        assets = []
        for asset_model in request.assets:
            asset = Asset(
                symbol=asset_model.symbol,
                name=asset_model.name or asset_model.symbol,
                sector=asset_model.sector or "Unknown"
            )

            period_str = f"{request.lookback_period}d" if request.lookback_period else config.portfolio.default_period

            try:
                history = data_service.fetch_historical_data(asset.symbol, period=period_str)
                if history.empty:
                    logger.warning(f"No historical data available for {asset.symbol}")
                    continue

                price_series = history["Adj Close"].dropna() if "Adj Close" in history else history.iloc[:, 0].dropna()
                returns_series = (
                    history["returns"].dropna()
                    if "returns" in history
                    else price_series.pct_change().dropna()
                )

                if returns_series.empty:
                    logger.warning(f"No return series available for {asset.symbol}")
                    continue

                asset.set_prices(price_series)
                asset.set_returns(returns_series)
            except Exception as e:
                logger.warning(f"Failed to fetch data for {asset.symbol}: {e}")
                continue

            assets.append(asset)

        if not assets:
            raise HTTPException(status_code=400, detail="No valid asset data available for optimization")

        # Create constraints from request
        constraints_dict = {}
        if request.constraints:
            constraints_dict = {
                'max_position_size': request.constraints.max_position_size or 0.1,
                'max_sector_concentration': request.constraints.max_sector_concentration or 0.3,
                'max_volatility': request.constraints.max_volatility or 0.3,
                'risk_free_rate': request.constraints.risk_free_rate or 0.02
            }

        constraints = PortfolioConstraints(**constraints_dict)

        # Handle market views for Black-Litterman
        market_views = None
        if request.market_views and request.method == "black_litterman":
            from portfolio.models.views import MarketViewCollection, MarketView
            views = []
            for view_model in request.market_views:
                view = MarketView(
                    asset_symbol=view_model.symbol,
                    expected_return=view_model.expected_return,
                    confidence=view_model.confidence,
                    view_type=view_model.view_type
                )
                views.append(view)
            market_views = MarketViewCollection(views)

        # Validate requested method explicitly (additional guard in case validation is bypassed)
        allowed_methods = {"mean_variance", "cvar", "black_litterman"}
        if request.method not in allowed_methods:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": f"Unsupported optimization method: {request.method}"
                }
            )

        # Run optimization
        start_time = datetime.now()
        result = optimizer.optimize(
            assets=assets,
            constraints=constraints,
            method=request.method,
            objective=request.objective,
            market_views=market_views
        )
        execution_time = (datetime.now() - start_time).total_seconds()
        execution_time = max(execution_time, 1e-6)

        # Create response
        response = OptimizeResponse(
            success=result.success,
            method=request.method,
            objective=request.objective,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )

        if result.success and result.optimal_weights:
            response.optimal_weights = result.optimal_weights
        else:
            response.error = ", ".join(result.error_messages) if result.error_messages else "Optimization failed"

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return OptimizeResponse(
            success=False,
            method=request.method,
            objective=request.objective,
            execution_time=0.0,
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_portfolio(request: AnalyzeRequest):
    """Analyze portfolio performance."""
    start_time = datetime.now()
    try:
        symbols = [asset.symbol for asset in request.assets]

        if len(symbols) != len(request.weights):
            raise HTTPException(status_code=422, detail="Symbols and weights must match")

        symbol_set = {asset.symbol.upper() for asset in request.assets}
        weight_set = {symbol.upper() for symbol in request.weights.keys()}
        if symbol_set != weight_set:
            raise HTTPException(status_code=422, detail="Weights must include every asset symbol exactly once")

        if abs(sum(request.weights.values()) - 1.0) > 0.01:
            raise HTTPException(status_code=422, detail="Weights must sum to 1.0")

        # Fetch data
        prices = optimizer.fetch_data(symbols, period=f"{request.lookback_period}d")
        if prices.empty:
            raise HTTPException(status_code=400, detail="No data found for symbols")

        # Calculate portfolio returns
        portfolio_returns = performance_calc.calculate_portfolio_returns(prices, request.weights)

        # Calculate metrics with custom risk-free rate
        metrics = performance_calc.calculate_metrics(portfolio_returns)
        if request.risk_free_rate:
            # Recalculate Sharpe ratio with custom risk-free rate
            if 'annual_volatility' in metrics and metrics['annual_volatility'] > 0:
                metrics['sharpe_ratio'] = (metrics['annual_return'] - request.risk_free_rate) / metrics['annual_volatility']

        # Generate report
        report = performance_calc.generate_report(metrics)

        execution_time = (datetime.now() - start_time).total_seconds()
        execution_time = max(execution_time, 1e-6)

        return AnalyzeResponse(
            success=True,
            weights=request.weights,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            report=report
        )

    except HTTPException:
        raise
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Analysis error: {e}")
        return AnalyzeResponse(
            success=False,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@app.get("/data/assets")
async def get_assets_data(symbols: str = None, period: str = "5y"):
    """Get asset data for analysis."""
    try:
        if not symbols:
            raise HTTPException(status_code=422, detail="symbols parameter is required")

        symbol_list = [symbol.strip().upper() for symbol in symbols.split(',') if symbol.strip()]
        if not symbol_list:
            raise HTTPException(status_code=422, detail="symbols parameter is required")

        assets_data: Dict[str, Dict[str, Any]] = {}

        benchmark_symbol = getattr(getattr(config, "performance", object()), "benchmark", "SPY")
        benchmark_data = data_service.fetch_historical_data(benchmark_symbol, period) if benchmark_symbol else pd.DataFrame()
        benchmark_returns = (
            benchmark_data["returns"].dropna()
            if isinstance(benchmark_data, pd.DataFrame) and "returns" in benchmark_data
            else None
        )

        for symbol in symbol_list:
            data = data_service.fetch_historical_data(symbol, period)
            if data.empty:
                logger.warning(f"No data returned for symbol {symbol} during period {period}")
                continue

            usable_prices = data.dropna(subset=["Open", "High", "Low", "Close", "Adj Close", "Volume"], how="any")
            returns_series = (
                usable_prices["returns"].dropna()
                if "returns" in usable_prices
                else usable_prices["Adj Close"].pct_change().dropna()
            )

            metrics = performance_calc.calculate_metrics(
                returns_series,
                benchmark_returns=benchmark_returns
            )

            price_records = [
                {
                    "date": index.to_pydatetime().strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "adj_close": float(row["Adj Close"]),
                    "volume": int(row["Volume"]),
                }
                for index, row in usable_prices.iterrows()
            ]

            assets_data[symbol] = {
                "symbol": symbol,
                "period": period,
                "data_points": int(len(returns_series)),
                "prices": price_records,
                "metrics": metrics,
            }

        if not assets_data:
            raise HTTPException(status_code=400, detail="No data found for symbols")

        summary = {
            "symbols": list(assets_data.keys()),
            "period": period,
            "total_assets": len(assets_data),
            "benchmark": benchmark_symbol,
        }

        if assets_data:
            summary["data_points"] = {symbol: data["data_points"] for symbol, data in assets_data.items()}

        return {"assets": assets_data, "summary": summary}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assets data error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/assets/{symbol}")
async def get_asset_data(symbol: str, period: str = "5y"):
    """Get basic data for a single asset."""
    try:
        prices = optimizer.fetch_data([symbol], period)
        returns = performance_calc.calculate_returns(prices[symbol])
        metrics = performance_calc.calculate_metrics(returns)

        return {
            "symbol": symbol,
            "period": period,
            "data_points": len(prices),
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Asset data error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
