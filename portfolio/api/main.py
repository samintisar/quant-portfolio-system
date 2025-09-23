"""
Simple API for portfolio optimization with basic endpoints only.

Provides essential functionality without overengineered abstractions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

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
    method: str = "mean_variance"
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
            raise HTTPException(status_code=400, detail="At least 2 symbols required")

        # Convert assets to Asset objects for optimizer
        assets = []
        for asset_model in request.assets:
            asset = Asset(
                symbol=asset_model.symbol,
                name=asset_model.name or asset_model.symbol,
                sector=asset_model.sector or "Unknown"
            )

            # Fetch data for the asset
            try:
                prices = optimizer.fetch_data([asset.symbol], period=f"{request.lookback_period}d")
                if not prices.empty:
                    returns = optimizer.calculate_returns(prices)
                    if not returns.empty:
                        asset.returns = returns[asset.symbol].dropna()
                        asset.prices = prices[asset.symbol].dropna()
            except Exception as e:
                logger.warning(f"Failed to fetch data for {asset.symbol}: {e}")
                continue

            assets.append(asset)

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
            raise HTTPException(status_code=400, detail="Symbols and weights must match")

        if abs(sum(request.weights.values()) - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")

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

        return AnalyzeResponse(
            success=True,
            weights=request.weights,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            report=report
        )

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

        symbol_list = symbols.split(',')

        # Fetch data for all symbols
        prices = optimizer.fetch_data(symbol_list, period=period)
        if prices.empty:
            raise HTTPException(status_code=400, detail="No data found for symbols")

        returns = optimizer.calculate_returns(prices)

        # Calculate metrics for each symbol
        assets_data = []
        for symbol in symbol_list:
            if symbol in returns.columns:
                symbol_returns = returns[symbol].dropna()
                symbol_prices = prices[symbol].dropna() if symbol in prices.columns else None

                metrics = performance_calc.calculate_metrics(symbol_returns)

                asset_data = AssetData(
                    symbol=symbol,
                    period=period,
                    data_points=len(symbol_returns),
                    metrics=metrics
                )

                assets_data.append(asset_data)

        return {"assets": assets_data}

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