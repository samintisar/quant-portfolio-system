"""
Simple API for portfolio optimization with basic endpoints only.

Provides essential functionality without overengineered abstractions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from portfolio_simple import SimplePortfolioOptimizer
from performance_simple import SimplePerformanceCalculator
from config_loader import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
optimizer = SimplePortfolioOptimizer()
performance_calc = SimplePerformanceCalculator()
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
class OptimizeRequest(BaseModel):
    symbols: List[str]
    method: str = "mean_variance"
    target_return: Optional[float] = None

class AnalyzeRequest(BaseModel):
    symbols: List[str]
    weights: Dict[str, float]

class OptimizeResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AnalyzeResponse(BaseModel):
    success: bool
    metrics: Optional[Dict[str, float]] = None
    report: Optional[str] = None
    error: Optional[str] = None

# Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Simple Portfolio Optimization API",
        "version": "1.0.0",
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
    """Optimize portfolio for given symbols."""
    try:
        if len(request.symbols) < 2:
            raise HTTPException(status_code=400, detail="At least 2 symbols required")

        if request.method not in ["mean_variance"]:
            raise HTTPException(status_code=400, detail="Only mean_variance method supported")

        result = optimizer.optimize_portfolio(
            symbols=request.symbols,
            method=request.method,
            target_return=request.target_return
        )

        return OptimizeResponse(success=True, result=result)

    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return OptimizeResponse(success=False, error=str(e))

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_portfolio(request: AnalyzeRequest):
    """Analyze portfolio performance."""
    try:
        if len(request.symbols) != len(request.weights):
            raise HTTPException(status_code=400, detail="Symbols and weights must match")

        if abs(sum(request.weights.values()) - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")

        # Fetch data
        prices = optimizer.fetch_data(request.symbols)
        if prices.empty:
            raise HTTPException(status_code=400, detail="No data found for symbols")

        # Calculate portfolio returns
        portfolio_returns = performance_calc.calculate_portfolio_returns(prices, request.weights)

        # Calculate metrics
        metrics = performance_calc.calculate_metrics(portfolio_returns)

        # Generate report
        report = performance_calc.generate_report(metrics)

        return AnalyzeResponse(
            success=True,
            metrics=metrics,
            report=report
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return AnalyzeResponse(success=False, error=str(e))

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