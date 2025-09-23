"""
Main FastAPI application for portfolio optimization system.

Provides REST API endpoints for portfolio optimization, analysis, and data retrieval.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from portfolio.api.endpoints import optimize_router, analyze_router, assets_router
from portfolio.api.models import HealthResponse
from portfolio.logging_config import get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Portfolio Optimization API",
    description="A REST API for portfolio optimization, performance analysis, and risk management",
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

# Include routers
app.include_router(optimize_router)
app.include_router(analyze_router)
app.include_router(assets_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Portfolio Optimization API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        services={"api": "healthy", "optimizer": "healthy", "data_service": "healthy"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "portfolio.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )