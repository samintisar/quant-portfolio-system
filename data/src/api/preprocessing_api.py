"""
FastAPI Preprocessing API Server

RESTful API server for data preprocessing operations with
automatic documentation, validation, and async processing.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import uuid
from contextlib import asynccontextmanager

from ..preprocessing import PreprocessingOrchestrator
from ..config.pipeline_config import PipelineConfigManager
from ..services.quality_service import QualityService
from ..services.performance_monitor import PerformanceMonitor
from ..services.data_versioning import DataVersioningService
from ..services.error_handling import ErrorHandler
from ..models.quality_metrics import QualityReport
from ..lib.cleaning import DataCleaner
from ..lib.validation import DataValidator
from ..lib.normalization import DataNormalizer


# Pydantic models for API
class PreprocessingRequest(BaseModel):
    """Request model for preprocessing operations."""
    pipeline_id: str = Field(..., description="Pipeline configuration identifier")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data (if not uploading file)")
    output_format: str = Field("parquet", description="Output format (parquet, csv, json)")
    async_processing: bool = Field(False, description="Process asynchronously")

class PipelineCreateRequest(BaseModel):
    """Request model for creating pipelines."""
    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    description: str = Field(..., description="Pipeline description")
    asset_classes: List[str] = Field(..., description="List of asset classes")
    rules: List[Dict[str, Any]] = Field(..., description="Preprocessing rules")
    quality_thresholds: Dict[str, float] = Field(None, description="Quality thresholds")

class QualityAssessmentRequest(BaseModel):
    """Request model for quality assessment."""
    data: Dict[str, Any] = Field(..., description="Data to assess")
    dataset_id: str = Field(..., description="Dataset identifier")
    detailed: bool = Field(False, description="Generate detailed report")

class PreprocessingResponse(BaseModel):
    """Response model for preprocessing operations."""
    success: bool
    session_id: str
    dataset_id: str
    original_shape: Optional[tuple] = None
    final_shape: Optional[tuple] = None
    quality_score: Optional[float] = None
    execution_time: Optional[float] = None
    output_path: Optional[str] = None
    message: Optional[str] = None

class PipelineResponse(BaseModel):
    """Response model for pipeline operations."""
    pipeline_id: str
    description: str
    asset_classes: List[str]
    rules_count: int
    quality_thresholds: Dict[str, float]
    is_valid: bool
    created_at: str

class QualityReportResponse(BaseModel):
    """Response model for quality reports."""
    dataset_id: str
    overall_score: float
    metrics: List[Dict[str, Any]]
    generated_at: str

class StatusResponse(BaseModel):
    """Response model for status checks."""
    status: str
    session_id: str
    progress: float
    message: str
    timestamp: str


# Global state
processing_tasks = {}
config_manager = PipelineConfigManager()
orchestrator = PreprocessingOrchestrator(config_manager)
quality_service = QualityService()
performance_monitor = PerformanceMonitor()
data_versioning = DataVersioningService()
error_handler = ErrorHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logging.info("Starting Preprocessing API server")
    yield
    # Shutdown
    logging.info("Shutting down Preprocessing API server")


# Create FastAPI app
app = FastAPI(
    title="Data Preprocessing API",
    description="RESTful API for financial data preprocessing and quality assessment",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions
def validate_pipeline_exists(pipeline_id: str):
    """Validate that pipeline exists."""
    if not config_manager.get_config(pipeline_id):
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")


def create_processing_session(session_id: str, pipeline_id: str):
    """Create a new processing session."""
    processing_tasks[session_id] = {
        'session_id': session_id,
        'pipeline_id': pipeline_id,
        'status': 'pending',
        'progress': 0.0,
        'message': 'Session created',
        'start_time': datetime.now(),
        'end_time': None,
        'result': None
    }


def update_processing_status(session_id: str, status: str, progress: float, message: str):
    """Update processing session status."""
    if session_id in processing_tasks:
        processing_tasks[session_id].update({
            'status': status,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })


def complete_processing_session(session_id: str, result: Dict[str, Any]):
    """Complete processing session with result."""
    if session_id in processing_tasks:
        processing_tasks[session_id].update({
            'status': 'completed',
            'progress': 1.0,
            'message': 'Processing completed',
            'end_time': datetime.now(),
            'result': result
        })


def fail_processing_session(session_id: str, error: str):
    """Mark processing session as failed."""
    if session_id in processing_tasks:
        processing_tasks[session_id].update({
            'status': 'failed',
            'progress': 0.0,
            'message': f'Processing failed: {error}',
            'end_time': datetime.now(),
            'result': {'error': error}
        })


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Data Preprocessing API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "config_manager": "running",
            "orchestrator": "running",
            "quality_service": "running"
        }
    }


# Pipeline Management Endpoints
@app.post("/pipelines", response_model=PipelineResponse)
async def create_pipeline(request: PipelineCreateRequest):
    """Create a new preprocessing pipeline."""
    try:
        config = config_manager.create_default_config(
            pipeline_id=request.pipeline_id,
            description=request.description,
            asset_classes=request.asset_classes,
            rules=request.rules,
            quality_thresholds=request.quality_thresholds
        )

        validation = config_manager.validate_config(config)
        if not validation['is_valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Pipeline validation failed: {validation['errors']}"
            )

        config_path = config_manager.save_config(config)

        return PipelineResponse(
            pipeline_id=config.pipeline_id,
            description=config.description,
            asset_classes=config.asset_classes,
            rules_count=len(config.rules),
            quality_thresholds=config.quality_thresholds,
            is_valid=True,
            created_at=config.created_at.isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create pipeline: {str(e)}")


@app.get("/pipelines")
async def list_pipelines():
    """List all available pipelines."""
    try:
        pipelines = config_manager.list_pipelines()
        return {"pipelines": pipelines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {str(e)}")


@app.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get pipeline details."""
    try:
        validate_pipeline_exists(pipeline_id)
        summary = config_manager.get_config_summary(pipeline_id)
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline: {str(e)}")


@app.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline."""
    try:
        if not config_manager.delete_config(pipeline_id):
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
        return {"message": f"Pipeline '{pipeline_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete pipeline: {str(e)}")


@app.post("/pipelines/{pipeline_id}/validate")
async def validate_pipeline(pipeline_id: str):
    """Validate a pipeline configuration."""
    try:
        validate_pipeline_exists(pipeline_id)
        validation = config_manager.validate_config(pipeline_id)
        return validation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate pipeline: {str(e)}")


# Data Processing Endpoints
@app.post("/preprocessing/process", response_model=PreprocessingResponse)
async def process_data(request: PreprocessingRequest):
    """Process data using specified pipeline."""
    try:
        validate_pipeline_exists(request.pipeline_id)

        session_id = str(uuid.uuid4())

        if request.async_processing:
            # Create processing session and run in background
            create_processing_session(session_id, request.pipeline_id)
            # In a real implementation, you'd use Celery or similar
            # For now, we'll process synchronously but return session ID
            return PreprocessingResponse(
                success=True,
                session_id=session_id,
                dataset_id=request.pipeline_id,
                message="Processing started",
                execution_time=0.0
            )
        else:
            # Process synchronously
            if request.input_data:
                df = pd.DataFrame(request.input_data)
            else:
                raise HTTPException(status_code=400, detail="No input data provided")

            results = orchestrator.preprocess_data(df, request.pipeline_id)

            return PreprocessingResponse(
                success=results.get('success', False),
                session_id=results.get('session_id', session_id),
                dataset_id=results.get('dataset_id', request.pipeline_id),
                original_shape=results.get('original_shape'),
                final_shape=results.get('final_shape'),
                quality_score=results.get('quality_score'),
                execution_time=results.get('execution_time'),
                output_path=results.get('output_path'),
                message=results.get('message', 'Processing completed')
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/preprocessing/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    pipeline_id: str = None,
    output_format: str = "parquet",
    background_tasks: BackgroundTasks = None
):
    """Upload file and process it."""
    try:
        if not pipeline_id:
            raise HTTPException(status_code=400, detail="pipeline_id is required")

        validate_pipeline_exists(pipeline_id)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Load data based on file type
            if file.filename.endswith('.csv'):
                df = pd.read_csv(tmp_file_path)
            elif file.filename.endswith('.parquet') or file.filename.endswith('.pq'):
                df = pd.read_parquet(tmp_file_path)
            elif file.filename.endswith('.json'):
                df = pd.read_json(tmp_file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")

            # Process data
            results = orchestrator.preprocess_data(df, pipeline_id)

            return PreprocessingResponse(
                success=results.get('success', False),
                session_id=results.get('session_id'),
                dataset_id=results.get('dataset_id', pipeline_id),
                original_shape=results.get('original_shape'),
                final_shape=results.get('final_shape'),
                quality_score=results.get('quality_score'),
                execution_time=results.get('execution_time'),
                output_path=results.get('output_path'),
                message=results.get('message', 'Processing completed')
            )

        finally:
            # Clean up temporary file
            Path(tmp_file_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload and processing failed: {str(e)}")


@app.get("/preprocessing/status/{session_id}", response_model=StatusResponse)
async def get_processing_status(session_id: str):
    """Get processing status for a session."""
    if session_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Session not found")

    task = processing_tasks[session_id]

    return StatusResponse(
        status=task['status'],
        session_id=task['session_id'],
        progress=task['progress'],
        message=task['message'],
        timestamp=task.get('timestamp', datetime.now().isoformat())
    )


@app.get("/preprocessing/result/{session_id}")
async def get_processing_result(session_id: str):
    """Get processing result for a completed session."""
    if session_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Session not found")

    task = processing_tasks[session_id]

    if task['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Processing not completed")

    return task['result']


# Quality Assessment Endpoints
@app.post("/quality/assess", response_model=QualityReportResponse)
async def assess_quality(request: QualityAssessmentRequest):
    """Assess data quality."""
    try:
        df = pd.DataFrame(request.data)
        quality_report = quality_service.calculate_all_metrics(df, request.dataset_id)

        return QualityReportResponse(
            dataset_id=quality_report.dataset_id,
            overall_score=quality_report.overall_score,
            metrics=[
                {
                    'metric_id': m.metric_id,
                    'metric_type': m.metric_type,
                    'value': m.value,
                    'threshold': m.threshold,
                    'status': m.status,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in quality_report.metrics
            ],
            generated_at=quality_report.generated_at.isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")


@app.post("/quality/upload")
async def upload_and_assess(file: UploadFile = File(...), dataset_id: str = None):
    """Upload file and assess its quality."""
    try:
        if not dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id is required")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Load data based on file type
            if file.filename.endswith('.csv'):
                df = pd.read_csv(tmp_file_path)
            elif file.filename.endswith('.parquet') or file.filename.endswith('.pq'):
                df = pd.read_parquet(tmp_file_path)
            elif file.filename.endswith('.json'):
                df = pd.read_json(tmp_file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")

            # Assess quality
            quality_report = quality_service.calculate_all_metrics(df, dataset_id)

            return QualityReportResponse(
                dataset_id=quality_report.dataset_id,
                overall_score=quality_report.overall_score,
                metrics=[
                    {
                        'metric_id': m.metric_id,
                        'metric_type': m.metric_type,
                        'value': m.value,
                        'threshold': m.threshold,
                        'status': m.status,
                        'timestamp': m.timestamp.isoformat()
                    }
                    for m in quality_report.metrics
                ],
                generated_at=quality_report.generated_at.isoformat()
            )

        finally:
            # Clean up temporary file
            Path(tmp_file_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload and assessment failed: {str(e)}")


# Utility Endpoints
@app.get("/metrics")
async def get_system_metrics():
    """Get system metrics."""
    try:
        processing_summary = orchestrator.get_processing_summary()
        return {
            "system_metrics": {
                "total_sessions": processing_summary.get('total_sessions', 0),
                "successful_sessions": processing_summary.get('successful_sessions', 0),
                "success_rate": processing_summary.get('success_rate', 0.0),
                "total_execution_time": processing_summary.get('total_execution_time', 0.0),
                "active_tasks": len(processing_tasks)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.delete("/tasks/{session_id}")
async def cancel_processing_task(session_id: str):
    """Cancel a processing task."""
    if session_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = processing_tasks[session_id]
    if task['status'] in ['pending', 'running']:
        fail_processing_session(session_id, "Task cancelled by user")
        return {"message": f"Task {session_id} cancelled"}
    else:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")


# Performance Monitoring Endpoints
@app.get("/performance/metrics")
async def get_performance_metrics(operation: Optional[str] = None):
    """Get performance metrics for operations."""
    try:
        metrics = orchestrator.get_performance_metrics(operation)
        return {"performance_metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@app.get("/performance/system-health")
async def get_system_health():
    """Get current system health status."""
    try:
        health = orchestrator.get_system_health()
        return {"system_health": health}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@app.post("/performance/export")
async def export_performance_metrics(output_path: str, operation: Optional[str] = None):
    """Export performance metrics to file."""
    try:
        exported_path = orchestrator.export_performance_metrics(output_path, operation)
        return {"message": f"Metrics exported to {exported_path}", "path": exported_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


# Data Versioning Endpoints
@app.get("/versions/{dataset_id}")
async def get_dataset_versions(dataset_id: str):
    """Get all versions for a dataset."""
    try:
        versions = orchestrator.get_dataset_versions(dataset_id)
        return {"dataset_id": dataset_id, "versions": versions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset versions: {str(e)}")


@app.get("/versions/lineage/{version_id}")
async def get_version_lineage(version_id: str):
    """Get complete lineage for a version."""
    try:
        lineage = orchestrator.get_version_lineage(version_id)
        return {"version_id": version_id, "lineage": lineage}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get version lineage: {str(e)}")


@app.get("/versions/data/{version_id}")
async def load_version_data(version_id: str):
    """Load data for a specific version."""
    try:
        data = orchestrator.load_version_data(version_id)
        return {"version_id": version_id, "data": data.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load version data: {str(e)}")


@app.get("/datasets/{dataset_id}/summary")
async def get_dataset_summary(dataset_id: str):
    """Get summary information for a dataset."""
    try:
        summary = orchestrator.get_dataset_summary(dataset_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset summary: {str(e)}")


@app.post("/datasets/{dataset_id}/cleanup")
async def cleanup_old_versions(dataset_id: str, keep_versions: int = 10):
    """Clean up old versions for a dataset."""
    try:
        removed_count = orchestrator.cleanup_old_versions(dataset_id, keep_versions)
        return {
            "dataset_id": dataset_id,
            "removed_versions": removed_count,
            "message": f"Removed {removed_count} old versions"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup old versions: {str(e)}")


@app.get("/versions/{version_id}/reproduce")
async def reproduce_version(version_id: str):
    """Get reproduction instructions for a version."""
    try:
        instructions = orchestrator.reproduce_version(version_id)
        return instructions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get reproduction instructions: {str(e)}")


# Error Handling Endpoints
@app.get("/errors/statistics")
async def get_error_statistics():
    """Get statistics about handled errors."""
    try:
        stats = orchestrator.get_error_statistics()
        return {"error_statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get error statistics: {str(e)}")


@app.get("/errors/recovery-strategies")
async def get_recovery_strategies():
    """Get available recovery strategies."""
    try:
        strategies = orchestrator.get_recovery_strategies()
        return {"recovery_strategies": strategies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recovery strategies: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "preprocessing_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )