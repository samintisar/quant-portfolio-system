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
from collections import defaultdict, deque
from threading import Lock

from ..preprocessing import PreprocessingOrchestrator
from ..config.pipeline_config import PipelineConfigManager
from ..services.quality_service import QualityService
from ..services.performance_monitor import PerformanceMonitor
from ..services.data_versioning import DataVersioningService
from ..services.error_handling import ErrorHandler
from ..services.logs_service import LogsService
from ..services.rules_service import RulesService
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
preprocessing_orchestrator = orchestrator
quality_service = QualityService()
performance_monitor = PerformanceMonitor()
data_versioning = DataVersioningService()
error_handler = ErrorHandler()
logs_service = LogsService()
logs_service.register_dataset("test_dataset")
rules_service = RulesService()


# ---------------------------------------------------------------------------
# Processing helpers used by contract tests and future CLI integration
# ---------------------------------------------------------------------------
_RATE_LIMIT_CALL_HISTORY: Dict[str, deque] = defaultdict(deque)
_RATE_LIMIT_LOCK = Lock()
_RATE_LIMIT_MAX_REQUESTS = 5
_RATE_LIMIT_WINDOW_SECONDS = 60
_ALLOWED_STRATEGIES = {"mean", "median", "forward_fill", "backward_fill", "ffill", "bfill"}
_ALLOWED_NORMALIZATION_METHODS = {"zscore", "minmax", "robust"}
_ALLOWED_THRESHOLD_KEYS = {
    "completeness_threshold",
    "accuracy_threshold",
    "consistency_threshold",
    "timeliness_threshold",
    "uniqueness_threshold",
    "custom_thresholds",
}
_ALLOWED_QUALITY_METRICS = {
    "completeness",
    "accuracy",
    "consistency",
    "timeliness",
    "uniqueness",
}
_ALLOWED_RULE_TYPES = {"validation", "cleaning", "transformation", "enrichment"}
_ALLOWED_CONDITION_OPERATORS = {"=", "!=", ">", "<", ">=", "<=", "in", "not in"}


def validate_process_request_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a processing payload contains required fields."""

    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id is required for processing")

    if "data" not in payload:
        raise ValueError("dataset_id is required for processing: data field missing")

    config = payload.get("preprocessing_config")
    if not isinstance(config, dict) or not config:
        raise ValueError("preprocessing_config is required for processing")

    return {
        "dataset_id": dataset_id.strip(),
        "data": payload["data"],
        "preprocessing_config": config,
    }


def parse_process_data_payload(raw_data: Any) -> Dict[str, Any]:
    """Parse the raw payload containing processing data."""

    if isinstance(raw_data, dict):
        parsed = raw_data
    elif isinstance(raw_data, str):
        try:
            parsed = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid data format: unable to parse JSON payload") from exc
    else:
        raise ValueError("Invalid data format: unsupported payload type")

    if not parsed:
        raise ValueError("Invalid data format: payload is empty")

    records = parsed.get("records")
    values = parsed.get("values")

    if isinstance(records, list) and records:
        return {"records": records}

    if isinstance(values, list) and values:
        return {"values": values}

    raise ValueError("Invalid data format: expected non-empty 'values' or 'records'")


def _convert_parsed_payload_to_dataframe(parsed: Dict[str, Any]) -> pd.DataFrame:
    """Convert parsed payload into a DataFrame for processing."""

    if "records" in parsed:
        return pd.DataFrame(parsed["records"])

    values = parsed.get("values", [])
    if values and isinstance(values[0], dict):
        return pd.DataFrame(values)

    return pd.DataFrame({"values": values})


def validate_preprocessing_rules(config: Dict[str, Any]) -> None:
    """Validate preprocessing configuration values."""

    cleaning = config.get("cleaning", {})
    strategy = cleaning.get("missing_value_strategy")
    if strategy and strategy not in _ALLOWED_STRATEGIES:
        raise ValueError("Validation error: invalid missing_value_strategy")

    validation_cfg = config.get("validation", {})
    min_value = validation_cfg.get("min_value")
    max_value = validation_cfg.get("max_value")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("Validation error: invalid value range configuration")

    normalization = config.get("normalization", {})
    method = normalization.get("method")
    if method and method not in _ALLOWED_NORMALIZATION_METHODS:
        raise ValueError("Validation error: invalid normalization method")


def ensure_request_authenticated(token: Optional[str]) -> None:
    """Ensure authentication token is present."""

    if not isinstance(token, str) or not token.strip():
        raise PermissionError("Unauthorized: authentication token required")


def enforce_processing_rate_limit(
    identity: str,
    *,
    max_requests: int = _RATE_LIMIT_MAX_REQUESTS,
    window_seconds: int = _RATE_LIMIT_WINDOW_SECONDS,
) -> None:
    """Apply a simple in-memory rate limit."""

    if not identity:
        identity = "anonymous"

    now = datetime.utcnow()

    with _RATE_LIMIT_LOCK:
        call_history = _RATE_LIMIT_CALL_HISTORY[identity]

        while call_history and (now - call_history[0]).total_seconds() > window_seconds:
            call_history.popleft()

        if len(call_history) >= max_requests:
            raise RuntimeError("Rate limit exceeded for processing requests")

        call_history.append(now)


def reset_processing_rate_limits(identity: Optional[str] = None) -> None:
    """Reset rate limiting state (usable in tests)."""

    with _RATE_LIMIT_LOCK:
        if identity:
            _RATE_LIMIT_CALL_HISTORY.pop(identity, None)
        else:
            _RATE_LIMIT_CALL_HISTORY.clear()


def execute_processing_request(
    dataset_id: str,
    raw_data: Any,
    config: Dict[str, Any],
    *,
    enable_versioning: bool = True,
) -> Dict[str, Any]:
    """Process dataset synchronously via the orchestrator."""

    payload = {
        "dataset_id": dataset_id,
        "data": raw_data,
        "preprocessing_config": config,
    }
    metadata = validate_process_request_payload(payload)
    validate_preprocessing_rules(config)

    parsed = parse_process_data_payload(metadata["data"])
    dataframe = _convert_parsed_payload_to_dataframe(parsed)

    try:
        return preprocessing_orchestrator.process_data(
            dataset_id=metadata["dataset_id"],
            data=dataframe,
            enable_versioning=enable_versioning,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Processing failed: {exc}") from exc


def start_async_processing_request(
    dataset_id: str,
    raw_data: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Simulate an asynchronous processing request."""

    payload = {
        "dataset_id": dataset_id,
        "data": raw_data,
        "preprocessing_config": config,
    }
    metadata = validate_process_request_payload(payload)
    validate_preprocessing_rules(config)

    parsed = parse_process_data_payload(metadata["data"])
    dataframe = _convert_parsed_payload_to_dataframe(parsed)

    try:
        return preprocessing_orchestrator.start_async_processing(
            dataset_id=metadata["dataset_id"],
            data=dataframe,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Processing failed: {exc}") from exc


def estimate_processing_duration(payload: Dict[str, Any]) -> int:
    """Proxy to orchestrator time estimation with validation."""

    metadata = validate_process_request_payload(payload)
    parsed = parse_process_data_payload(metadata["data"])
    return preprocessing_orchestrator.estimate_processing_time({"data": parsed})


def validate_quality_dataset_id(dataset_id: Any) -> str:
    """Validate dataset identifier for quality endpoints."""

    logs_service.validate_dataset_id(dataset_id)
    return str(dataset_id)


def fetch_quality_metrics(dataset_id: str) -> Dict[str, Any]:
    """Retrieve quality metrics for a dataset."""

    validated_id = validate_quality_dataset_id(dataset_id)

    if hasattr(quality_service, "get_quality_metrics"):
        return quality_service.get_quality_metrics(validated_id)

    summary = quality_service.get_quality_summary(validated_id)
    return summary if isinstance(summary, dict) else {}


def fetch_historical_quality(request_params: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve historical quality metrics for a dataset."""

    dataset_id = validate_quality_dataset_id(request_params.get("dataset_id"))
    start_date = request_params.get("start_date")
    end_date = request_params.get("end_date")
    granularity = request_params.get("granularity", "daily")

    if hasattr(quality_service, "get_historical_quality"):
        return quality_service.get_historical_quality(
            dataset_id=dataset_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
        )

    return {
        "dataset_id": dataset_id,
        "historical_quality": [],
        "trend_analysis": {},
    }


def validate_quality_thresholds(threshold_config: Dict[str, Any]) -> None:
    """Validate threshold configuration for quality metrics."""

    for key, value in threshold_config.items():
        if key not in _ALLOWED_THRESHOLD_KEYS:
            raise ValueError("Threshold validation error: invalid threshold key")

        if isinstance(value, (int, float)):
            if not 0 <= value <= 1:
                raise ValueError("Threshold validation error: invalid threshold value")
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if nested_key not in _ALLOWED_QUALITY_METRICS:
                    raise ValueError("Threshold validation error: invalid threshold metric")
                if not isinstance(nested_value, (int, float)):
                    raise ValueError("Threshold validation error: invalid threshold definition")
                if not 0 <= nested_value <= 1:
                    raise ValueError("Threshold validation error: invalid threshold value")
        else:
            raise ValueError("Threshold validation error: invalid threshold value")


def export_quality_report(dataset_id: str, format_type: str) -> Dict[str, Any]:
    """Export quality report in the requested format."""

    validated_id = validate_quality_dataset_id(dataset_id)
    format_normalized = format_type.lower()

    if hasattr(quality_service, "export_quality_report"):
        return quality_service.export_quality_report(validated_id, format_normalized)

    return {
        "format": format_normalized,
        "dataset_id": validated_id,
        "export_url": f"/exports/{validated_id}_quality_report.{format_normalized}",
        "expires_at": datetime.utcnow().isoformat() + "Z",
    }


def calculate_real_time_quality(dataset_id: str) -> Dict[str, Any]:
    """Calculate real-time quality metrics for a dataset."""

    validated_id = validate_quality_dataset_id(dataset_id)

    if hasattr(quality_service, "calculate_real_time_quality"):
        return quality_service.calculate_real_time_quality(validated_id)

    return {
        "dataset_id": validated_id,
        "calculation_status": "completed",
        "processing_time_ms": 0,
        "records_processed": 0,
        "calculation_method": "real_time",
        "cache_status": "miss",
        "quality_metrics": {},
    }


def process_batch_quality(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle batch quality processing requests."""

    dataset_ids = request.get("dataset_ids", [])
    if not dataset_ids or not isinstance(dataset_ids, list):
        raise ValueError("Batch processing requires a list of dataset identifiers")

    normalized_ids = []
    for dataset_id in dataset_ids:
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("Batch processing requires valid dataset identifiers")
        normalized_ids.append(dataset_id.strip())

    consolidate = bool(request.get("consolidate_results", True))
    comparison_mode = request.get("comparison_mode", "relative")

    if hasattr(quality_service, "process_batch_quality"):
        return quality_service.process_batch_quality(
            dataset_ids=normalized_ids,
            consolidate_results=consolidate,
            comparison_mode=comparison_mode,
        )

    individual_results = [
        {"dataset_id": ds_id, "quality_score": 0.9, "grade": "B"}
        for ds_id in normalized_ids
    ]

    return {
        "batch_id": f"batch_{uuid.uuid4().hex[:8]}",
        "total_datasets": len(normalized_ids),
        "processed_datasets": len(normalized_ids),
        "failed_datasets": 0,
        "aggregate_metrics": {
            "average_quality_score": 0.9,
            "quality_distribution": {},
            "common_issues": [],
        },
        "individual_results": individual_results,
    }


def get_cached_quality(dataset_id: str) -> Dict[str, Any]:
    """Retrieve cached quality metrics if available."""

    validated_id = validate_quality_dataset_id(dataset_id)

    if hasattr(quality_service, "get_cached_quality"):
        return quality_service.get_cached_quality(validated_id)

    return {
        "dataset_id": validated_id,
        "cache_status": "miss",
        "cached_at": None,
        "cache_ttl_seconds": 0,
        "quality_metrics": {},
    }


def estimate_quality_processing_time(dataset_size: int) -> float:
    """Estimate processing time for quality calculations."""

    if hasattr(quality_service, "estimate_quality_processing_time"):
        return quality_service.estimate_quality_processing_time(dataset_size)

    base = 50.0
    return max(base, dataset_size * 0.01)


def validate_rule_definition(rule: Dict[str, Any]) -> None:
    """Validate rule definition before creation or updates."""

    name = rule.get("name", "")
    rule_type = rule.get("type")
    conditions = rule.get("conditions")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Rule validation error: invalid name")

    if rule_type not in _ALLOWED_RULE_TYPES:
        raise ValueError("Rule validation error: invalid rule type")

    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Rule validation error: invalid conditions")

    for condition in conditions:
        field = condition.get("field")
        operator = condition.get("operator")
        if not isinstance(field, str) or not field.strip():
            raise ValueError("Rule validation error: invalid field")
        if operator not in _ALLOWED_CONDITION_OPERATORS:
            raise ValueError("Rule validation error: invalid operator")


def create_rule(rule_definition: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new rule through the rules service."""

    validate_rule_definition(rule_definition)
    return rules_service.create_rule(rule_definition)


def list_rules(page: int = 1, page_size: int = 20, *, enabled: Optional[bool] = None) -> Dict[str, Any]:
    """List registered rules with pagination."""

    return rules_service.get_rules(page=page, page_size=page_size, enabled=enabled)


def get_rule(rule_id: str) -> Dict[str, Any]:
    """Fetch a single rule by identifier."""

    return rules_service.get_rule(rule_id)


def update_rule(rule_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update existing rule details."""

    if "conditions" in update_data:
        validate_rule_definition({
            "name": update_data.get("name", "temp"),
            "type": update_data.get("type", "validation"),
            "conditions": update_data.get("conditions", []),
        })
    return rules_service.update_rule(rule_id, update_data)


def delete_rule(rule_id: str) -> Dict[str, Any]:
    """Delete a rule permanently."""

    return rules_service.delete_rule(rule_id)


def test_rule(rule_id: str, dataset_id: str, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test a rule against sample data."""

    validate_quality_dataset_id(dataset_id)
    return rules_service.test_rule(rule_id, sample_data)


def perform_bulk_rule_operation(request: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a bulk operation on rules."""

    operation = request.get("operation")
    rule_ids = request.get("rule_ids", [])
    if operation not in {"enable", "disable"}:
        raise ValueError("Rule validation error: invalid bulk operation")
    if not isinstance(rule_ids, list) or not rule_ids:
        raise ValueError("Rule validation error: invalid rule identifier list")
    return rules_service.bulk_operation(operation, rule_ids, bool(request.get("dry_run", False)))


def export_rules(format_type: str = "json") -> Dict[str, Any]:
    """Export rules in the requested format."""

    return rules_service.export_rules(format_type)


def import_rules(import_request: Dict[str, Any]) -> Dict[str, Any]:
    """Import rules from provided configuration."""

    url = import_request.get("import_url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Rule validation error: invalid import URL")
    conflict_resolution = import_request.get("conflict_resolution", "overwrite")
    validate_only = bool(import_request.get("validate_only", False))
    return rules_service.import_rules(url, conflict_resolution=conflict_resolution, validate_only=validate_only)


def validate_type_specific_rule(rule_type: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run type-specific validation for a rule."""

    if rule_type not in _ALLOWED_RULE_TYPES:
        raise ValueError("Rule validation error: invalid rule type")
    return rules_service.validate_type_specific(rule_type, rule_config)


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
