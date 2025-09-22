# API Documentation

This comprehensive API documentation covers all available endpoints for the quantitative trading system.

## Table of Contents

- [Data Preprocessing API](#data-preprocessing-api)
- [Feature Generation API](#feature-generation-api)
- [Quality Assessment API](#quality-assessment-api)
- [Rules Management API](#rules-management-api)
- [Authentication & Authorization](#authentication--authorization)
- [Response Formats](#response-formats)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Data Preprocessing API

The Data Preprocessing API provides comprehensive data cleaning, validation, and transformation capabilities for financial datasets.

### Base URL
```
http://localhost:8000
```

### Authentication
All API endpoints require Bearer token authentication:
```
Authorization: Bearer <your-api-token>
```

### Endpoints

#### Health Check
```http
GET /health
```

Check API service health and status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "config_manager": "running",
    "orchestrator": "running",
    "quality_service": "running"
  }
}
```

#### Pipeline Management

##### Create Pipeline
```http
POST /pipelines
```

Create a new data preprocessing pipeline.

**Request Body:**
```json
{
  "pipeline_id": "equity_daily_pipeline",
  "description": "Daily equity data preprocessing",
  "asset_classes": ["equity", "etf"],
  "rules": [
    {
      "type": "validation",
      "conditions": [{"field": "close", "operator": "greater_than", "value": 0}],
      "actions": [{"type": "flag", "severity": "warning"}]
    }
  ],
  "quality_thresholds": {
    "completeness": 0.95,
    "accuracy": 0.90
  }
}
```

**Response:**
```json
{
  "pipeline_id": "equity_daily_pipeline",
  "description": "Daily equity data preprocessing",
  "asset_classes": ["equity", "etf"],
  "rules_count": 1,
  "quality_thresholds": {
    "completeness": 0.95,
    "accuracy": 0.90
  },
  "is_valid": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

##### List Pipelines
```http
GET /pipelines
```

Get all available pipelines.

**Response:**
```json
{
  "pipelines": [
    {
      "pipeline_id": "equity_daily_pipeline",
      "description": "Daily equity data preprocessing",
      "asset_classes": ["equity", "etf"],
      "rules_count": 1,
      "status": "active"
    }
  ]
}
```

##### Get Pipeline
```http
GET /pipelines/{pipeline_id}
```

Get detailed pipeline configuration.

##### Delete Pipeline
```http
DELETE /pipelines/{pipeline_id}
```

Delete a pipeline.

##### Validate Pipeline
```http
POST /pipelines/{pipeline_id}/validate
```

Validate pipeline configuration.

#### Data Processing

##### Process Data
```http
POST /preprocessing/process
```

Process data using a specified pipeline.

**Request Body:**
```json
{
  "pipeline_id": "equity_daily_pipeline",
  "input_data": {
    "dates": ["2024-01-01", "2024-01-02"],
    "close": [100.5, 101.2],
    "volume": [1000000, 1200000]
  },
  "output_format": "parquet",
  "async_processing": false
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "12345678-1234-1234-1234-123456789012",
  "dataset_id": "equity_daily_pipeline",
  "original_shape": [2, 3],
  "final_shape": [2, 3],
  "quality_score": 0.95,
  "execution_time": 0.45,
  "output_path": "/path/to/output.parquet",
  "message": "Processing completed"
}
```

##### Upload and Process
```http
POST /preprocessing/upload
```

Upload file and process it.

**Parameters:**
- `file`: Data file (CSV, Parquet, JSON)
- `pipeline_id`: Pipeline identifier
- `output_format`: Output format (default: parquet)

##### Get Processing Status
```http
GET /preprocessing/status/{session_id}
```

Check processing status for async operations.

**Response:**
```json
{
  "status": "completed",
  "session_id": "12345678-1234-1234-1234-123456789012",
  "progress": 1.0,
  "message": "Processing completed",
  "timestamp": "2024-01-15T10:31:00Z"
}
```

##### Get Processing Result
```http
GET /preprocessing/result/{session_id}
```

Get processing results for completed session.

#### Quality Assessment

##### Assess Quality
```http
POST /quality/assess
```

Assess data quality metrics.

**Request Body:**
```json
{
  "dataset_id": "equity_data_2024",
  "data": {
    "dates": ["2024-01-01", "2024-01-02"],
    "close": [100.5, 101.2],
    "volume": [1000000, 1200000]
  },
  "detailed": true
}
```

**Response:**
```json
{
  "dataset_id": "equity_data_2024",
  "overall_score": 0.92,
  "metrics": [
    {
      "metric_id": "completeness_001",
      "metric_type": "completeness",
      "value": 0.98,
      "threshold": 0.9,
      "status": "pass",
      "timestamp": "2024-01-15T10:31:00Z"
    }
  ],
  "generated_at": "2024-01-15T10:31:00Z"
}
```

##### Upload and Assess
```http
POST /quality/upload
```

Upload file and assess quality.

#### Performance Monitoring

##### Get System Metrics
```http
GET /metrics
```

Get system performance metrics.

**Response:**
```json
{
  "system_metrics": {
    "total_sessions": 1250,
    "successful_sessions": 1187,
    "success_rate": 0.95,
    "total_execution_time": 456.78,
    "active_tasks": 5
  },
  "timestamp": "2024-01-15T10:31:00Z"
}
```

##### Get Performance Metrics
```http
GET /performance/metrics
```

Get detailed performance metrics.

**Parameters:**
- `operation`: Optional operation filter

##### Get System Health
```http
GET /performance/system-health
```

Get current system health status.

##### Export Performance Metrics
```http
POST /performance/export
```

Export performance metrics to file.

#### Data Versioning

##### Get Dataset Versions
```http
GET /versions/{dataset_id}
```

Get all versions for a dataset.

**Response:**
```json
{
  "dataset_id": "equity_data_2024",
  "versions": [
    {
      "version_id": "v1.0.0",
      "created_at": "2024-01-15T10:30:00Z",
      "description": "Initial version",
      "data_points": 252
    }
  ]
}
```

##### Get Version Lineage
```http
GET /versions/lineage/{version_id}
```

Get complete lineage for a version.

##### Load Version Data
```http
GET /versions/data/{version_id}
```

Load data for a specific version.

##### Get Dataset Summary
```http
GET /datasets/{dataset_id}/summary
```

Get summary information for a dataset.

##### Cleanup Old Versions
```http
POST /datasets/{dataset_id}/cleanup
```

Clean up old versions for a dataset.

**Parameters:**
- `keep_versions`: Number of versions to keep (default: 10)

##### Reproduce Version
```http
GET /versions/{version_id}/reproduce
```

Get reproduction instructions for a version.

#### Error Handling

##### Get Error Statistics
```http
GET /errors/statistics
```

Get statistics about handled errors.

##### Get Recovery Strategies
```http
GET /errors/recovery-strategies
```

Get available recovery strategies.

## Feature Generation API

The Feature Generation API provides endpoints for creating, managing, and validating financial features.

### Base URL
```
http://localhost:8001
```

### Endpoints

#### Generate Features
```http
POST /features/generate
```

Generate financial features from input data.

**Request Body:**
```json
{
  "data": {
    "dates": ["2024-01-01", "2024-01-02"],
    "close": [100.5, 101.2],
    "volume": [1000000, 1200000]
  },
  "features": ["returns", "volatility", "momentum"],
  "config": {
    "return_periods": [1, 5, 21],
    "volatility_windows": [5, 21],
    "momentum_periods": [5, 14]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "feature_set": {
      "returns_1": [0.007, 0.012],
      "returns_5": [0.015, 0.023],
      "volatility_5": [0.012, 0.015],
      "momentum_5": [0.85, 0.92]
    },
    "features": ["returns_1", "returns_5", "volatility_5", "momentum_5"],
    "quality_score": 0.95
  },
  "metadata": {
    "features_generated": 4,
    "data_points": 2,
    "config_used": {
      "return_periods": [1, 5, 21],
      "volatility_windows": [5, 21],
      "momentum_periods": [5, 14]
    }
  }
}
```

#### Validate Features
```http
POST /features/validate
```

Validate generated features.

**Request Body:**
```json
{
  "features": ["returns_1", "volatility_5", "momentum_5"]
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "valid_features": ["returns_1", "volatility_5"],
    "invalid_features": ["momentum_5"],
    "validation_issues": [
      {
        "feature": "momentum_5",
        "issue": "Insufficient data points",
        "severity": "error"
      }
    ]
  },
  "metadata": {
    "features_validated": 3,
    "validation_timestamp": "2024-01-15T10:31:00Z"
  }
}
```

#### Get Feature Metadata
```http
GET /features/metadata
```

Get metadata for features.

**Parameters:**
- `feature_name`: Optional specific feature name

**Response:**
```json
{
  "status": "success",
  "data": {
    "feature_name": "returns_1",
    "description": "1-day simple returns",
    "calculation_method": "simple",
    "unit": "decimal",
    "data_requirements": ["close"],
    "window_size": 1,
    "created_at": "2024-01-15T10:30:00Z"
  },
  "metadata": {
    "feature_name": "returns_1",
    "metadata_timestamp": "2024-01-15T10:31:00Z"
  }
}
```

#### Get Available Features
```http
GET /features/available
```

Get list of available features.

**Response:**
```json
{
  "status": "success",
  "data": {
    "available_features": [
      "returns_1", "returns_5", "returns_21",
      "volatility_5", "volatility_21",
      "momentum_5", "momentum_14"
    ]
  },
  "metadata": {
    "total_features": 7,
    "timestamp": "2024-01-15T10:31:00Z"
  }
}
```

#### Health Check
```http
GET /features/health
```

Check feature generation service health.

## Quality Assessment API

The Quality Assessment API provides comprehensive data quality metrics and monitoring capabilities.

### Base URL
```
http://localhost:8002
```

### Endpoints

#### Quality Assessment
```http
POST /quality/assess
```

Perform comprehensive quality assessment.

**Request Body:**
```json
{
  "dataset_id": "equity_data_2024",
  "data": {
    "dates": ["2024-01-01", "2024-01-02"],
    "close": [100.5, 101.2],
    "volume": [1000000, 1200000]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "dataset_id": "equity_data_2024",
    "overall_score": 0.92,
    "metric_scores": [
      {
        "metric": "completeness",
        "score": 0.98,
        "weight": 0.25,
        "threshold": 0.9,
        "details": {
          "missing_values": 0,
          "total_cells": 6,
          "missing_by_column": {},
          "completeness_ratio": 0.98
        }
      },
      {
        "metric": "accuracy",
        "score": 0.95,
        "weight": 0.2,
        "threshold": 0.85,
        "details": {
          "validation_issues": 0,
          "max_possible_issues": 4,
          "accuracy_ratio": 0.95
        }
      }
    ],
    "data_points": 2,
    "issues_found": 0,
    "recommendations": []
  },
  "metadata": {
    "assessment_timestamp": "2024-01-15T10:31:00Z",
    "metrics_calculated": 6
  }
}
```

#### Get Historical Quality
```http
GET /quality/historical/{dataset_id}
```

Get historical quality data for a dataset.

**Response:**
```json
{
  "status": "success",
  "data": {
    "dataset_id": "equity_data_2024",
    "historical_reports": [
      {
        "id": 1,
        "dataset_id": "equity_data_2024",
        "overall_score": 0.92,
        "metric_scores": [...],
        "timestamp": "2024-01-15T10:31:00Z",
        "data_points": 2,
        "issues_found": 0,
        "recommendations": []
      }
    ],
    "total_reports": 1
  },
  "metadata": {
    "query_timestamp": "2024-01-15T10:31:00Z"
  }
}
```

#### Get Quality Thresholds
```http
GET /quality/thresholds
```

Get quality threshold configuration.

**Response:**
```json
{
  "status": "success",
  "data": {
    "thresholds": {
      "completeness": 0.9,
      "accuracy": 0.85,
      "consistency": 0.9,
      "timeliness": 0.8,
      "validity": 0.85,
      "uniqueness": 0.95,
      "overall": 0.85
    },
    "description": "Quality score thresholds for different metrics"
  },
  "metadata": {
    "threshold_version": "1.0",
    "timestamp": "2024-01-15T10:31:00Z"
  }
}
```

## Rules Management API

The Rules Management API provides endpoints for creating, managing, and executing data preprocessing rules.

### Base URL
```
http://localhost:8003
```

### Endpoints

#### Create Rule
```http
POST /rules
```

Create a new preprocessing rule.

**Request Body:**
```json
{
  "name": "Price Validation Rule",
  "description": "Validate that prices are positive",
  "rule_type": "validation",
  "parameters": {
    "severity": "error"
  },
  "conditions": [
    {
      "field": "close",
      "operator": "less_than_or_equal",
      "value": 0
    }
  ],
  "actions": [
    {
      "type": "flag",
      "severity": "error"
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "rule_id": "12345678-1234-1234-1234-123456789012",
    "name": "Price Validation Rule",
    "rule_type": "validation",
    "status": "active",
    "version": "1.0.0",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "metadata": {
    "operation": "create_rule",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Get Rules
```http
GET /rules
```

Get list of rules with optional filtering.

**Parameters:**
- `rule_type`: Filter by rule type
- `status`: Filter by status
- `limit`: Maximum number of rules to return

**Response:**
```json
{
  "status": "success",
  "data": {
    "rules": [
      {
        "id": "12345678-1234-1234-1234-123456789012",
        "name": "Price Validation Rule",
        "description": "Validate that prices are positive",
        "rule_type": "validation",
        "status": "active",
        "version": "1.0.0",
        "created_at": "2024-01-15T10:30:00Z"
      }
    ],
    "total_count": 1,
    "filters_applied": {
      "rule_type": null,
      "status": null,
      "limit": 100
    }
  },
  "metadata": {
    "query_timestamp": "2024-01-15T10:31:00Z"
  }
}
```

#### Get Rule
```http
GET /rules/{rule_id}
```

Get a specific rule by ID.

#### Update Rule
```http
PUT /rules/{rule_id}
```

Update an existing rule.

#### Delete Rule
```http
DELETE /rules/{rule_id}
```

Delete a rule.

#### Test Rule
```http
POST /rules/{rule_id}/test
```

Test a rule against sample data.

**Request Body:**
```json
{
  "test_data": {
    "close": [100.5, -50.0, 101.2],
    "volume": [1000000, 1200000, 900000]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "rule_id": "12345678-1234-1234-1234-123456789012",
    "execution_id": "87654321-4321-4321-4321-210987654321",
    "status": "completed",
    "data_points_processed": 3,
    "violations_found": 1,
    "actions_taken": [
      {
        "type": "flag",
        "applied": true,
        "items_affected": 1
      }
    ],
    "execution_time": 0.023,
    "timestamp": "2024-01-15T10:31:00Z",
    "rule_type": "validation",
    "simulation": true
  },
  "metadata": {
    "operation": "test_rule",
    "timestamp": "2024-01-15T10:31:00Z"
  }
}
```

#### Bulk Operations
```http
POST /rules/bulk
```

Perform bulk operations on rules.

**Request Body:**
```json
{
  "operation": "activate",
  "rule_ids": [
    "12345678-1234-1234-1234-123456789012",
    "87654321-4321-4321-4321-210987654321"
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "operation": "activate",
    "total_rules": 2,
    "successful": 2,
    "failed": 0,
    "results": [
      {
        "rule_id": "12345678-1234-1234-1234-123456789012",
        "status": "success",
        "message": ""
      },
      {
        "rule_id": "87654321-4321-4321-4321-210987654321",
        "status": "success",
        "message": ""
      }
    ]
  },
  "metadata": {
    "operation": "bulk_operations",
    "timestamp": "2024-01-15T10:31:00Z"
  }
}
```

## Authentication & Authorization

### Bearer Token Authentication
All API endpoints require a valid Bearer token:

```http
Authorization: Bearer <your-api-token>
```

### Token Management
Tokens can be obtained through the authentication service and have the following properties:
- **Expiration**: 24 hours
- **Scope**: Feature-specific permissions
- **Rate Limits**: Based on subscription tier

## Response Formats

### Standard Response Structure
All API responses follow a consistent structure:

```json
{
  "status": "success|error",
  "data": {
    // Response data varies by endpoint
  },
  "metadata": {
    // Metadata about the request/response
    "timestamp": "2024-01-15T10:31:00Z",
    "request_id": "12345678-1234-1234-1234-123456789012"
  }
}
```

### Error Response Structure
Error responses include additional detail:

```json
{
  "status": "error",
  "message": "Descriptive error message",
  "code": 400,
  "details": {
    // Additional error context
  },
  "metadata": {
    "timestamp": "2024-01-15T10:31:00Z",
    "request_id": "12345678-1234-1234-1234-123456789012"
  }
}
```

## Error Handling

### HTTP Status Codes
- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Error Codes
- `1000`: General error
- `1001`: Validation error
- `1002`: Authentication error
- `1003`: Authorization error
- `1004`: Rate limit exceeded
- `2000`: Data processing error
- `2001`: Quality assessment error
- `2002`: Feature generation error
- `3000`: Rule execution error

## Rate Limiting

### Default Limits
- **Free Tier**: 100 requests per minute
- **Professional Tier**: 1000 requests per minute
- **Enterprise Tier**: Custom limits

### Headers
Rate limiting information is returned in response headers:
- `X-RateLimit-Limit`: Requests allowed per minute
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time until reset (seconds)

### Retry-After
When rate limited, the response includes a `Retry-After` header indicating when to retry.

## Examples

### Complete Data Processing Workflow

#### Step 1: Create Pipeline
```bash
curl -X POST http://localhost:8000/pipelines \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_id": "equity_daily_pipeline",
    "description": "Daily equity data preprocessing",
    "asset_classes": ["equity"],
    "rules": [
      {
        "type": "validation",
        "conditions": [{"field": "close", "operator": "greater_than", "value": 0}],
        "actions": [{"type": "flag", "severity": "error"}]
      }
    ],
    "quality_thresholds": {
      "completeness": 0.95,
      "accuracy": 0.90
    }
  }'
```

#### Step 2: Process Data
```bash
curl -X POST http://localhost:8000/preprocessing/process \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_id": "equity_daily_pipeline",
    "input_data": {
      "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "close": [100.5, 101.2, 99.8],
      "volume": [1000000, 1200000, 900000]
    },
    "output_format": "parquet"
  }'
```

#### Step 3: Generate Features
```bash
curl -X POST http://localhost:8001/features/generate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "close": [100.5, 101.2, 99.8],
      "volume": [1000000, 1200000, 900000]
    },
    "features": ["returns", "volatility", "momentum"]
  }'
```

#### Step 4: Assess Quality
```bash
curl -X POST http://localhost:8002/quality/assess \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "equity_data_2024",
    "data": {
      "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "close": [100.5, 101.2, 99.8],
      "volume": [1000000, 1200000, 900000]
    }
  }'
```

### Advanced Rule Management

#### Create Custom Validation Rule
```bash
curl -X POST http://localhost:8003/rules \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "OHLC Validation Rule",
    "description": "Validate OHLC relationships",
    "rule_type": "validation",
    "parameters": {
      "severity": "error"
    },
    "conditions": [
      {
        "field": "high",
        "operator": "less_than",
        "value": {"reference": "low"}
      }
    ],
    "actions": [
      {
        "type": "flag",
        "severity": "error"
      }
    ]
  }'
```

#### Test Rule with Sample Data
```bash
curl -X POST http://localhost:8003/rules/12345678-1234-1234-1234-123456789012/test \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "test_data": {
      "open": [100.0, 101.0, 99.5],
      "high": [101.0, 102.0, 100.0],
      "low": [99.5, 100.5, 99.0],
      "close": [100.5, 101.5, 99.8]
    }
  }'
```

### Batch Processing

#### Upload and Process File
```bash
curl -X POST http://localhost:8000/preprocessing/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@equity_data.csv" \
  -F "pipeline_id=equity_daily_pipeline" \
  -F "output_format=parquet"
```

#### Bulk Rule Operations
```bash
curl -X POST http://localhost:8003/rules/bulk \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "activate",
    "rule_ids": [
      "12345678-1234-1234-1234-123456789012",
      "87654321-4321-4321-4321-210987654321"
    ]
  }'
```

## SDKs and Libraries

### Python SDK
```python
from quant_portfolio_sdk import DataProcessor, FeatureGenerator, QualityAssessor

# Initialize clients
processor = DataProcessor(api_key="your-api-key")
feature_gen = FeatureGenerator(api_key="your-api-key")
quality_assessor = QualityAssessor(api_key="your-api-key")

# Process data
result = processor.process_data(
    pipeline_id="equity_daily_pipeline",
    data={
        "dates": ["2024-01-01", "2024-01-02"],
        "close": [100.5, 101.2],
        "volume": [1000000, 1200000]
    }
)

# Generate features
features = feature_gen.generate_features(
    data=result.data,
    features=["returns", "volatility", "momentum"]
)

# Assess quality
quality_report = quality_assessor.assess_quality(
    dataset_id="equity_data_2024",
    data=features.data
)
```

### JavaScript SDK
```javascript
const { DataProcessor, FeatureGenerator, QualityAssessor } = require('quant-portfolio-sdk');

// Initialize clients
const processor = new DataProcessor({ apiKey: 'your-api-key' });
const featureGen = new FeatureGenerator({ apiKey: 'your-api-key' });
const qualityAssessor = new QualityAssessor({ apiKey: 'your-api-key' });

// Process data
const result = await processor.processData({
    pipelineId: 'equity_daily_pipeline',
    data: {
        dates: ['2024-01-01', '2024-01-02'],
        close: [100.5, 101.2],
        volume: [1000000, 1200000]
    }
});

// Generate features
const features = await featureGen.generateFeatures({
    data: result.data,
    features: ['returns', 'volatility', 'momentum']
});

// Assess quality
const qualityReport = await qualityAssessor.assessQuality({
    datasetId: 'equity_data_2024',
    data: features.data
});
```

## Best Practices

### Security
- Always use HTTPS in production
- Store API tokens securely
- Implement proper error handling
- Validate input data before sending

### Performance
- Use batch operations for large datasets
- Monitor rate limits and implement backoff
- Cache frequently accessed data
- Use appropriate data formats (Parquet recommended)

### Error Handling
- Implement retry logic with exponential backoff
- Handle rate limiting gracefully
- Log errors for debugging
- Provide meaningful error messages to users

### Data Validation
- Validate input data before processing
- Use appropriate data types
- Handle missing values appropriately
- Validate data ranges and formats

## Support

For API support and questions:
- **Documentation**: [https://docs.quant-portfolio-system.com](https://docs.quant-portfolio-system.com)
- **API Status**: [https://status.quant-portfolio-system.com](https://status.quant-portfolio-system.com)
- **Support Email**: [api-support@quant-portfolio-system.com](mailto:api-support@quant-portfolio-system.com)
- **Community Forum**: [https://community.quant-portfolio-system.com](https://community.quant-portfolio-system.com)

## Changelog

### Version 1.0.0 (2024-01-15)
- Initial API release
- Data preprocessing endpoints
- Feature generation endpoints
- Quality assessment endpoints
- Rules management endpoints
- Comprehensive error handling
- Rate limiting implementation