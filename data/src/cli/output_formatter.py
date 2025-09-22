"""
Output Formatter for CLI Operations

Simple data formatting capabilities for CSV, JSON, and table outputs.
"""

import json
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime


class OutputFormatter:
    """
    Simple Output Formatter for CLI operations
    """

    def format_csv(self, data: pd.DataFrame) -> str:
        """Format data as a CSV string without the index column."""

        csv_output = data.to_csv(index=False)
        return csv_output.strip()

    def format_json(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format data as JSON string with optional metadata"""
        result = {
            'data': data.to_dict('records'),
            'index': data.index.astype(str).tolist(),
            'columns': data.columns.tolist()
        }

        if metadata:
            result['metadata'] = metadata

        return json.dumps(result, indent=2, default=str)

    def format_table(self, data: pd.DataFrame) -> str:
        """Format data as table string"""
        return data.to_string(index=True)

    def format_summary(self, data: pd.DataFrame) -> str:
        """Format data as summary statistics"""
        summary = []
        summary.append(f"Data Shape: {data.shape}")
        summary.append(f"Columns: {len(data.columns)}")
        summary.append(f"Rows: {len(data)}")

        if not data.empty:
            summary.append(f"Date Range: {data.index.min()} to {data.index.max()}")
            summary.append(f"Missing Values: {data.isnull().sum().sum()}")
            summary.append(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024:.2f} KB")

        return "\n".join(summary)
