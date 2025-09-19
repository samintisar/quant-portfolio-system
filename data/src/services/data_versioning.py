"""
Data Versioning and Reproducibility Service

Manages versioning of datasets, preprocessing operations, and ensures
reproducibility of all data transformations with complete lineage tracking.
"""

import hashlib
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import shutil
import pandas as pd
import numpy as np
from contextlib import contextmanager

from ..config.pipeline_config import PreprocessingConfig
from ..models.processing_log import ProcessingLog


@dataclass
class DataVersion:
    """Data version information."""
    version_id: str
    dataset_id: str
    parent_version_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    size_bytes: int = 0
    row_count: int = 0
    column_count: int = 0
    format: str = "parquet"
    storage_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_info: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataVersion':
        """Create from dictionary representation."""
        data = data.copy()
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ProcessingStep:
    """Individual processing step in the lineage."""
    step_id: str
    operation: str
    input_version_id: str
    output_version_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    environment_info: Dict[str, Any] = field(default_factory=dict)
    code_version: Optional[str] = None
    config_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingStep':
        """Create from dictionary representation."""
        data = data.copy()
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class DatasetLineage:
    """Complete lineage tracking for a dataset."""
    dataset_id: str
    root_version_id: str
    versions: Dict[str, DataVersion] = field(default_factory=dict)
    processing_steps: List[ProcessingStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'dataset_id': self.dataset_id,
            'root_version_id': self.root_version_id,
            'versions': {vid: v.to_dict() for vid, v in self.versions.items()},
            'processing_steps': [step.to_dict() for step in self.processing_steps],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetLineage':
        """Create from dictionary representation."""
        lineage = cls(
            dataset_id=data['dataset_id'],
            root_version_id=data['root_version_id'],
            metadata=data.get('metadata', {})
        )

        # Load versions
        for vid, v_data in data['versions'].items():
            lineage.versions[vid] = DataVersion.from_dict(v_data)

        # Load processing steps
        for step_data in data['processing_steps']:
            lineage.processing_steps.append(ProcessingStep.from_dict(step_data))

        # Load timestamps
        if 'created_at' in data:
            lineage.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            lineage.updated_at = datetime.fromisoformat(data['updated_at'])

        return lineage


class DataVersioningService:
    """Service for managing data versioning and reproducibility."""

    def __init__(self, storage_root: Optional[str] = None):
        """Initialize the data versioning service.

        Args:
            storage_root: Root directory for versioned data storage
        """
        self.logger = logging.getLogger(__name__)
        self.storage_root = Path(storage_root) if storage_root else Path(__file__).parent / "versions"
        self.storage_root.mkdir(parents=True, exist_ok=True)

        # Lineage tracking
        self.lineages: Dict[str, DatasetLineage] = {}
        self._load_lineages()

        # Version management
        self.version_index: Dict[str, str] = {}  # version_id -> dataset_id
        self.tag_index: Dict[str, List[str]] = {}  # tag -> [version_ids]

        # Environment tracking
        self._capture_environment_info()

    def _capture_environment_info(self) -> Dict[str, Any]:
        """Capture current environment information."""
        try:
            import platform
            import sys
            import pandas as pd
            import numpy as np

            return {
                'python_version': sys.version,
                'platform': platform.platform(),
                'pandas_version': pd.__version__,
                'numpy_version': np.__version__,
                'timestamp': datetime.now().isoformat(),
                'working_directory': str(Path.cwd())
            }
        except Exception as e:
            self.logger.warning(f"Failed to capture environment info: {e}")
            return {}

    def _load_lineages(self):
        """Load existing lineages from storage."""
        lineage_file = self.storage_root / "lineages.json"
        if lineage_file.exists():
            try:
                with open(lineage_file, 'r') as f:
                    data = json.load(f)

                for dataset_id, lineage_data in data.items():
                    self.lineages[dataset_id] = DatasetLineage.from_dict(lineage_data)

                # Rebuild indexes
                self._rebuild_indexes()

                self.logger.info(f"Loaded {len(self.lineages)} existing lineages")
            except Exception as e:
                self.logger.error(f"Failed to load lineages: {e}")

    def _rebuild_indexes(self):
        """Rebuild internal indexes from loaded lineages."""
        self.version_index.clear()
        self.tag_index.clear()

        for lineage in self.lineages.values():
            for version_id, version in lineage.versions.items():
                self.version_index[version_id] = lineage.dataset_id

                for tag in version.tags:
                    if tag not in self.tag_index:
                        self.tag_index[tag] = []
                    self.tag_index[tag].append(version_id)

    def _save_lineages(self):
        """Save lineages to storage."""
        lineage_file = self.storage_root / "lineages.json"
        try:
            data = {dataset_id: lineage.to_dict() for dataset_id, lineage in self.lineages.items()}
            with open(lineage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save lineages: {e}")

    def _calculate_checksum(self, data: Union[pd.DataFrame, Path]) -> str:
        """Calculate checksum for data.

        Args:
            data: DataFrame or file path

        Returns:
            SHA256 checksum
        """
        if isinstance(data, pd.DataFrame):
            # Use DataFrame string representation for checksum
            df_str = data.to_string(index=False)
            return hashlib.sha256(df_str.encode()).hexdigest()
        elif isinstance(data, Path):
            # Use file contents for checksum
            with open(data, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        else:
            raise ValueError(f"Unsupported data type for checksum: {type(data)}")

    def create_version(self, dataset_id: str, data: Union[pd.DataFrame, Path],
                      parent_version_id: Optional[str] = None,
                      description: str = "", tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> DataVersion:
        """Create a new version of a dataset.

        Args:
            dataset_id: Dataset identifier
            data: Data to version (DataFrame or file path)
            parent_version_id: Parent version ID
            description: Version description
            tags: Version tags
            metadata: Additional metadata

        Returns:
            Created DataVersion
        """
        # Generate version ID
        version_id = str(uuid.uuid4())

        # Create dataset lineage if it doesn't exist
        if dataset_id not in self.lineages:
            self.lineages[dataset_id] = DatasetLineage(
                dataset_id=dataset_id,
                root_version_id=version_id
            )

        lineage = self.lineages[dataset_id]

        # Calculate checksum and get data info
        checksum = self._calculate_checksum(data)
        if isinstance(data, pd.DataFrame):
            row_count, column_count = data.shape
            size_bytes = len(data.to_string(index=False).encode())
        else:
            row_count, column_count = 0, 0
            size_bytes = data.stat().st_size

        # Determine storage path
        version_dir = self.storage_root / dataset_id / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        storage_path = str(version_dir / f"data.{data.suffix if isinstance(data, Path) else 'parquet'}")

        # Save data
        if isinstance(data, pd.DataFrame):
            data.to_parquet(storage_path)
        else:
            shutil.copy2(data, storage_path)

        # Create version
        version = DataVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            parent_version_id=parent_version_id,
            checksum=checksum,
            size_bytes=size_bytes,
            row_count=row_count,
            column_count=column_count,
            format='parquet' if isinstance(data, pd.DataFrame) else data.suffix,
            storage_path=storage_path,
            metadata=metadata or {},
            tags=tags or [],
            description=description
        )

        # Add to lineage
        lineage.versions[version_id] = version
        lineage.updated_at = datetime.now()

        # Update indexes
        self.version_index[version_id] = dataset_id
        for tag in version.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(version_id)

        # Save changes
        self._save_lineages()

        self.logger.info(f"Created version {version_id} for dataset {dataset_id}")
        return version

    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get version by ID.

        Args:
            version_id: Version identifier

        Returns:
            DataVersion or None if not found
        """
        dataset_id = self.version_index.get(version_id)
        if dataset_id and dataset_id in self.lineages:
            return self.lineages[dataset_id].versions.get(version_id)
        return None

    def get_dataset_versions(self, dataset_id: str) -> List[DataVersion]:
        """Get all versions for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            List of DataVersion objects
        """
        if dataset_id not in self.lineages:
            return []

        lineage = self.lineages[dataset_id]
        return sorted(lineage.versions.values(), key=lambda v: v.timestamp)

    def get_version_lineage(self, version_id: str) -> List[DataVersion]:
        """Get complete lineage for a version.

        Args:
            version_id: Version identifier

        Returns:
            List of versions in lineage order (oldest first)
        """
        version = self.get_version(version_id)
        if not version:
            return []

        lineage = []
        current_version = version

        # Trace back to root
        while current_version:
            lineage.insert(0, current_version)
            current_version = self.get_version(current_version.parent_version_id) if current_version.parent_version_id else None

        return lineage

    def load_version_data(self, version_id: str) -> pd.DataFrame:
        """Load data for a specific version.

        Args:
            version_id: Version identifier

        Returns:
            DataFrame with version data
        """
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")

        storage_path = Path(version.storage_path)
        if not storage_path.exists():
            raise FileNotFoundError(f"Version data not found: {storage_path}")

        # Load based on format
        if version.format == 'parquet':
            return pd.read_parquet(storage_path)
        elif version.format == 'csv':
            return pd.read_csv(storage_path)
        elif version.format == 'json':
            return pd.read_json(storage_path)
        else:
            raise ValueError(f"Unsupported format: {version.format}")

    def record_processing_step(self, input_version_id: str, output_version_id: str,
                             operation: str, parameters: Dict[str, Any],
                             execution_time: float, success: bool = True,
                             error_message: Optional[str] = None,
                             config_hash: str = "") -> ProcessingStep:
        """Record a processing step in the lineage.

        Args:
            input_version_id: Input version ID
            output_version_id: Output version ID
            operation: Operation name
            parameters: Operation parameters
            execution_time: Execution time in seconds
            success: Whether operation was successful
            error_message: Error message if failed
            config_hash: Hash of configuration used

        Returns:
            Created ProcessingStep
        """
        # Get dataset IDs
        input_dataset = self.version_index.get(input_version_id)
        output_dataset = self.version_index.get(output_version_id)

        if not input_dataset or not output_dataset:
            raise ValueError("Invalid version IDs")

        # Create processing step
        step = ProcessingStep(
            step_id=str(uuid.uuid4()),
            operation=operation,
            input_version_id=input_version_id,
            output_version_id=output_version_id,
            parameters=parameters,
            execution_time=execution_time,
            success=success,
            error_message=error_message,
            environment_info=self._capture_environment_info(),
            config_hash=config_hash
        )

        # Add to lineages
        for dataset_id in [input_dataset, output_dataset]:
            if dataset_id in self.lineages:
                self.lineages[dataset_id].processing_steps.append(step)
                self.lineages[dataset_id].updated_at = datetime.now()

        # Save changes
        self._save_lineages()

        self.logger.info(f"Recorded processing step {step.step_id}: {operation}")
        return step

    def get_processing_history(self, dataset_id: str) -> List[ProcessingStep]:
        """Get processing history for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            List of ProcessingStep objects
        """
        if dataset_id not in self.lineages:
            return []

        lineage = self.lineages[dataset_id]
        return sorted(lineage.processing_steps, key=lambda s: s.timestamp)

    def find_versions_by_tag(self, tag: str) -> List[DataVersion]:
        """Find versions by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of DataVersion objects
        """
        version_ids = self.tag_index.get(tag, [])
        versions = []

        for version_id in version_ids:
            version = self.get_version(version_id)
            if version:
                versions.append(version)

        return sorted(versions, key=lambda v: v.timestamp)

    def get_dataset_summary(self, dataset_id: str) -> Dict[str, Any]:
        """Get summary information for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset summary
        """
        if dataset_id not in self.lineages:
            return {'error': f'Dataset {dataset_id} not found'}

        lineage = self.lineages[dataset_id]
        versions = self.get_dataset_versions(dataset_id)

        if not versions:
            return {'error': f'No versions found for dataset {dataset_id}'}

        # Calculate summary statistics
        total_versions = len(versions)
        total_size = sum(v.size_bytes for v in versions)
        total_rows = sum(v.row_count for v in versions)
        avg_row_count = total_rows / total_versions if total_versions > 0 else 0

        # Find latest version
        latest_version = versions[-1]

        # Get processing statistics
        processing_steps = self.get_processing_history(dataset_id)
        successful_steps = sum(1 for s in processing_steps if s.success)
        total_processing_time = sum(s.execution_time for s in processing_steps)

        return {
            'dataset_id': dataset_id,
            'total_versions': total_versions,
            'total_size_bytes': total_size,
            'total_rows': total_rows,
            'average_rows_per_version': avg_row_count,
            'latest_version': latest_version.version_id,
            'latest_version_timestamp': latest_version.timestamp.isoformat(),
            'created_at': lineage.created_at.isoformat(),
            'updated_at': lineage.updated_at.isoformat(),
            'processing_stats': {
                'total_steps': len(processing_steps),
                'successful_steps': successful_steps,
                'success_rate': successful_steps / len(processing_steps) if processing_steps else 0,
                'total_processing_time': total_processing_time
            },
            'tags': list(set(tag for v in versions for tag in v.tags))
        }

    def reproduce_version(self, version_id: str) -> Dict[str, Any]:
        """Generate reproduction instructions for a version.

        Args:
            version_id: Version identifier

        Returns:
            Reproduction instructions
        """
        lineage = self.get_version_lineage(version_id)
        if not lineage:
            return {'error': f'Version {version_id} not found'}

        target_version = lineage[-1]
        dataset_id = target_version.dataset_id

        # Get processing steps that led to this version
        processing_steps = self.get_processing_history(dataset_id)
        relevant_steps = []

        for step in processing_steps:
            if step.output_version_id == version_id:
                relevant_steps.append(step)
            elif any(v.parent_version_id == step.input_version_id for v in lineage):
                relevant_steps.append(step)

        # Sort by timestamp
        relevant_steps.sort(key=lambda s: s.timestamp)

        return {
            'version_id': version_id,
            'dataset_id': dataset_id,
            'description': target_version.description,
            'created_at': target_version.timestamp.isoformat(),
            'environment_requirements': relevant_steps[0].environment_info if relevant_steps else {},
            'reproduction_steps': [
                {
                    'step': i + 1,
                    'operation': step.operation,
                    'input_version': step.input_version_id,
                    'output_version': step.output_version_id,
                    'parameters': step.parameters,
                    'config_hash': step.config_hash,
                    'execution_time': step.execution_time,
                    'success': step.success
                }
                for i, step in enumerate(relevant_steps)
            ],
            'lineage': [v.version_id for v in lineage],
            'metadata': target_version.metadata
        }

    def cleanup_old_versions(self, dataset_id: str, keep_versions: int = 10,
                           keep_tags: Optional[List[str]] = None) -> int:
        """Clean up old versions, keeping only the most recent ones.

        Args:
            dataset_id: Dataset identifier
            keep_versions: Number of versions to keep
            keep_tags: Tags that should always be kept

        Returns:
            Number of versions removed
        """
        if dataset_id not in self.lineages:
            return 0

        lineage = self.lineages[dataset_id]
        versions = self.get_dataset_versions(dataset_id)

        if len(versions) <= keep_versions:
            return 0

        keep_tags = keep_tags or []
        versions_to_remove = []

        # Start from oldest versions
        for version in versions[:-keep_versions]:
            # Don't remove versions with protected tags
            if any(tag in version.tags for tag in keep_tags):
                continue

            # Don't remove root version
            if version.version_id == lineage.root_version_id:
                continue

            versions_to_remove.append(version)

        # Remove versions
        removed_count = 0
        for version in versions_to_remove:
            # Remove from storage
            storage_path = Path(version.storage_path)
            if storage_path.exists():
                storage_path.parent.rmdir()  # Remove version directory

            # Remove from lineage
            del lineage.versions[version.version_id]

            # Update indexes
            del self.version_index[version.version_id]
            for tag in version.tags:
                if tag in self.tag_index and version.version_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(version.version_id)

            removed_count += 1

        if removed_count > 0:
            lineage.updated_at = datetime.now()
            self._save_lineages()
            self.logger.info(f"Cleaned up {removed_count} old versions for dataset {dataset_id}")

        return removed_count

    def export_lineage(self, dataset_id: str, output_path: str) -> str:
        """Export lineage information to file.

        Args:
            dataset_id: Dataset identifier
            output_path: Path to export lineage

        Returns:
            Path where lineage was exported
        """
        if dataset_id not in self.lineages:
            raise ValueError(f"Dataset {dataset_id} not found")

        lineage = self.lineages[dataset_id]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(lineage.to_dict(), f, indent=2)

        self.logger.info(f"Exported lineage for dataset {dataset_id} to {output_path}")
        return str(output_path)

    @contextmanager
    def version_context(self, dataset_id: str, data: Union[pd.DataFrame, Path],
                       description: str = "", tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Context manager for automatic versioning.

        Args:
            dataset_id: Dataset identifier
            data: Data to version
            description: Version description
            tags: Version tags
            metadata: Additional metadata
        """
        version = None
        try:
            version = self.create_version(
                dataset_id=dataset_id,
                data=data,
                description=description,
                tags=tags,
                metadata=metadata
            )
            yield version
        except Exception as e:
            if version:
                self.logger.error(f"Error during versioning context: {e}")
            raise


# Global data versioning service instance
data_versioning_service = DataVersioningService()


def get_data_versioning_service() -> DataVersioningService:
    """Get the global data versioning service instance."""
    return data_versioning_service