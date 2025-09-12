"""
In-memory storage for performance metrics (temporary implementation for TDD GREEN phase).
Later this will be replaced with proper SQLAlchemy models and database storage.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid


class PerformanceMetricModel:
    """Temporary in-memory performance metric model for TDD GREEN phase."""
    
    def __init__(self, model_id: str, metric_type: str, metric_value: float,
                 dataset_id: Optional[str] = None, threshold: Optional[float] = None,
                 class_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.metric_type = metric_type  # mAP, IoU, precision, recall, F1, execution_time
        self.metric_value = metric_value
        self.threshold = threshold  # IoU threshold for mAP calculations
        self.class_name = class_name  # For class-specific metrics
        self.evaluation_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.metadata = metadata or {}


class PerformanceMetricStorage:
    """Temporary in-memory storage for performance metrics."""
    
    def __init__(self):
        self._performance_metrics: Dict[str, PerformanceMetricModel] = {}
    
    def save(self, performance_metric: PerformanceMetricModel) -> PerformanceMetricModel:
        """Save a performance metric to storage."""
        self._performance_metrics[performance_metric.id] = performance_metric
        return performance_metric
    
    def get_by_id(self, performance_metric_id: str) -> Optional[PerformanceMetricModel]:
        """Get a performance metric by ID."""
        return self._performance_metrics.get(performance_metric_id)
    
    def list_performance_metrics(self, model_id: Optional[str] = None, dataset_id: Optional[str] = None,
                                metric_type: Optional[str] = None, class_name: Optional[str] = None,
                                limit: int = 50, offset: int = 0) -> tuple[List[PerformanceMetricModel], int]:
        """List performance metrics with optional filtering and pagination."""
        metrics = list(self._performance_metrics.values())
        
        # Apply filters
        if model_id:
            metrics = [m for m in metrics if m.model_id == model_id]
        if dataset_id:
            metrics = [m for m in metrics if m.dataset_id == dataset_id]
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]
        if class_name:
            metrics = [m for m in metrics if m.class_name == class_name]
        
        # Sort by evaluation_timestamp descending
        metrics.sort(key=lambda x: x.evaluation_timestamp, reverse=True)
        
        total_count = len(metrics)
        
        # Apply pagination
        paginated_metrics = metrics[offset:offset + limit]
        
        return paginated_metrics, total_count
    
    def get_metrics_by_model(self, model_id: str, metric_type: Optional[str] = None) -> List[PerformanceMetricModel]:
        """Get all performance metrics for a specific model."""
        metrics = [m for m in self._performance_metrics.values() if m.model_id == model_id]
        
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]
        
        return metrics
    
    def get_metrics_by_dataset(self, dataset_id: str, metric_type: Optional[str] = None) -> List[PerformanceMetricModel]:
        """Get all performance metrics for a specific dataset."""
        metrics = [m for m in self._performance_metrics.values() if m.dataset_id == dataset_id]
        
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]
        
        return metrics
    
    def get_latest_metric(self, model_id: str, metric_type: str, 
                         dataset_id: Optional[str] = None, class_name: Optional[str] = None) -> Optional[PerformanceMetricModel]:
        """Get the most recent metric for a model, metric type, and optional filters."""
        metrics = self.get_metrics_by_model(model_id, metric_type)
        
        if dataset_id:
            metrics = [m for m in metrics if m.dataset_id == dataset_id]
        if class_name:
            metrics = [m for m in metrics if m.class_name == class_name]
        
        if not metrics:
            return None
        
        # Return the most recent metric
        return max(metrics, key=lambda x: x.evaluation_timestamp)
    
    def get_class_metrics(self, model_id: str, dataset_id: Optional[str] = None) -> Dict[str, List[PerformanceMetricModel]]:
        """Get performance metrics grouped by class name."""
        metrics = self.get_metrics_by_model(model_id)
        
        if dataset_id:
            metrics = [m for m in metrics if m.dataset_id == dataset_id]
        
        # Group by class name
        class_metrics = {}
        for metric in metrics:
            if metric.class_name:
                if metric.class_name not in class_metrics:
                    class_metrics[metric.class_name] = []
                class_metrics[metric.class_name].append(metric)
        
        return class_metrics


# Global storage instance (temporary for TDD)
performance_metric_storage = PerformanceMetricStorage()