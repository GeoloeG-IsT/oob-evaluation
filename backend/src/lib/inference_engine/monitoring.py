"""
Performance monitoring and metrics collection for inference operations.
"""
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import json

from .engine import InferenceResult, BatchInferenceJob, InferenceStatus


@dataclass
class InferenceMetrics:
    """Detailed metrics for inference operations."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    model_id: str = ""
    request_id: str = ""
    image_path: str = ""
    status: str = ""
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    error_message: Optional[str] = None
    predictions_count: int = 0
    confidence_scores: List[float] = field(default_factory=list)


@dataclass 
class ModelPerformanceStats:
    """Aggregated performance statistics for a model."""
    model_id: str = ""
    total_inferences: int = 0
    successful_inferences: int = 0
    failed_inferences: int = 0
    success_rate: float = 0.0
    average_inference_time_ms: float = 0.0
    min_inference_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    std_inference_time_ms: float = 0.0
    average_total_time_ms: float = 0.0
    average_throughput_fps: float = 0.0
    total_predictions: int = 0
    average_predictions_per_image: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


@dataclass
class SystemPerformanceMetrics:
    """System-wide performance metrics."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    total_inferences: int = 0
    active_jobs: int = 0
    queue_size: int = 0
    average_response_time_ms: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class PerformanceMonitor:
    """Monitor and collect performance metrics for inference operations."""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self._metrics_history: deque = deque(maxlen=max_history_size)
        self._model_stats: Dict[str, ModelPerformanceStats] = {}
        self._system_metrics: deque = deque(maxlen=1000)  # Keep last 1000 system snapshots
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[InferenceMetrics], None]] = []
        
        # Performance tracking
        self._request_times: Dict[str, float] = {}
        self._recent_requests: deque = deque(maxlen=100)  # Track last 100 requests for RPS calculation
    
    def register_callback(self, callback: Callable[[InferenceMetrics], None]) -> None:
        """Register a callback to be called when metrics are recorded."""
        with self._lock:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[InferenceMetrics], None]) -> None:
        """Unregister a callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def start_request_timing(self, request_id: str) -> None:
        """Start timing a request."""
        with self._lock:
            self._request_times[request_id] = time.time()
    
    def record_inference_result(self, result: InferenceResult, image_path: str = "") -> None:
        """Record metrics from an inference result."""
        with self._lock:
            # Calculate confidence scores
            confidence_scores = []
            if result.predictions:
                confidence_scores = [
                    pred.get("confidence", 0.0) 
                    for pred in result.predictions 
                    if "confidence" in pred
                ]
            
            # Create metrics record
            metrics = InferenceMetrics(
                model_id=result.model_id,
                request_id=result.request_id,
                image_path=image_path,
                status=result.status,
                inference_time_ms=result.performance_metrics.inference_time_ms,
                total_time_ms=result.performance_metrics.total_time_ms,
                memory_usage_mb=result.performance_metrics.memory_usage_mb,
                error_message=result.error_message,
                predictions_count=len(result.predictions),
                confidence_scores=confidence_scores
            )
            
            # Add to history
            self._metrics_history.append(metrics)
            
            # Update model statistics
            self._update_model_stats(metrics)
            
            # Track request timing
            current_time = time.time()
            self._recent_requests.append(current_time)
            
            # Clean up timing data
            if result.request_id in self._request_times:
                del self._request_times[result.request_id]
            
            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(metrics)
                except Exception:
                    pass  # Don't let callback errors break monitoring
    
    def record_batch_job_metrics(self, job: BatchInferenceJob) -> None:
        """Record metrics from a batch job."""
        with self._lock:
            for result in job.results:
                # Extract image path from job if available
                image_path = ""
                try:
                    if job.image_paths:
                        result_index = job.results.index(result)
                        if result_index < len(job.image_paths):
                            image_path = job.image_paths[result_index]
                except (ValueError, IndexError):
                    pass
                
                self.record_inference_result(result, image_path)
    
    def _update_model_stats(self, metrics: InferenceMetrics) -> None:
        """Update aggregated model statistics."""
        model_id = metrics.model_id
        
        if model_id not in self._model_stats:
            self._model_stats[model_id] = ModelPerformanceStats(model_id=model_id)
        
        stats = self._model_stats[model_id]
        
        # Update counters
        stats.total_inferences += 1
        if metrics.status == InferenceStatus.COMPLETED:
            stats.successful_inferences += 1
        else:
            stats.failed_inferences += 1
        
        # Update success rate
        stats.success_rate = (stats.successful_inferences / stats.total_inferences) * 100.0
        
        # Update timing statistics (only for successful inferences)
        if metrics.status == InferenceStatus.COMPLETED and metrics.inference_time_ms > 0:
            if stats.min_inference_time_ms == 0:
                stats.min_inference_time_ms = metrics.inference_time_ms
            else:
                stats.min_inference_time_ms = min(stats.min_inference_time_ms, metrics.inference_time_ms)
            
            stats.max_inference_time_ms = max(stats.max_inference_time_ms, metrics.inference_time_ms)
            
            # Running average for inference time
            n = stats.successful_inferences
            old_avg = stats.average_inference_time_ms
            stats.average_inference_time_ms = (old_avg * (n - 1) + metrics.inference_time_ms) / n
            
            # Running average for total time
            old_total_avg = stats.average_total_time_ms
            stats.average_total_time_ms = (old_total_avg * (n - 1) + metrics.total_time_ms) / n
            
            # Calculate throughput
            if metrics.total_time_ms > 0:
                throughput = 1000.0 / metrics.total_time_ms
                old_throughput_avg = stats.average_throughput_fps
                stats.average_throughput_fps = (old_throughput_avg * (n - 1) + throughput) / n
        
        # Update prediction statistics
        stats.total_predictions += metrics.predictions_count
        stats.average_predictions_per_image = stats.total_predictions / stats.total_inferences
        
        # Update confidence distribution
        for confidence in metrics.confidence_scores:
            # Create confidence buckets
            bucket = f"{int(confidence * 10) * 10}-{int(confidence * 10) * 10 + 10}%"
            if bucket not in stats.confidence_distribution:
                stats.confidence_distribution[bucket] = 0
            stats.confidence_distribution[bucket] += 1
        
        stats.last_updated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    def get_model_stats(self, model_id: str) -> Optional[ModelPerformanceStats]:
        """Get performance statistics for a specific model."""
        with self._lock:
            return self._model_stats.get(model_id)
    
    def get_all_model_stats(self) -> Dict[str, ModelPerformanceStats]:
        """Get performance statistics for all models."""
        with self._lock:
            return self._model_stats.copy()
    
    def get_recent_metrics(self, limit: int = 100, model_id: Optional[str] = None) -> List[InferenceMetrics]:
        """Get recent inference metrics."""
        with self._lock:
            metrics = list(self._metrics_history)
            
            if model_id:
                metrics = [m for m in metrics if m.model_id == model_id]
            
            return metrics[-limit:]
    
    def get_system_metrics(self) -> SystemPerformanceMetrics:
        """Get current system performance metrics."""
        with self._lock:
            current_time = time.time()
            
            # Calculate requests per second
            recent_cutoff = current_time - 60.0  # Last 60 seconds
            recent_count = sum(1 for req_time in self._recent_requests if req_time > recent_cutoff)
            rps = recent_count / 60.0
            
            # Calculate overall statistics
            total_inferences = len(self._metrics_history)
            error_count = sum(1 for m in self._metrics_history if m.status != InferenceStatus.COMPLETED)
            error_rate = (error_count / total_inferences * 100.0) if total_inferences > 0 else 0.0
            
            # Calculate average response time
            avg_response_time = 0.0
            if self._metrics_history:
                successful_metrics = [m for m in self._metrics_history if m.status == InferenceStatus.COMPLETED]
                if successful_metrics:
                    avg_response_time = sum(m.total_time_ms for m in successful_metrics) / len(successful_metrics)
            
            return SystemPerformanceMetrics(
                total_inferences=total_inferences,
                active_jobs=len(self._request_times),
                queue_size=0,  # Would need to be tracked by inference engine
                average_response_time_ms=avg_response_time,
                requests_per_second=rps,
                error_rate=error_rate,
                memory_usage_mb=0.0,  # Would need system monitoring
                cpu_usage_percent=0.0  # Would need system monitoring
            )
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - (time_window_minutes * 60)
            
            # Filter metrics to time window
            recent_metrics = [
                m for m in self._metrics_history
                if datetime.fromisoformat(m.timestamp.replace("Z", "+00:00")).timestamp() > cutoff_time
            ]
            
            if not recent_metrics:
                return {"time_window_minutes": time_window_minutes, "total_inferences": 0}
            
            successful_metrics = [m for m in recent_metrics if m.status == InferenceStatus.COMPLETED]
            failed_metrics = [m for m in recent_metrics if m.status != InferenceStatus.COMPLETED]
            
            # Model breakdown
            model_breakdown = defaultdict(lambda: {"count": 0, "success": 0, "fail": 0})
            for metric in recent_metrics:
                model_breakdown[metric.model_id]["count"] += 1
                if metric.status == InferenceStatus.COMPLETED:
                    model_breakdown[metric.model_id]["success"] += 1
                else:
                    model_breakdown[metric.model_id]["fail"] += 1
            
            # Performance statistics
            performance_stats = {}
            if successful_metrics:
                inference_times = [m.inference_time_ms for m in successful_metrics]
                total_times = [m.total_time_ms for m in successful_metrics]
                
                performance_stats = {
                    "average_inference_time_ms": sum(inference_times) / len(inference_times),
                    "min_inference_time_ms": min(inference_times),
                    "max_inference_time_ms": max(inference_times),
                    "average_total_time_ms": sum(total_times) / len(total_times),
                    "median_inference_time_ms": sorted(inference_times)[len(inference_times) // 2],
                    "p95_inference_time_ms": sorted(inference_times)[int(len(inference_times) * 0.95)]
                }
            
            return {
                "time_window_minutes": time_window_minutes,
                "summary": {
                    "total_inferences": len(recent_metrics),
                    "successful_inferences": len(successful_metrics),
                    "failed_inferences": len(failed_metrics),
                    "success_rate": len(successful_metrics) / len(recent_metrics) * 100.0,
                    "requests_per_minute": len(recent_metrics) / time_window_minutes
                },
                "performance": performance_stats,
                "model_breakdown": dict(model_breakdown),
                "error_breakdown": {
                    error.error_message or "Unknown": sum(
                        1 for m in failed_metrics 
                        if m.error_message == error.error_message
                    )
                    for error in failed_metrics
                }
            }
    
    def export_metrics(self, format_type: str = "json", model_id: Optional[str] = None) -> str:
        """Export metrics in specified format."""
        with self._lock:
            metrics = list(self._metrics_history)
            if model_id:
                metrics = [m for m in metrics if m.model_id == model_id]
            
            if format_type.lower() == "json":
                # Convert to JSON-serializable format
                export_data = {
                    "export_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "total_metrics": len(metrics),
                    "model_filter": model_id,
                    "metrics": [
                        {
                            "timestamp": m.timestamp,
                            "model_id": m.model_id,
                            "request_id": m.request_id,
                            "image_path": m.image_path,
                            "status": m.status,
                            "inference_time_ms": m.inference_time_ms,
                            "total_time_ms": m.total_time_ms,
                            "memory_usage_mb": m.memory_usage_mb,
                            "error_message": m.error_message,
                            "predictions_count": m.predictions_count,
                            "confidence_scores": m.confidence_scores
                        }
                        for m in metrics
                    ],
                    "model_stats": {
                        k: {
                            "model_id": v.model_id,
                            "total_inferences": v.total_inferences,
                            "successful_inferences": v.successful_inferences,
                            "failed_inferences": v.failed_inferences,
                            "success_rate": v.success_rate,
                            "average_inference_time_ms": v.average_inference_time_ms,
                            "min_inference_time_ms": v.min_inference_time_ms,
                            "max_inference_time_ms": v.max_inference_time_ms,
                            "average_total_time_ms": v.average_total_time_ms,
                            "average_throughput_fps": v.average_throughput_fps,
                            "total_predictions": v.total_predictions,
                            "average_predictions_per_image": v.average_predictions_per_image,
                            "confidence_distribution": v.confidence_distribution,
                            "last_updated": v.last_updated
                        }
                        for k, v in self._model_stats.items()
                        if not model_id or k == model_id
                    }
                }
                return json.dumps(export_data, indent=2)
            
            elif format_type.lower() == "csv":
                # CSV export
                csv_lines = [
                    "timestamp,model_id,request_id,image_path,status,inference_time_ms,total_time_ms,memory_usage_mb,predictions_count,avg_confidence,error_message"
                ]
                
                for m in metrics:
                    avg_confidence = sum(m.confidence_scores) / len(m.confidence_scores) if m.confidence_scores else 0.0
                    csv_lines.append(
                        f'"{m.timestamp}","{m.model_id}","{m.request_id}","{m.image_path}","{m.status}",'
                        f'{m.inference_time_ms},{m.total_time_ms},{m.memory_usage_mb},{m.predictions_count},'
                        f'{avg_confidence:.3f},"{m.error_message or ""}"'
                    )
                
                return "\n".join(csv_lines)
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
    
    def clear_metrics(self, older_than_hours: int = 24) -> int:
        """Clear old metrics to free memory."""
        with self._lock:
            if older_than_hours <= 0:
                # Clear all
                count = len(self._metrics_history)
                self._metrics_history.clear()
                self._model_stats.clear()
                self._system_metrics.clear()
                return count
            
            current_time = time.time()
            cutoff_time = current_time - (older_than_hours * 3600)
            
            # Filter metrics
            old_count = len(self._metrics_history)
            self._metrics_history = deque([
                m for m in self._metrics_history
                if datetime.fromisoformat(m.timestamp.replace("Z", "+00:00")).timestamp() > cutoff_time
            ], maxlen=self.max_history_size)
            
            # Recalculate model stats from remaining metrics
            self._model_stats.clear()
            for metric in self._metrics_history:
                self._update_model_stats(metric)
            
            return old_count - len(self._metrics_history)