"""
Training monitoring and logging for the training pipeline.
"""
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path

from .pipeline import TrainingJob, TrainingMetrics, TrainingStatus


@dataclass
class TrainingEvent:
    """Represents a training event for logging."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    job_id: str = ""
    event_type: str = ""  # epoch_complete, checkpoint_saved, error, etc.
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    level: str = "info"  # debug, info, warning, error, critical


@dataclass
class SystemMetrics:
    """System resource metrics during training."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    network_io_mb: float = 0.0
    temperature_celsius: float = 0.0


@dataclass
class TrainingSnapshot:
    """Snapshot of training state at a point in time."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    job_id: str = ""
    epoch: int = 0
    metrics: Optional[TrainingMetrics] = None
    system_metrics: Optional[SystemMetrics] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


class TrainingLogger:
    """Logger for training events and metrics."""
    
    def __init__(self, log_dir: str = "./training_logs", max_events: int = 10000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_events = max_events
        
        self._events: deque = deque(maxlen=max_events)
        self._job_logs: Dict[str, List[TrainingEvent]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def log_event(self, event: TrainingEvent) -> None:
        """Log a training event."""
        with self._lock:
            self._events.append(event)
            self._job_logs[event.job_id].append(event)
            
            # Write to file
            self._write_event_to_file(event)
    
    def log(
        self,
        job_id: str,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "info"
    ) -> None:
        """Log a training event with given parameters."""
        event = TrainingEvent(
            job_id=job_id,
            event_type=event_type,
            message=message,
            data=data or {},
            level=level
        )
        self.log_event(event)
    
    def _write_event_to_file(self, event: TrainingEvent) -> None:
        """Write event to log file."""
        try:
            # Create job-specific log file
            log_file = self.log_dir / f"{event.job_id}.log"
            
            log_entry = {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "level": event.level,
                "message": event.message,
                "data": event.data
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception:
            # Don't let logging errors break training
            pass
    
    def get_events(
        self,
        job_id: Optional[str] = None,
        event_type: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100
    ) -> List[TrainingEvent]:
        """Get filtered training events."""
        with self._lock:
            if job_id:
                events = list(self._job_logs.get(job_id, []))
            else:
                events = list(self._events)
            
            # Apply filters
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            if level:
                events = [e for e in events if e.level == level]
            
            # Sort by timestamp (most recent first) and limit
            events.sort(key=lambda x: x.timestamp, reverse=True)
            return events[:limit]
    
    def get_job_summary(self, job_id: str) -> Dict[str, Any]:
        """Get summary of events for a job."""
        events = self._job_logs.get(job_id, [])
        
        if not events:
            return {"job_id": job_id, "total_events": 0}
        
        # Count events by type and level
        event_counts = defaultdict(int)
        level_counts = defaultdict(int)
        
        for event in events:
            event_counts[event.event_type] += 1
            level_counts[event.level] += 1
        
        # Get first and last events
        sorted_events = sorted(events, key=lambda x: x.timestamp)
        first_event = sorted_events[0] if sorted_events else None
        last_event = sorted_events[-1] if sorted_events else None
        
        return {
            "job_id": job_id,
            "total_events": len(events),
            "event_types": dict(event_counts),
            "levels": dict(level_counts),
            "first_event": first_event.timestamp if first_event else None,
            "last_event": last_event.timestamp if last_event else None
        }
    
    def export_job_logs(self, job_id: str, format_type: str = "json") -> str:
        """Export logs for a job in specified format."""
        events = self._job_logs.get(job_id, [])
        
        if format_type.lower() == "json":
            export_data = {
                "job_id": job_id,
                "export_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "total_events": len(events),
                "events": [
                    {
                        "timestamp": e.timestamp,
                        "event_type": e.event_type,
                        "level": e.level,
                        "message": e.message,
                        "data": e.data
                    }
                    for e in sorted(events, key=lambda x: x.timestamp)
                ]
            }
            return json.dumps(export_data, indent=2)
        
        elif format_type.lower() == "csv":
            lines = ["timestamp,event_type,level,message,data"]
            for event in sorted(events, key=lambda x: x.timestamp):
                data_str = json.dumps(event.data).replace('"', '""')
                lines.append(f'"{event.timestamp}","{event.event_type}","{event.level}","{event.message}","{data_str}"')
            return '\n'.join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


class MetricsCollector:
    """Collects and aggregates training metrics."""
    
    def __init__(self, collection_interval_seconds: int = 10):
        self.collection_interval = collection_interval_seconds
        self._snapshots: Dict[str, List[TrainingSnapshot]] = defaultdict(list)
        self._system_metrics: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._collecting = False
        self._collection_thread: Optional[threading.Thread] = None
    
    def start_collection(self) -> None:
        """Start metrics collection."""
        if not self._collecting:
            self._collecting = True
            self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self._collection_thread.start()
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
    
    def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._collecting:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                with self._lock:
                    self._system_metrics.append(system_metrics)
                
                time.sleep(self.collection_interval)
            except Exception:
                # Don't let collection errors stop the loop
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # In a real implementation, this would use psutil or similar
        # For now, return simulated metrics
        import random
        
        return SystemMetrics(
            cpu_usage_percent=random.uniform(20, 80),
            memory_usage_mb=random.uniform(1000, 8000),
            gpu_usage_percent=random.uniform(50, 95),
            gpu_memory_usage_mb=random.uniform(2000, 10000),
            disk_usage_mb=random.uniform(100, 1000),
            network_io_mb=random.uniform(10, 100),
            temperature_celsius=random.uniform(60, 85)
        )
    
    def record_training_snapshot(self, snapshot: TrainingSnapshot) -> None:
        """Record a training snapshot."""
        with self._lock:
            self._snapshots[snapshot.job_id].append(snapshot)
            
            # Limit snapshots per job
            if len(self._snapshots[snapshot.job_id]) > 1000:
                self._snapshots[snapshot.job_id] = self._snapshots[snapshot.job_id][-1000:]
    
    def get_training_history(
        self,
        job_id: str,
        metric_names: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get training history for a job."""
        with self._lock:
            snapshots = self._snapshots.get(job_id, [])
            
            if not snapshots:
                return []
            
            # Sort by epoch
            sorted_snapshots = sorted(snapshots, key=lambda x: x.epoch)
            
            # Limit results
            if limit:
                sorted_snapshots = sorted_snapshots[-limit:]
            
            # Extract requested metrics
            history = []
            for snapshot in sorted_snapshots:
                entry = {
                    "timestamp": snapshot.timestamp,
                    "epoch": snapshot.epoch
                }
                
                if snapshot.metrics:
                    if metric_names:
                        for metric in metric_names:
                            if hasattr(snapshot.metrics, metric):
                                entry[metric] = getattr(snapshot.metrics, metric)
                    else:
                        # Include all metrics
                        entry.update(snapshot.metrics.to_dict())
                
                if snapshot.system_metrics:
                    entry["system"] = {
                        "cpu_usage": snapshot.system_metrics.cpu_usage_percent,
                        "memory_usage": snapshot.system_metrics.memory_usage_mb,
                        "gpu_usage": snapshot.system_metrics.gpu_usage_percent,
                        "gpu_memory": snapshot.system_metrics.gpu_memory_usage_mb
                    }
                
                history.append(entry)
            
            return history
    
    def get_metrics_summary(self, job_id: str) -> Dict[str, Any]:
        """Get summary statistics for job metrics."""
        with self._lock:
            snapshots = self._snapshots.get(job_id, [])
            
            if not snapshots:
                return {"job_id": job_id, "total_snapshots": 0}
            
            # Calculate summary statistics
            metrics_data = []
            for snapshot in snapshots:
                if snapshot.metrics:
                    metrics_data.append(snapshot.metrics)
            
            if not metrics_data:
                return {"job_id": job_id, "total_snapshots": len(snapshots), "metrics": None}
            
            # Calculate averages, min, max for key metrics
            summary = {
                "job_id": job_id,
                "total_snapshots": len(snapshots),
                "total_epochs": max(s.epoch for s in snapshots),
                "metrics": {
                    "train_loss": {
                        "final": metrics_data[-1].train_loss,
                        "best": min(m.train_loss for m in metrics_data if m.train_loss > 0),
                        "average": sum(m.train_loss for m in metrics_data) / len(metrics_data)
                    },
                    "val_loss": {
                        "final": metrics_data[-1].val_loss,
                        "best": min(m.val_loss for m in metrics_data if m.val_loss > 0),
                        "average": sum(m.val_loss for m in metrics_data) / len(metrics_data)
                    },
                    "map50": {
                        "final": metrics_data[-1].map50,
                        "best": max(m.map50 for m in metrics_data),
                        "average": sum(m.map50 for m in metrics_data) / len(metrics_data)
                    },
                    "map50_95": {
                        "final": metrics_data[-1].map50_95,
                        "best": max(m.map50_95 for m in metrics_data),
                        "average": sum(m.map50_95 for m in metrics_data) / len(metrics_data)
                    }
                }
            }
            
            return summary
    
    def get_system_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get system metrics history for specified time window."""
        with self._lock:
            cutoff_time = time.time() - (minutes * 60)
            
            recent_metrics = []
            for metrics in self._system_metrics:
                try:
                    metrics_time = datetime.fromisoformat(
                        metrics.timestamp.replace("Z", "+00:00")
                    ).timestamp()
                    
                    if metrics_time > cutoff_time:
                        recent_metrics.append(metrics)
                except ValueError:
                    # Skip invalid timestamps
                    continue
            
            return sorted(recent_metrics, key=lambda x: x.timestamp)


class TrainingMonitor:
    """Main training monitor that coordinates logging and metrics collection."""
    
    def __init__(
        self,
        log_dir: str = "./training_logs",
        enable_system_metrics: bool = True,
        metrics_collection_interval: int = 10
    ):
        self.logger = TrainingLogger(log_dir)
        self.metrics_collector = MetricsCollector(metrics_collection_interval)
        
        self._monitored_jobs: Dict[str, TrainingJob] = {}
        self._callbacks: Dict[str, List[Callable[[TrainingJob], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Start metrics collection if enabled
        if enable_system_metrics:
            self.metrics_collector.start_collection()
    
    def register_job(self, job: TrainingJob) -> None:
        """Register a job for monitoring."""
        with self._lock:
            self._monitored_jobs[job.job_id] = job
            
            # Add progress callback to job
            job.progress_callbacks.append(self._on_job_progress)
            
            # Log job registration
            self.logger.log(
                job.job_id,
                "job_registered",
                f"Training job registered: {job.config.experiment_name}",
                {
                    "model_type": job.config.model_type,
                    "variant": job.config.variant,
                    "dataset_path": job.config.dataset_path,
                    "epochs": job.config.hyperparameters.epochs
                }
            )
    
    def unregister_job(self, job_id: str) -> None:
        """Unregister a job from monitoring."""
        with self._lock:
            if job_id in self._monitored_jobs:
                job = self._monitored_jobs[job_id]
                
                # Remove progress callback
                if self._on_job_progress in job.progress_callbacks:
                    job.progress_callbacks.remove(self._on_job_progress)
                
                del self._monitored_jobs[job_id]
                
                # Log job unregistration
                self.logger.log(
                    job_id,
                    "job_unregistered",
                    "Training job unregistered"
                )
    
    def _on_job_progress(self, job: TrainingJob) -> None:
        """Callback for job progress updates."""
        # Log progress
        self.logger.log(
            job.job_id,
            "epoch_complete",
            f"Epoch {job.current_epoch}/{job.total_epochs} completed ({job.progress_percentage:.1f}%)",
            {
                "epoch": job.current_epoch,
                "progress_percentage": job.progress_percentage,
                "metrics": job.latest_metrics.to_dict() if job.latest_metrics else None
            }
        )
        
        # Record training snapshot
        if job.latest_metrics:
            snapshot = TrainingSnapshot(
                job_id=job.job_id,
                epoch=job.current_epoch,
                metrics=job.latest_metrics,
                system_metrics=self.metrics_collector._collect_system_metrics(),
                hyperparameters=job.config.hyperparameters.to_dict()
            )
            
            self.metrics_collector.record_training_snapshot(snapshot)
        
        # Call registered callbacks
        for callback in self._callbacks.get(job.job_id, []):
            try:
                callback(job)
            except Exception:
                # Don't let callback errors break monitoring
                pass
    
    def add_job_callback(
        self,
        job_id: str,
        callback: Callable[[TrainingJob], None]
    ) -> None:
        """Add a callback for job progress updates."""
        with self._lock:
            self._callbacks[job_id].append(callback)
    
    def remove_job_callback(
        self,
        job_id: str,
        callback: Callable[[TrainingJob], None]
    ) -> None:
        """Remove a job progress callback."""
        with self._lock:
            if job_id in self._callbacks:
                if callback in self._callbacks[job_id]:
                    self._callbacks[job_id].remove(callback)
    
    def log_training_event(
        self,
        job_id: str,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "info"
    ) -> None:
        """Log a training event."""
        self.logger.log(job_id, event_type, message, data, level)
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive job status."""
        with self._lock:
            job = self._monitored_jobs.get(job_id)
            if not job:
                return None
            
            # Get recent events
            recent_events = self.logger.get_events(job_id, limit=10)
            
            # Get metrics summary
            metrics_summary = self.metrics_collector.get_metrics_summary(job_id)
            
            # Get training history
            training_history = self.metrics_collector.get_training_history(
                job_id, 
                ["train_loss", "val_loss", "map50", "map50_95"], 
                limit=20
            )
            
            return {
                "job_id": job_id,
                "job_status": {
                    "status": job.status,
                    "progress_percentage": job.progress_percentage,
                    "current_epoch": job.current_epoch,
                    "total_epochs": job.total_epochs,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at
                },
                "latest_metrics": job.latest_metrics.to_dict() if job.latest_metrics else None,
                "best_metrics": job.best_metrics.to_dict() if job.best_metrics else None,
                "recent_events": [
                    {
                        "timestamp": e.timestamp,
                        "event_type": e.event_type,
                        "message": e.message,
                        "level": e.level
                    }
                    for e in recent_events
                ],
                "metrics_summary": metrics_summary,
                "training_history": training_history[-10:],  # Last 10 epochs
                "warnings": job.warnings,
                "error_message": job.error_message
            }
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        with self._lock:
            # Get all monitored jobs
            active_jobs = [job for job in self._monitored_jobs.values() if job.is_active]
            completed_jobs = [job for job in self._monitored_jobs.values() if job.is_complete]
            
            # Get system metrics
            system_metrics = self.metrics_collector.get_system_metrics_history(minutes=30)
            
            # Calculate overall statistics
            total_jobs = len(self._monitored_jobs)
            success_rate = 0.0
            
            if total_jobs > 0:
                successful_jobs = len([job for job in self._monitored_jobs.values() 
                                    if job.status == TrainingStatus.COMPLETED])
                success_rate = (successful_jobs / total_jobs) * 100.0
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "summary": {
                    "total_jobs": total_jobs,
                    "active_jobs": len(active_jobs),
                    "completed_jobs": len(completed_jobs),
                    "success_rate": success_rate
                },
                "active_jobs": [
                    {
                        "job_id": job.job_id,
                        "experiment_name": job.config.experiment_name,
                        "status": job.status,
                        "progress_percentage": job.progress_percentage,
                        "current_epoch": job.current_epoch,
                        "total_epochs": job.total_epochs,
                        "latest_map50": job.latest_metrics.map50 if job.latest_metrics else 0.0
                    }
                    for job in active_jobs
                ],
                "system_metrics": {
                    "current": system_metrics[-1].__dict__ if system_metrics else None,
                    "history": [m.__dict__ for m in system_metrics[-20:]]  # Last 20 readings
                }
            }
    
    def export_monitoring_data(
        self,
        job_id: Optional[str] = None,
        format_type: str = "json"
    ) -> str:
        """Export monitoring data."""
        if job_id:
            # Export specific job data
            job_status = self.get_job_status(job_id)
            logs = self.logger.export_job_logs(job_id, format_type)
            
            if format_type.lower() == "json":
                export_data = {
                    "export_type": "job_data",
                    "job_id": job_id,
                    "job_status": job_status,
                    "logs": json.loads(logs) if format_type == "json" else logs
                }
                return json.dumps(export_data, indent=2)
            else:
                return logs
        else:
            # Export dashboard data
            dashboard_data = self.get_monitoring_dashboard()
            
            if format_type.lower() == "json":
                return json.dumps(dashboard_data, indent=2)
            else:
                # CSV format for dashboard
                lines = ["timestamp,job_id,status,progress,current_epoch,total_epochs"]
                for job in dashboard_data.get("active_jobs", []):
                    lines.append(f"{dashboard_data['timestamp']},{job['job_id']},{job['status']},{job['progress_percentage']},{job['current_epoch']},{job['total_epochs']}")
                return '\n'.join(lines)
    
    def shutdown(self) -> None:
        """Shutdown the training monitor."""
        self.metrics_collector.stop_collection()
        
        # Unregister all jobs
        with self._lock:
            job_ids = list(self._monitored_jobs.keys())
            for job_id in job_ids:
                self.unregister_job(job_id)