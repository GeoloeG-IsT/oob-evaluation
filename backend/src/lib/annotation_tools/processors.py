"""
Batch annotation processing and quality checking utilities.
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import uuid

from .assistants import AnnotationAssistant, AssistanceRequest, AssistanceResult, AssistanceMode
from .tools import AnnotationShape, AnnotationValidator, ValidationLevel
from .converters import FormatConverter, ConversionConfig


class ProcessingStatus(str, Enum):
    """Status of batch processing operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingConfig:
    """Configuration for batch annotation processing."""
    # Processing options
    max_concurrent: int = 4
    batch_size: int = 10
    timeout_seconds: int = 300
    
    # Quality control
    enable_validation: bool = True
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    min_confidence_threshold: float = 0.5
    auto_fix_issues: bool = False
    
    # Output options
    output_format: str = "coco"
    save_intermediate: bool = False
    include_metadata: bool = True
    
    # Error handling
    continue_on_error: bool = True
    max_retries: int = 2
    retry_delay_seconds: float = 1.0


@dataclass
class ProcessingJob:
    """Batch processing job tracking."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ProcessingStatus = ProcessingStatus.PENDING
    
    # Input/Output
    input_images: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    
    # Progress tracking
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    successful_images: int = 0
    
    # Results
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Timing
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Configuration
    config: Optional[ProcessingConfig] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_images == 0:
            return 0.0
        return (self.processed_images / self.total_images) * 100.0
    
    @property
    def is_complete(self) -> bool:
        """Check if job is complete."""
        return self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(f"[{datetime.now(timezone.utc).isoformat()}] {message}")
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(f"[{datetime.now(timezone.utc).isoformat()}] {message}")


@dataclass
class QualityMetrics:
    """Quality metrics for annotations."""
    total_annotations: int = 0
    valid_annotations: int = 0
    invalid_annotations: int = 0
    
    # Confidence statistics
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    
    # Geometric statistics
    avg_area: float = 0.0
    min_area: float = 0.0
    max_area: float = 0.0
    
    # Error counts
    geometry_errors: int = 0
    size_errors: int = 0
    position_errors: int = 0
    quality_warnings: int = 0
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        if self.total_annotations == 0:
            return 0.0
        
        valid_ratio = self.valid_annotations / self.total_annotations
        confidence_factor = min(self.avg_confidence, 1.0) if self.avg_confidence > 0 else 0.5
        error_penalty = min(len([self.geometry_errors, self.size_errors, self.position_errors]) * 0.1, 0.5)
        
        return max(0.0, (valid_ratio * 0.6 + confidence_factor * 0.4) - error_penalty)


class QualityChecker:
    """Quality checker for annotation validation and improvement."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validator = AnnotationValidator(validation_level)
        self.stats = QualityMetrics()
    
    def check_annotations(
        self,
        annotations: List[AnnotationShape],
        image_width: int,
        image_height: int
    ) -> Dict[str, Any]:
        """Check quality of annotations."""
        
        # Reset stats
        self.stats = QualityMetrics()
        self.stats.total_annotations = len(annotations)
        
        if not annotations:
            return self._create_quality_report()
        
        # Validate annotations
        validation_result = self.validator.validate_annotations(
            annotations, image_width, image_height
        )
        
        # Update stats from validation
        self.stats.valid_annotations = validation_result["valid_annotations"]
        self.stats.invalid_annotations = validation_result["invalid_annotations"]
        
        # Calculate confidence statistics
        confidences = [ann.confidence for ann in annotations]
        if confidences:
            self.stats.avg_confidence = sum(confidences) / len(confidences)
            self.stats.min_confidence = min(confidences)
            self.stats.max_confidence = max(confidences)
        
        # Calculate area statistics
        areas = [ann.area for ann in annotations if ann.area > 0]
        if areas:
            self.stats.avg_area = sum(areas) / len(areas)
            self.stats.min_area = min(areas)
            self.stats.max_area = max(areas)
        
        # Count error types
        for ann_result in validation_result["annotation_results"]:
            validation_data = ann_result["validation_result"]
            
            for error in validation_data["errors"]:
                if "geometry" in error.lower():
                    self.stats.geometry_errors += 1
                elif "size" in error.lower() or "area" in error.lower():
                    self.stats.size_errors += 1
                elif "bound" in error.lower() or "position" in error.lower():
                    self.stats.position_errors += 1
            
            self.stats.quality_warnings += len(validation_data["warnings"])
        
        return self._create_quality_report()
    
    def _create_quality_report(self) -> Dict[str, Any]:
        """Create quality report from current stats."""
        return {
            "quality_score": self.stats.quality_score,
            "total_annotations": self.stats.total_annotations,
            "valid_annotations": self.stats.valid_annotations,
            "invalid_annotations": self.stats.invalid_annotations,
            "validation_rate": (
                self.stats.valid_annotations / self.stats.total_annotations * 100.0
                if self.stats.total_annotations > 0 else 0.0
            ),
            "confidence_stats": {
                "average": self.stats.avg_confidence,
                "minimum": self.stats.min_confidence,
                "maximum": self.stats.max_confidence
            },
            "area_stats": {
                "average": self.stats.avg_area,
                "minimum": self.stats.min_area,
                "maximum": self.stats.max_area
            },
            "error_counts": {
                "geometry_errors": self.stats.geometry_errors,
                "size_errors": self.stats.size_errors,
                "position_errors": self.stats.position_errors,
                "quality_warnings": self.stats.quality_warnings
            }
        }
    
    def auto_fix_annotations(
        self,
        annotations: List[AnnotationShape],
        image_width: int,
        image_height: int
    ) -> List[AnnotationShape]:
        """Automatically fix common annotation issues."""
        
        fixed_annotations = []
        
        for annotation in annotations:
            fixed_ann = self._fix_annotation(annotation, image_width, image_height)
            if fixed_ann:
                fixed_annotations.append(fixed_ann)
        
        return fixed_annotations
    
    def _fix_annotation(
        self,
        annotation: AnnotationShape,
        image_width: int,
        image_height: int
    ) -> Optional[AnnotationShape]:
        """Fix a single annotation."""
        
        # Create a copy to avoid modifying original
        import copy
        fixed_ann = copy.deepcopy(annotation)
        
        # Fix bounding box issues
        if fixed_ann.bbox:
            x, y, w, h = fixed_ann.bbox
            
            # Clamp to image bounds
            x = max(0, min(x, image_width - 1))
            y = max(0, min(y, image_height - 1))
            w = max(1, min(w, image_width - x))
            h = max(1, min(h, image_height - y))
            
            fixed_ann.bbox = [x, y, w, h]
            fixed_ann.area = w * h
        
        # Fix points outside bounds
        for point in fixed_ann.points:
            point.x = max(0, min(point.x, image_width))
            point.y = max(0, min(point.y, image_height))
        
        # Remove if too small
        if fixed_ann.area < 4:  # Minimum viable area
            return None
        
        # Fix confidence bounds
        fixed_ann.confidence = max(0.0, min(1.0, fixed_ann.confidence))
        
        return fixed_ann


class BatchAnnotationProcessor:
    """Processes multiple images for annotation assistance."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.completed_jobs: Dict[str, ProcessingJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent)
    
    def submit_job(
        self,
        image_paths: List[str],
        assistant: AnnotationAssistant,
        output_path: Optional[str] = None
    ) -> str:
        """Submit a batch annotation job."""
        
        job = ProcessingJob(
            input_images=image_paths,
            output_path=output_path,
            total_images=len(image_paths),
            config=self.config
        )
        
        self.active_jobs[job.job_id] = job
        
        # Start processing asynchronously
        asyncio.create_task(self._process_job(job, assistant))
        
        return job.job_id
    
    async def _process_job(
        self,
        job: ProcessingJob,
        assistant: AnnotationAssistant
    ) -> None:
        """Process a batch annotation job."""
        
        job.status = ProcessingStatus.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        try:
            # Process images in batches
            batch_size = self.config.batch_size
            image_batches = [
                job.input_images[i:i + batch_size]
                for i in range(0, len(job.input_images), batch_size)
            ]
            
            all_results = []
            
            for batch in image_batches:
                if job.status == ProcessingStatus.CANCELLED:
                    break
                
                batch_results = await self._process_batch(batch, assistant, job)
                all_results.extend(batch_results)
                
                # Update progress
                job.processed_images += len(batch_results)
                job.successful_images += len([r for r in batch_results if r["status"] == "completed"])
                job.failed_images += len([r for r in batch_results if r["status"] == "failed"])
            
            job.results = all_results
            
            # Save results if output path specified
            if job.output_path and job.successful_images > 0:
                await self._save_results(job)
            
            job.status = ProcessingStatus.COMPLETED
            
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.add_error(f"Job processing failed: {str(e)}")
        
        finally:
            job.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            # Move to completed jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
                self.completed_jobs[job.job_id] = job
    
    async def _process_batch(
        self,
        image_paths: List[str],
        assistant: AnnotationAssistant,
        job: ProcessingJob
    ) -> List[Dict[str, Any]]:
        """Process a batch of images."""
        
        # Create assistance requests
        requests = []
        for image_path in image_paths:
            try:
                # Get image dimensions (simplified - would use actual image reading)
                image_width, image_height = 640, 640  # Default dimensions
                
                request = AssistanceRequest(
                    image_path=image_path,
                    image_width=image_width,
                    image_height=image_height,
                    mode=AssistanceMode.DETECTION,  # Default mode
                    config=assistant.config
                )
                requests.append(request)
                
            except Exception as e:
                job.add_error(f"Failed to create request for {image_path}: {str(e)}")
        
        # Process requests with timeout and retry
        batch_results = []
        
        for request in requests:
            result_data = {"image_path": request.image_path, "status": "failed"}
            
            for attempt in range(self.config.max_retries + 1):
                try:
                    # Process with timeout
                    result = await asyncio.wait_for(
                        assistant.assist(request),
                        timeout=self.config.timeout_seconds
                    )
                    
                    # Validate and process result
                    processed_result = await self._process_result(result, job)
                    result_data.update(processed_result)
                    result_data["status"] = "completed"
                    break
                    
                except asyncio.TimeoutError:
                    error_msg = f"Timeout processing {request.image_path} (attempt {attempt + 1})"
                    job.add_warning(error_msg)
                    
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(self.config.retry_delay_seconds)
                    else:
                        result_data["error"] = "Timeout after retries"
                        
                except Exception as e:
                    error_msg = f"Error processing {request.image_path}: {str(e)}"
                    job.add_error(error_msg)
                    
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(self.config.retry_delay_seconds)
                    else:
                        result_data["error"] = str(e)
                        
                    if not self.config.continue_on_error:
                        raise
            
            batch_results.append(result_data)
        
        return batch_results
    
    async def _process_result(
        self,
        result: AssistanceResult,
        job: ProcessingJob
    ) -> Dict[str, Any]:
        """Process and validate an assistance result."""
        
        processed_data = {
            "annotations": result.annotations,
            "total_annotations": result.total_annotations,
            "processing_time_ms": result.processing_time_ms,
            "confidence_stats": result.confidence_stats,
            "quality_score": result.quality_score
        }
        
        # Quality checking if enabled
        if self.config.enable_validation and result.annotations:
            try:
                quality_checker = QualityChecker(self.config.validation_level)
                
                # Convert annotations to AnnotationShape objects if needed
                annotations = []
                for ann_data in result.annotations:
                    if isinstance(ann_data, dict):
                        # Convert dict to AnnotationShape
                        annotation = self._dict_to_annotation_shape(ann_data)
                        annotations.append(annotation)
                    else:
                        annotations.append(ann_data)
                
                # Get image dimensions from first annotation or use defaults
                image_width = 640  # Default
                image_height = 640  # Default
                
                quality_report = quality_checker.check_annotations(
                    annotations, image_width, image_height
                )
                processed_data["quality_report"] = quality_report
                
                # Auto-fix if enabled
                if self.config.auto_fix_issues:
                    fixed_annotations = quality_checker.auto_fix_annotations(
                        annotations, image_width, image_height
                    )
                    processed_data["fixed_annotations"] = [
                        ann.to_coco_format() for ann in fixed_annotations
                    ]
                    
                    if len(fixed_annotations) != len(annotations):
                        job.add_warning(f"Auto-fixed annotations: {len(annotations)} -> {len(fixed_annotations)}")
                
            except Exception as e:
                job.add_warning(f"Quality checking failed: {str(e)}")
        
        return processed_data
    
    def _dict_to_annotation_shape(self, ann_data: Dict[str, Any]) -> AnnotationShape:
        """Convert annotation dictionary to AnnotationShape."""
        from .tools import AnnotationShape, AnnotationType, AnnotationPoint
        
        annotation = AnnotationShape()
        annotation.category_id = ann_data.get("category_id", 0)
        annotation.category_name = ann_data.get("class_name", "")
        annotation.confidence = ann_data.get("confidence", 1.0)
        annotation.bbox = ann_data.get("bbox", [])
        annotation.area = ann_data.get("area", 0.0)
        annotation.is_crowd = bool(ann_data.get("iscrowd", 0))
        
        # Create points from bbox
        if annotation.bbox and len(annotation.bbox) >= 4:
            x, y, w, h = annotation.bbox
            annotation.points = [
                AnnotationPoint(x=x, y=y),
                AnnotationPoint(x=x+w, y=y+h)
            ]
        
        return annotation
    
    async def _save_results(self, job: ProcessingJob) -> None:
        """Save job results to output path."""
        
        if not job.output_path:
            return
        
        try:
            output_path = Path(job.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Aggregate all annotations
            all_annotations = []
            for result in job.results:
                if result["status"] == "completed":
                    annotations = result.get("annotations", [])
                    all_annotations.extend(annotations)
            
            # Convert to desired output format
            if self.config.output_format.lower() == "coco":
                output_data = {
                    "info": {
                        "description": f"Batch processing job {job.job_id}",
                        "version": "1.0",
                        "year": 2024,
                        "contributor": "Annotation Tools",
                        "date_created": job.created_at
                    },
                    "images": [],
                    "annotations": all_annotations,
                    "categories": []
                }
                
                # Add job metadata if enabled
                if self.config.include_metadata:
                    output_data["job_metadata"] = {
                        "job_id": job.job_id,
                        "total_images": job.total_images,
                        "successful_images": job.successful_images,
                        "failed_images": job.failed_images,
                        "processing_config": {
                            "max_concurrent": self.config.max_concurrent,
                            "validation_enabled": self.config.enable_validation,
                            "auto_fix_enabled": self.config.auto_fix_issues
                        }
                    }
                
                import json
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
            
            else:
                job.add_warning(f"Unsupported output format: {self.config.output_format}")
        
        except Exception as e:
            job.add_error(f"Failed to save results: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get status of a processing job."""
        return (self.active_jobs.get(job_id) or 
                self.completed_jobs.get(job_id))
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job."""
        job = self.active_jobs.get(job_id)
        if job and not job.is_complete:
            job.status = ProcessingStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            return True
        return False
    
    def list_active_jobs(self) -> List[ProcessingJob]:
        """List all active processing jobs."""
        return list(self.active_jobs.values())
    
    def list_completed_jobs(self) -> List[ProcessingJob]:
        """List completed processing jobs."""
        return list(self.completed_jobs.values())
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        all_jobs = list(self.active_jobs.values()) + list(self.completed_jobs.values())
        
        if not all_jobs:
            return {"total_jobs": 0}
        
        completed_jobs = [j for j in all_jobs if j.status == ProcessingStatus.COMPLETED]
        
        stats = {
            "total_jobs": len(all_jobs),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(completed_jobs),
            "failed_jobs": len([j for j in all_jobs if j.status == ProcessingStatus.FAILED]),
            "total_images_processed": sum(j.processed_images for j in all_jobs),
            "total_successful_annotations": sum(j.successful_images for j in all_jobs)
        }
        
        # Calculate averages for completed jobs
        if completed_jobs:
            processing_times = []
            for job in completed_jobs:
                if job.started_at and job.completed_at:
                    start_dt = datetime.fromisoformat(job.started_at.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(job.completed_at.replace("Z", "+00:00"))
                    processing_time = (end_dt - start_dt).total_seconds()
                    processing_times.append(processing_time)
            
            if processing_times:
                stats["average_job_time_seconds"] = sum(processing_times) / len(processing_times)
                stats["average_images_per_second"] = sum(
                    j.processed_images / (processing_times[i] if processing_times[i] > 0 else 1)
                    for i, j in enumerate(completed_jobs)
                ) / len(completed_jobs)
        
        return stats
    
    def cleanup_completed_jobs(self, older_than_hours: int = 24) -> int:
        """Clean up old completed jobs."""
        cutoff_time = time.time() - (older_than_hours * 3600)
        jobs_to_remove = []
        
        for job_id, job in self.completed_jobs.items():
            if job.completed_at:
                try:
                    completed_time = datetime.fromisoformat(
                        job.completed_at.replace("Z", "+00:00")
                    ).timestamp()
                    
                    if completed_time < cutoff_time:
                        jobs_to_remove.append(job_id)
                except ValueError:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.completed_jobs[job_id]
        
        return len(jobs_to_remove)
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        # Cancel all active jobs
        for job_id in list(self.active_jobs.keys()):
            self.cancel_job(job_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)


class AnnotationPipeline:
    """High-level annotation processing pipeline."""
    
    def __init__(self):
        self.processors: Dict[str, BatchAnnotationProcessor] = {}
        self.assistants: Dict[str, AnnotationAssistant] = {}
    
    def register_assistant(self, name: str, assistant: AnnotationAssistant) -> None:
        """Register an annotation assistant."""
        self.assistants[name] = assistant
    
    def create_processor(self, name: str, config: ProcessingConfig) -> BatchAnnotationProcessor:
        """Create and register a batch processor."""
        processor = BatchAnnotationProcessor(config)
        self.processors[name] = processor
        return processor
    
    def process_images(
        self,
        image_paths: List[str],
        assistant_name: str,
        processor_name: str = "default",
        output_path: Optional[str] = None
    ) -> str:
        """Process images using specified assistant and processor."""
        
        # Get or create processor
        if processor_name not in self.processors:
            config = ProcessingConfig()  # Use default config
            self.create_processor(processor_name, config)
        
        processor = self.processors[processor_name]
        
        # Get assistant
        if assistant_name not in self.assistants:
            raise ValueError(f"Assistant '{assistant_name}' not registered")
        
        assistant = self.assistants[assistant_name]
        
        # Submit job
        return processor.submit_job(image_paths, assistant, output_path)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status."""
        total_active_jobs = 0
        total_completed_jobs = 0
        
        processor_stats = {}
        
        for name, processor in self.processors.items():
            stats = processor.get_processing_stats()
            processor_stats[name] = stats
            total_active_jobs += stats.get("active_jobs", 0)
            total_completed_jobs += stats.get("completed_jobs", 0)
        
        return {
            "registered_assistants": len(self.assistants),
            "active_processors": len(self.processors),
            "total_active_jobs": total_active_jobs,
            "total_completed_jobs": total_completed_jobs,
            "processor_stats": processor_stats
        }
    
    def shutdown_all(self) -> None:
        """Shutdown all processors."""
        for processor in self.processors.values():
            processor.shutdown()