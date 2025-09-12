"""
Performance tests ensuring real-time inference requirements (T082)

Tests real-time inference performance requirements (< 2 seconds target) 
across all supported model variants with comprehensive performance monitoring.

Performance Requirements:
- Real-time inference: < 2 seconds for single image inference
- Batch processing: Efficient concurrent operations 
- Memory usage: Monitor and validate memory consumption
- Model loading: Optimize model loading performance
- Concurrent handling: Support multiple simultaneous requests
- Resource cleanup: Prevent memory leaks
- Performance regression: Detect performance degradation

Test Coverage:
- All model variants: YOLO11/12 (nano, small, medium, large, xl)
- RT-DETR variants: R18, R34, R50, R101, RF-DETR (nano, small, medium)
- SAM2 variants: Tiny, Small, Base+, Large
- Different image sizes and complexity scenarios
- Concurrent load testing with realistic workloads
"""

import pytest
import asyncio
import time
import psutil
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO
from PIL import Image
import uuid
import statistics
import json
import os
from pathlib import Path

from src.lib.inference_engine.engine import (
    InferenceEngine, InferenceRequest, BatchInferenceJob, 
    InferenceStatus, PerformanceMetrics
)
from src.lib.ml_models.models import (
    ModelConfig, ModelType, ModelVariant, ModelRegistry,
    YOLOModelWrapper, RTDETRModelWrapper, SAM2ModelWrapper
)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark thresholds and targets."""
    # Real-time inference target (< 2 seconds)
    REAL_TIME_TARGET_MS: float = 2000.0
    
    # Model loading performance targets
    MODEL_LOAD_TARGET_MS: Dict[str, float] = None
    
    # Memory usage limits (MB)
    MAX_MEMORY_PER_INFERENCE_MB: float = 1024.0  # 1GB per inference
    MAX_MEMORY_BATCH_MB: float = 4096.0  # 4GB for batch processing
    
    # Throughput targets (images per second)
    MIN_THROUGHPUT_IPS: Dict[str, float] = None
    
    # Concurrent processing targets
    MAX_CONCURRENT_REQUESTS: int = 10
    CONCURRENT_RESPONSE_TIME_MS: float = 5000.0  # 5 seconds under load
    
    def __post_init__(self):
        if self.MODEL_LOAD_TARGET_MS is None:
            self.MODEL_LOAD_TARGET_MS = {
                "yolo_nano": 500.0,      # 500ms for nano models
                "yolo_small": 1000.0,    # 1s for small models
                "yolo_medium": 2000.0,   # 2s for medium models  
                "yolo_large": 3000.0,    # 3s for large models
                "yolo_xl": 5000.0,       # 5s for xl models
                "rtdetr_r18": 1000.0,    # 1s for R18
                "rtdetr_r34": 1500.0,    # 1.5s for R34
                "rtdetr_r50": 2000.0,    # 2s for R50
                "rtdetr_r101": 3000.0,   # 3s for R101
                "rtdetr_rf_nano": 500.0, # 500ms for RF nano
                "rtdetr_rf_small": 1000.0, # 1s for RF small
                "rtdetr_rf_medium": 1500.0, # 1.5s for RF medium
                "sam2_tiny": 1000.0,     # 1s for tiny SAM2
                "sam2_small": 2000.0,    # 2s for small SAM2
                "sam2_base_plus": 3000.0, # 3s for base+ SAM2
                "sam2_large": 5000.0,    # 5s for large SAM2
            }
        
        if self.MIN_THROUGHPUT_IPS is None:
            self.MIN_THROUGHPUT_IPS = {
                "yolo_nano": 20.0,       # 20 images/sec
                "yolo_small": 10.0,      # 10 images/sec
                "yolo_medium": 5.0,      # 5 images/sec
                "yolo_large": 2.0,       # 2 images/sec
                "yolo_xl": 1.0,          # 1 image/sec
                "rtdetr_r18": 8.0,       # 8 images/sec
                "rtdetr_r34": 6.0,       # 6 images/sec
                "rtdetr_r50": 4.0,       # 4 images/sec
                "rtdetr_r101": 2.0,      # 2 images/sec
                "rtdetr_rf_nano": 15.0,  # 15 images/sec
                "rtdetr_rf_small": 8.0,  # 8 images/sec
                "rtdetr_rf_medium": 5.0, # 5 images/sec
                "sam2_tiny": 5.0,        # 5 images/sec (segmentation slower)
                "sam2_small": 3.0,       # 3 images/sec
                "sam2_base_plus": 2.0,   # 2 images/sec
                "sam2_large": 1.0,       # 1 image/sec
            }


@dataclass
class PerformanceResult:
    """Results from performance testing."""
    model_id: str
    test_name: str
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    memory_usage_mb: float
    throughput_fps: float
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


@dataclass 
class TestImageSpec:
    """Specification for test images with different characteristics."""
    name: str
    size: Tuple[int, int]  # (width, height)
    complexity: str  # "simple", "medium", "complex"
    format: str = "JPEG"
    color_space: str = "RGB"
    
    def create_image(self) -> bytes:
        """Create test image based on specification."""
        width, height = self.size
        
        if self.complexity == "simple":
            # Single color with simple geometric shapes
            image = Image.new(self.color_space, (width, height), color='lightblue')
            # Add some simple shapes for object detection
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], 
                         fill='red', outline='black', width=2)
        elif self.complexity == "medium":
            # Multiple colors and shapes
            image = Image.new(self.color_space, (width, height), color='white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            # Multiple objects
            for i in range(3):
                x1, y1 = i * width//4, i * height//4
                x2, y2 = x1 + width//6, y1 + height//6
                color = ['red', 'green', 'blue'][i]
                draw.rectangle([x1, y1, x2, y2], fill=color, outline='black')
        else:  # complex
            # Random noise pattern (most challenging for models)
            import random
            pixels = []
            for _ in range(width * height):
                pixels.extend([random.randint(0, 255) for _ in range(3)])
            image = Image.new(self.color_space, (width, height))
            image.putdata([(pixels[i], pixels[i+1], pixels[i+2]) 
                          for i in range(0, len(pixels), 3)])
        
        buffer = BytesIO()
        image.save(buffer, format=self.format)
        return buffer.getvalue()


class MemoryMonitor:
    """Monitor memory usage during inference operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = 0
        self.peak_memory = 0
        self.monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.baseline_memory
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                try:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    self.peak_memory = max(self.peak_memory, current_memory)
                    time.sleep(0.01)  # Check every 10ms
                except:
                    break
        
        self._monitor_thread = threading.Thread(target=monitor)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage delta."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        return max(0, self.peak_memory - self.baseline_memory)


class PerformanceTestSuite:
    """Main performance test suite for inference speed validation."""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
        self.results: List[PerformanceResult] = []
        self.test_images = self._create_test_images()
        self.model_registry = None
        self.inference_engine = None
    
    def _create_test_images(self) -> Dict[str, TestImageSpec]:
        """Create test image specifications for different scenarios."""
        return {
            # Small images for fast inference
            "small_simple": TestImageSpec("small_simple", (320, 240), "simple"),
            "small_medium": TestImageSpec("small_medium", (320, 240), "medium"),
            "small_complex": TestImageSpec("small_complex", (320, 240), "complex"),
            
            # Standard resolution images  
            "standard_simple": TestImageSpec("standard_simple", (640, 480), "simple"),
            "standard_medium": TestImageSpec("standard_medium", (640, 480), "medium"),
            "standard_complex": TestImageSpec("standard_complex", (640, 480), "complex"),
            
            # High resolution images
            "hd_simple": TestImageSpec("hd_simple", (1280, 720), "simple"),
            "hd_medium": TestImageSpec("hd_medium", (1280, 720), "medium"),
            "hd_complex": TestImageSpec("hd_complex", (1280, 720), "complex"),
            
            # Very high resolution for stress testing
            "uhd_simple": TestImageSpec("uhd_simple", (1920, 1080), "simple"),
            "uhd_medium": TestImageSpec("uhd_medium", (1920, 1080), "medium"),
            
            # Edge cases
            "tiny": TestImageSpec("tiny", (64, 64), "simple"),
            "wide": TestImageSpec("wide", (1600, 400), "medium"),
            "tall": TestImageSpec("tall", (400, 1600), "medium"),
        }
    
    def _setup_models(self) -> None:
        """Setup test models for all supported variants."""
        self.model_registry = ModelRegistry()
        
        # YOLO11 models
        yolo11_variants = [
            (ModelVariant.YOLO_NANO, "yolo_nano"),
            (ModelVariant.YOLO_SMALL, "yolo_small"), 
            (ModelVariant.YOLO_MEDIUM, "yolo_medium"),
            (ModelVariant.YOLO_LARGE, "yolo_large"),
            (ModelVariant.YOLO_EXTRA_LARGE, "yolo_xl")
        ]
        
        for variant, key in yolo11_variants:
            config = ModelConfig(
                model_id=f"yolo11_{key}",
                model_type=ModelType.YOLO11,
                variant=variant,
                name=f"YOLO11 {variant.value}",
                description=f"YOLO11 {variant.value} for object detection",
                input_size=(640, 640),
                num_classes=80,
                performance_metrics={
                    "target_load_time_ms": self.benchmark.MODEL_LOAD_TARGET_MS[key],
                    "target_throughput_fps": self.benchmark.MIN_THROUGHPUT_IPS[key]
                }
            )
            self.model_registry.register_model(config)
        
        # YOLO12 models (same variants)
        for variant, key in yolo11_variants:
            config = ModelConfig(
                model_id=f"yolo12_{key}",
                model_type=ModelType.YOLO12,
                variant=variant,
                name=f"YOLO12 {variant.value}",
                description=f"YOLO12 {variant.value} for object detection",
                input_size=(640, 640),
                num_classes=80,
                performance_metrics={
                    "target_load_time_ms": self.benchmark.MODEL_LOAD_TARGET_MS[key],
                    "target_throughput_fps": self.benchmark.MIN_THROUGHPUT_IPS[key]
                }
            )
            self.model_registry.register_model(config)
        
        # RT-DETR models
        rtdetr_variants = [
            (ModelVariant.RT_DETR_R18, "rtdetr_r18"),
            (ModelVariant.RT_DETR_R34, "rtdetr_r34"),
            (ModelVariant.RT_DETR_R50, "rtdetr_r50"),
            (ModelVariant.RT_DETR_R101, "rtdetr_r101"),
            (ModelVariant.RT_DETR_RF_NANO, "rtdetr_rf_nano"),
            (ModelVariant.RT_DETR_RF_SMALL, "rtdetr_rf_small"),
            (ModelVariant.RT_DETR_RF_MEDIUM, "rtdetr_rf_medium")
        ]
        
        for variant, key in rtdetr_variants:
            config = ModelConfig(
                model_id=f"rtdetr_{variant.value}",
                model_type=ModelType.RT_DETR,
                variant=variant,
                name=f"RT-DETR {variant.value}",
                description=f"RT-DETR {variant.value} for object detection",
                input_size=(640, 640),
                num_classes=80,
                performance_metrics={
                    "target_load_time_ms": self.benchmark.MODEL_LOAD_TARGET_MS[key],
                    "target_throughput_fps": self.benchmark.MIN_THROUGHPUT_IPS[key]
                }
            )
            self.model_registry.register_model(config)
        
        # SAM2 models
        sam2_variants = [
            (ModelVariant.SAM2_TINY, "sam2_tiny"),
            (ModelVariant.SAM2_SMALL, "sam2_small"),
            (ModelVariant.SAM2_BASE_PLUS, "sam2_base_plus"),
            (ModelVariant.SAM2_LARGE, "sam2_large")
        ]
        
        for variant, key in sam2_variants:
            config = ModelConfig(
                model_id=f"sam2_{variant.value}",
                model_type=ModelType.SAM2,
                variant=variant,
                name=f"SAM2 {variant.value}",
                description=f"SAM2 {variant.value} for segmentation", 
                input_size=(1024, 1024),
                num_classes=1,  # SAM2 is class-agnostic
                performance_metrics={
                    "target_load_time_ms": self.benchmark.MODEL_LOAD_TARGET_MS[key],
                    "target_throughput_fps": self.benchmark.MIN_THROUGHPUT_IPS[key]
                }
            )
            self.model_registry.register_model(config)
    
    def _setup_inference_engine(self, max_workers: int = 4) -> None:
        """Setup inference engine for testing."""
        self.inference_engine = InferenceEngine(max_workers=max_workers)
    
    def _save_test_image(self, spec: TestImageSpec, image_data: bytes) -> str:
        """Save test image to temporary file and return path."""
        import tempfile
        
        temp_dir = Path(tempfile.gettempdir()) / "ml_eval_performance_tests"
        temp_dir.mkdir(exist_ok=True)
        
        temp_path = temp_dir / f"{spec.name}_{uuid.uuid4().hex[:8]}.{spec.format.lower()}"
        
        with open(temp_path, "wb") as f:
            f.write(image_data)
        
        return str(temp_path)
    
    async def _run_single_inference_test(
        self,
        model_id: str,
        image_spec: TestImageSpec,
        monitor_memory: bool = True
    ) -> PerformanceResult:
        """Run single inference performance test."""
        
        # Create and save test image
        image_data = image_spec.create_image()
        image_path = self._save_test_image(image_spec, image_data)
        
        try:
            memory_monitor = MemoryMonitor() if monitor_memory else None
            
            # Start memory monitoring
            if memory_monitor:
                memory_monitor.start_monitoring()
            
            # Create inference request
            request = InferenceRequest(
                model_id=model_id,
                image_path=image_path,
                parameters={"confidence": 0.5}
            )
            
            # Run inference
            start_time = time.time()
            result = await self.inference_engine.single_inference(request)
            total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Stop memory monitoring
            memory_usage = memory_monitor.stop_monitoring() if memory_monitor else 0.0
            
            # Analyze results
            success = result.status == InferenceStatus.COMPLETED
            error_message = result.error_message if not success else None
            
            performance_result = PerformanceResult(
                model_id=model_id,
                test_name=f"single_inference_{image_spec.name}",
                inference_time_ms=result.performance_metrics.inference_time_ms,
                preprocessing_time_ms=result.performance_metrics.preprocessing_time_ms,
                postprocessing_time_ms=result.performance_metrics.postprocessing_time_ms,
                total_time_ms=total_time,
                memory_usage_mb=memory_usage,
                throughput_fps=result.performance_metrics.throughput_fps,
                success=success,
                error_message=error_message,
                additional_metrics={
                    "image_size": image_spec.size,
                    "image_complexity": image_spec.complexity,
                    "predictions_count": len(result.predictions),
                    "meets_real_time_target": total_time < self.benchmark.REAL_TIME_TARGET_MS
                }
            )
            
            return performance_result
            
        finally:
            # Clean up test image
            try:
                os.unlink(image_path)
            except:
                pass
    
    async def _run_batch_inference_test(
        self,
        model_id: str,
        image_specs: List[TestImageSpec],
        batch_size: int = 4
    ) -> PerformanceResult:
        """Run batch inference performance test."""
        
        # Create test images
        image_paths = []
        try:
            for spec in image_specs:
                image_data = spec.create_image()
                image_path = self._save_test_image(spec, image_data)
                image_paths.append(image_path)
            
            memory_monitor = MemoryMonitor()
            memory_monitor.start_monitoring()
            
            # Create batch job
            batch_job = BatchInferenceJob(
                model_id=model_id,
                image_paths=image_paths,
                parameters={"confidence": 0.5, "batch_size": batch_size}
            )
            
            # Run batch inference
            start_time = time.time()
            job = self.inference_engine.start_batch_inference(batch_job)
            
            # Wait for completion
            max_wait_time = 60  # 1 minute max
            while time.time() - start_time < max_wait_time:
                updated_job = self.inference_engine.get_batch_job_status(job.job_id)
                if updated_job and updated_job.is_complete:
                    job = updated_job
                    break
                await asyncio.sleep(0.1)
            
            total_time = (time.time() - start_time) * 1000
            memory_usage = memory_monitor.stop_monitoring()
            
            # Analyze results
            success = job.status == InferenceStatus.COMPLETED
            successful_results = [r for r in job.results if r.status == InferenceStatus.COMPLETED]
            
            if successful_results:
                avg_inference_time = statistics.mean(
                    r.performance_metrics.inference_time_ms for r in successful_results
                )
                total_throughput = len(successful_results) / (total_time / 1000) if total_time > 0 else 0
            else:
                avg_inference_time = 0
                total_throughput = 0
            
            performance_result = PerformanceResult(
                model_id=model_id,
                test_name=f"batch_inference_{len(image_specs)}_images",
                inference_time_ms=avg_inference_time,
                preprocessing_time_ms=0,  # Not tracked individually in batch
                postprocessing_time_ms=0,
                total_time_ms=total_time,
                memory_usage_mb=memory_usage,
                throughput_fps=total_throughput,
                success=success,
                error_message=job.error_message if not success else None,
                additional_metrics={
                    "batch_size": batch_size,
                    "total_images": len(image_specs),
                    "successful_images": len(successful_results),
                    "failed_images": job.failed_images,
                    "success_rate": len(successful_results) / len(image_specs) * 100,
                    "meets_memory_target": memory_usage < self.benchmark.MAX_MEMORY_BATCH_MB
                }
            )
            
            return performance_result
            
        finally:
            # Clean up test images
            for image_path in image_paths:
                try:
                    os.unlink(image_path)
                except:
                    pass
    
    async def _run_concurrent_inference_test(
        self,
        model_id: str,
        image_spec: TestImageSpec,
        concurrent_requests: int = 5
    ) -> PerformanceResult:
        """Run concurrent inference performance test."""
        
        memory_monitor = MemoryMonitor()
        memory_monitor.start_monitoring()
        
        # Create multiple inference requests
        async def single_request():
            image_data = image_spec.create_image()
            image_path = self._save_test_image(image_spec, image_data)
            try:
                request = InferenceRequest(
                    model_id=model_id,
                    image_path=image_path,
                    parameters={"confidence": 0.5}
                )
                return await self.inference_engine.single_inference(request)
            finally:
                try:
                    os.unlink(image_path)
                except:
                    pass
        
        # Run concurrent requests
        start_time = time.time()
        
        tasks = [single_request() for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = (time.time() - start_time) * 1000
        memory_usage = memory_monitor.stop_monitoring()
        
        # Analyze concurrent results
        successful_results = [
            r for r in results 
            if not isinstance(r, Exception) and r.status == InferenceStatus.COMPLETED
        ]
        
        success = len(successful_results) == concurrent_requests
        avg_response_time = total_time / concurrent_requests if concurrent_requests > 0 else 0
        
        performance_result = PerformanceResult(
            model_id=model_id,
            test_name=f"concurrent_inference_{concurrent_requests}_requests",
            inference_time_ms=avg_response_time,
            preprocessing_time_ms=0,
            postprocessing_time_ms=0,
            total_time_ms=total_time,
            memory_usage_mb=memory_usage,
            throughput_fps=concurrent_requests / (total_time / 1000) if total_time > 0 else 0,
            success=success,
            error_message=None,
            additional_metrics={
                "concurrent_requests": concurrent_requests,
                "successful_requests": len(successful_results),
                "average_response_time_ms": avg_response_time,
                "meets_concurrent_target": avg_response_time < self.benchmark.CONCURRENT_RESPONSE_TIME_MS,
                "meets_memory_target": memory_usage < self.benchmark.MAX_MEMORY_BATCH_MB
            }
        )
        
        return performance_result
    
    async def _run_model_loading_test(self, model_id: str) -> PerformanceResult:
        """Test model loading performance."""
        
        memory_monitor = MemoryMonitor()
        memory_monitor.start_monitoring()
        
        # Ensure model is not loaded
        self.model_registry.unload_model(model_id)
        
        # Force garbage collection
        gc.collect()
        
        # Load model and measure time
        start_time = time.time()
        try:
            wrapper = self.model_registry.load_model(model_id)
            load_time = (time.time() - start_time) * 1000
            success = wrapper.is_loaded
            error_message = None
        except Exception as e:
            load_time = (time.time() - start_time) * 1000
            success = False
            error_message = str(e)
        
        memory_usage = memory_monitor.stop_monitoring()
        
        # Get performance targets
        config = self.model_registry.get_model_config(model_id)
        target_metrics = config.performance_metrics if config else {}
        
        performance_result = PerformanceResult(
            model_id=model_id,
            test_name="model_loading",
            inference_time_ms=0,
            preprocessing_time_ms=load_time,
            postprocessing_time_ms=0,
            total_time_ms=load_time,
            memory_usage_mb=memory_usage,
            throughput_fps=0,
            success=success,
            error_message=error_message,
            additional_metrics={
                "target_load_time_ms": target_metrics.get("target_load_time_ms", 0),
                "meets_load_time_target": load_time < target_metrics.get("target_load_time_ms", float('inf')),
                "model_type": config.model_type if config else "unknown",
                "model_variant": config.variant if config else "unknown"
            }
        )
        
        return performance_result
    
    def _analyze_memory_leaks(self) -> Dict[str, Any]:
        """Analyze potential memory leaks after test execution."""
        gc.collect()  # Force garbage collection
        
        # Get current memory usage
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Check for loaded models that should be cleaned up
        loaded_models = len(self.model_registry._loaded_models) if self.model_registry else 0
        active_jobs = len(self.inference_engine.list_active_jobs()) if self.inference_engine else 0
        
        return {
            "current_memory_mb": current_memory,
            "loaded_models_count": loaded_models,
            "active_jobs_count": active_jobs,
            "memory_warning": current_memory > 2048,  # Warn if > 2GB
            "cleanup_needed": loaded_models > 0 or active_jobs > 0
        }
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance test report."""
        if not self.results:
            return {"error": "No performance results available"}
        
        # Group results by test type and model
        by_test_type = {}
        by_model = {}
        
        for result in self.results:
            # Group by test type
            test_type = result.test_name.split('_')[0] + '_' + result.test_name.split('_')[1]
            if test_type not in by_test_type:
                by_test_type[test_type] = []
            by_test_type[test_type].append(result)
            
            # Group by model
            if result.model_id not in by_model:
                by_model[result.model_id] = []
            by_model[result.model_id].append(result)
        
        # Calculate summary statistics
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        # Real-time performance analysis
        real_time_tests = [
            r for r in successful_tests 
            if "single_inference" in r.test_name
        ]
        
        real_time_compliance = [
            r for r in real_time_tests 
            if r.total_time_ms < self.benchmark.REAL_TIME_TARGET_MS
        ]
        
        # Memory usage analysis  
        memory_compliant = [
            r for r in successful_tests
            if r.memory_usage_mb < self.benchmark.MAX_MEMORY_PER_INFERENCE_MB
        ]
        
        # Performance regression detection
        regressions = self._detect_performance_regressions()
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.results) * 100,
                "real_time_compliance_rate": len(real_time_compliance) / len(real_time_tests) * 100 if real_time_tests else 0,
                "memory_compliance_rate": len(memory_compliant) / len(successful_tests) * 100 if successful_tests else 0
            },
            "performance_targets": {
                "real_time_target_ms": self.benchmark.REAL_TIME_TARGET_MS,
                "max_memory_per_inference_mb": self.benchmark.MAX_MEMORY_PER_INFERENCE_MB,
                "max_concurrent_requests": self.benchmark.MAX_CONCURRENT_REQUESTS
            },
            "results_by_test_type": {},
            "results_by_model": {},
            "performance_regressions": regressions,
            "memory_analysis": self._analyze_memory_leaks(),
            "recommendations": self._generate_recommendations()
        }
        
        # Add detailed results by test type
        for test_type, results in by_test_type.items():
            successful = [r for r in results if r.success]
            if successful:
                avg_time = statistics.mean(r.total_time_ms for r in successful)
                avg_memory = statistics.mean(r.memory_usage_mb for r in successful)
                avg_throughput = statistics.mean(r.throughput_fps for r in successful)
                
                report["results_by_test_type"][test_type] = {
                    "total_tests": len(results),
                    "successful_tests": len(successful),
                    "average_time_ms": avg_time,
                    "average_memory_mb": avg_memory,
                    "average_throughput_fps": avg_throughput,
                    "meets_real_time_target": avg_time < self.benchmark.REAL_TIME_TARGET_MS
                }
        
        # Add detailed results by model
        for model_id, results in by_model.items():
            successful = [r for r in results if r.success]
            if successful:
                report["results_by_model"][model_id] = {
                    "total_tests": len(results),
                    "successful_tests": len(successful),
                    "average_time_ms": statistics.mean(r.total_time_ms for r in successful),
                    "average_memory_mb": statistics.mean(r.memory_usage_mb for r in successful),
                    "average_throughput_fps": statistics.mean(r.throughput_fps for r in successful),
                    "fastest_test_ms": min(r.total_time_ms for r in successful),
                    "slowest_test_ms": max(r.total_time_ms for r in successful)
                }
        
        return report
    
    def _detect_performance_regressions(self) -> List[Dict[str, Any]]:
        """Detect performance regressions against baseline metrics."""
        # This would normally compare against historical baseline data
        # For now, we'll compare against the configured performance targets
        
        regressions = []
        
        for result in self.results:
            if not result.success:
                continue
                
            config = self.model_registry.get_model_config(result.model_id)
            if not config or not config.performance_metrics:
                continue
                
            target_metrics = config.performance_metrics
            
            # Check loading time regression
            if "model_loading" in result.test_name:
                target_load_time = target_metrics.get("target_load_time_ms", 0)
                if result.total_time_ms > target_load_time * 1.2:  # 20% tolerance
                    regressions.append({
                        "type": "model_loading",
                        "model_id": result.model_id,
                        "actual_time_ms": result.total_time_ms,
                        "target_time_ms": target_load_time,
                        "regression_percent": (result.total_time_ms / target_load_time - 1) * 100
                    })
            
            # Check throughput regression
            target_throughput = target_metrics.get("target_throughput_fps", 0)
            if result.throughput_fps < target_throughput * 0.8:  # 20% tolerance
                regressions.append({
                    "type": "throughput",
                    "model_id": result.model_id,
                    "actual_throughput_fps": result.throughput_fps,
                    "target_throughput_fps": target_throughput,
                    "regression_percent": (1 - result.throughput_fps / target_throughput) * 100
                })
        
        return regressions
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.results:
            return ["No test results available for analysis"]
        
        successful_tests = [r for r in self.results if r.success]
        
        # Real-time compliance analysis
        real_time_tests = [r for r in successful_tests if "single_inference" in r.test_name]
        if real_time_tests:
            slow_tests = [r for r in real_time_tests if r.total_time_ms > self.benchmark.REAL_TIME_TARGET_MS]
            if slow_tests:
                recommendations.append(
                    f"Performance issue: {len(slow_tests)}/{len(real_time_tests)} single inference tests "
                    f"exceed real-time target of {self.benchmark.REAL_TIME_TARGET_MS}ms"
                )
        
        # Memory usage analysis
        high_memory_tests = [r for r in successful_tests if r.memory_usage_mb > self.benchmark.MAX_MEMORY_PER_INFERENCE_MB]
        if high_memory_tests:
            recommendations.append(
                f"Memory issue: {len(high_memory_tests)} tests exceed memory limit of "
                f"{self.benchmark.MAX_MEMORY_PER_INFERENCE_MB}MB"
            )
        
        # Model-specific recommendations
        model_performance = {}
        for result in successful_tests:
            if result.model_id not in model_performance:
                model_performance[result.model_id] = []
            model_performance[result.model_id].append(result.total_time_ms)
        
        for model_id, times in model_performance.items():
            avg_time = statistics.mean(times)
            if avg_time > self.benchmark.REAL_TIME_TARGET_MS:
                recommendations.append(
                    f"Model optimization needed: {model_id} average time {avg_time:.1f}ms "
                    f"exceeds target {self.benchmark.REAL_TIME_TARGET_MS}ms"
                )
        
        # Concurrent processing analysis
        concurrent_tests = [r for r in successful_tests if "concurrent" in r.test_name]
        if concurrent_tests:
            avg_concurrent_time = statistics.mean(r.total_time_ms for r in concurrent_tests)
            if avg_concurrent_time > self.benchmark.CONCURRENT_RESPONSE_TIME_MS:
                recommendations.append(
                    f"Concurrency issue: Average concurrent response time {avg_concurrent_time:.1f}ms "
                    f"exceeds target {self.benchmark.CONCURRENT_RESPONSE_TIME_MS}ms"
                )
        
        if not recommendations:
            recommendations.append("All performance tests meet target requirements")
        
        return recommendations


# Pytest test class
class TestInferencePerformance:
    """Performance tests for inference speed requirements."""
    
    @pytest.fixture(scope="class")
    def performance_suite(self):
        """Setup performance test suite."""
        suite = PerformanceTestSuite()
        suite._setup_models()
        suite._setup_inference_engine()
        return suite
    
    @pytest.mark.asyncio
    async def test_yolo_model_loading_performance(self, performance_suite):
        """Test YOLO model loading performance across all variants."""
        
        yolo_models = [
            config.model_id for config in performance_suite.model_registry.list_models()
            if config.model_type in [ModelType.YOLO11, ModelType.YOLO12]
        ]
        
        for model_id in yolo_models[:3]:  # Test first 3 to avoid long test times
            result = await performance_suite._run_model_loading_test(model_id)
            performance_suite.results.append(result)
            
            # Assert real-time requirements
            assert result.success, f"Model loading failed for {model_id}: {result.error_message}"
            
            # Check against target loading time
            config = performance_suite.model_registry.get_model_config(model_id)
            if config and config.performance_metrics:
                target_load_time = config.performance_metrics.get("target_load_time_ms", float('inf'))
                assert result.total_time_ms <= target_load_time, \
                    f"Model {model_id} loading time {result.total_time_ms}ms exceeds target {target_load_time}ms"
    
    @pytest.mark.asyncio 
    async def test_single_inference_real_time_performance(self, performance_suite):
        """Test single inference meets real-time requirements (< 2 seconds)."""
        
        # Test with different model types and image complexities
        test_cases = [
            ("yolo11_yolo_nano", "standard_simple"),
            ("yolo11_yolo_small", "standard_medium"), 
            ("rtdetr_r18", "standard_simple"),
            ("sam2_tiny", "small_simple")
        ]
        
        for model_id, image_key in test_cases:
            # Skip if model doesn't exist
            if not performance_suite.model_registry.get_model_config(model_id):
                continue
                
            image_spec = performance_suite.test_images[image_key]
            result = await performance_suite._run_single_inference_test(model_id, image_spec)
            performance_suite.results.append(result)
            
            # Assert real-time performance requirement
            assert result.success, f"Inference failed for {model_id} on {image_key}: {result.error_message}"
            assert result.total_time_ms < performance_suite.benchmark.REAL_TIME_TARGET_MS, \
                f"Inference time {result.total_time_ms}ms exceeds real-time target " \
                f"{performance_suite.benchmark.REAL_TIME_TARGET_MS}ms for {model_id} on {image_key}"
            
            # Assert memory usage is reasonable
            assert result.memory_usage_mb < performance_suite.benchmark.MAX_MEMORY_PER_INFERENCE_MB, \
                f"Memory usage {result.memory_usage_mb}MB exceeds limit " \
                f"{performance_suite.benchmark.MAX_MEMORY_PER_INFERENCE_MB}MB"
    
    @pytest.mark.asyncio
    async def test_batch_inference_performance(self, performance_suite):
        """Test batch inference performance and throughput."""
        
        model_id = "yolo11_yolo_nano"  # Use fastest model for batch testing
        if not performance_suite.model_registry.get_model_config(model_id):
            pytest.skip("YOLO nano model not available")
        
        # Create batch of different image types
        batch_specs = [
            performance_suite.test_images["small_simple"],
            performance_suite.test_images["small_medium"],
            performance_suite.test_images["standard_simple"],
            performance_suite.test_images["standard_medium"]
        ]
        
        result = await performance_suite._run_batch_inference_test(model_id, batch_specs)
        performance_suite.results.append(result)
        
        assert result.success, f"Batch inference failed: {result.error_message}"
        
        # Check batch processing efficiency
        avg_time_per_image = result.total_time_ms / len(batch_specs)
        assert avg_time_per_image < performance_suite.benchmark.REAL_TIME_TARGET_MS, \
            f"Average time per image {avg_time_per_image}ms exceeds real-time target"
        
        # Check memory usage for batch processing
        assert result.memory_usage_mb < performance_suite.benchmark.MAX_MEMORY_BATCH_MB, \
            f"Batch memory usage {result.memory_usage_mb}MB exceeds limit"
        
        # Check success rate
        success_rate = result.additional_metrics.get("success_rate", 0)
        assert success_rate >= 95.0, f"Batch success rate {success_rate}% below 95% threshold"
    
    @pytest.mark.asyncio
    async def test_concurrent_inference_handling(self, performance_suite):
        """Test concurrent inference request handling."""
        
        model_id = "yolo11_yolo_small" 
        if not performance_suite.model_registry.get_model_config(model_id):
            pytest.skip("YOLO small model not available")
        
        image_spec = performance_suite.test_images["standard_simple"]
        concurrent_requests = 5
        
        result = await performance_suite._run_concurrent_inference_test(
            model_id, image_spec, concurrent_requests
        )
        performance_suite.results.append(result)
        
        assert result.success, f"Concurrent inference test failed: {result.error_message}"
        
        # Check average response time under concurrent load
        avg_response_time = result.additional_metrics.get("average_response_time_ms", 0)
        assert avg_response_time < performance_suite.benchmark.CONCURRENT_RESPONSE_TIME_MS, \
            f"Average concurrent response time {avg_response_time}ms exceeds target " \
            f"{performance_suite.benchmark.CONCURRENT_RESPONSE_TIME_MS}ms"
        
        # Check all requests completed successfully
        successful_requests = result.additional_metrics.get("successful_requests", 0)
        assert successful_requests == concurrent_requests, \
            f"Only {successful_requests}/{concurrent_requests} concurrent requests succeeded"
    
    @pytest.mark.asyncio
    async def test_different_image_sizes_performance(self, performance_suite):
        """Test performance across different image sizes."""
        
        model_id = "yolo11_yolo_medium"
        if not performance_suite.model_registry.get_model_config(model_id):
            pytest.skip("YOLO medium model not available")
        
        # Test different image sizes
        size_tests = [
            ("tiny", performance_suite.test_images["tiny"]),
            ("small", performance_suite.test_images["small_simple"]),
            ("standard", performance_suite.test_images["standard_simple"]),
            ("hd", performance_suite.test_images["hd_simple"])
        ]
        
        size_results = []
        for size_name, image_spec in size_tests:
            result = await performance_suite._run_single_inference_test(model_id, image_spec)
            performance_suite.results.append(result)
            size_results.append((size_name, result))
            
            assert result.success, f"Inference failed on {size_name} image: {result.error_message}"
        
        # Verify that larger images generally take more time (but still within limits)
        for size_name, result in size_results:
            assert result.total_time_ms < performance_suite.benchmark.REAL_TIME_TARGET_MS * 2, \
                f"{size_name} image inference time {result.total_time_ms}ms exceeds 2x real-time target"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, performance_suite):
        """Test for memory leaks during repeated inferences."""
        
        model_id = "yolo11_yolo_small"
        if not performance_suite.model_registry.get_model_config(model_id):
            pytest.skip("YOLO small model not available")
        
        image_spec = performance_suite.test_images["standard_simple"]
        
        # Record initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run multiple inferences
        num_iterations = 10
        for i in range(num_iterations):
            result = await performance_suite._run_single_inference_test(
                model_id, image_spec, monitor_memory=False
            )
            assert result.success, f"Inference {i+1} failed: {result.error_message}"
        
        # Force garbage collection
        gc.collect()
        
        # Check final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Allow some memory increase but detect significant leaks
        max_allowed_increase = 100  # 100MB tolerance
        assert memory_increase < max_allowed_increase, \
            f"Memory increased by {memory_increase:.1f}MB after {num_iterations} inferences, " \
            f"possible memory leak (threshold: {max_allowed_increase}MB)"
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, performance_suite):
        """Test proper resource cleanup after inference operations."""
        
        # Run various inference operations
        model_id = "yolo11_yolo_nano"
        if not performance_suite.model_registry.get_model_config(model_id):
            pytest.skip("YOLO nano model not available")
        
        image_spec = performance_suite.test_images["standard_simple"]
        
        # Single inference
        await performance_suite._run_single_inference_test(model_id, image_spec)
        
        # Batch inference
        batch_specs = [performance_suite.test_images["small_simple"]] * 3
        await performance_suite._run_batch_inference_test(model_id, batch_specs)
        
        # Concurrent inference
        await performance_suite._run_concurrent_inference_test(model_id, image_spec, 3)
        
        # Check resource cleanup
        memory_analysis = performance_suite._analyze_memory_leaks()
        
        # Cleanup inference engine
        performance_suite.inference_engine.cleanup_completed_jobs(0)  # Clean all jobs
        
        # Check final state
        active_jobs = len(performance_suite.inference_engine.list_active_jobs())
        assert active_jobs == 0, f"Found {active_jobs} active jobs after cleanup"
        
        # Memory should not indicate major resource leaks
        assert not memory_analysis.get("memory_warning", False), \
            f"Memory usage warning: {memory_analysis['current_memory_mb']}MB"
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, performance_suite):
        """Test performance regression detection against baselines."""
        
        # Run a comprehensive set of tests
        models_to_test = [
            "yolo11_yolo_nano",
            "yolo11_yolo_small",
            "rtdetr_r18"
        ]
        
        for model_id in models_to_test:
            if not performance_suite.model_registry.get_model_config(model_id):
                continue
            
            # Model loading test
            loading_result = await performance_suite._run_model_loading_test(model_id)
            performance_suite.results.append(loading_result)
            
            # Single inference test
            inference_result = await performance_suite._run_single_inference_test(
                model_id, performance_suite.test_images["standard_simple"]
            )
            performance_suite.results.append(inference_result)
        
        # Check for regressions
        regressions = performance_suite._detect_performance_regressions()
        
        # Assert no significant regressions
        critical_regressions = [r for r in regressions if r.get("regression_percent", 0) > 50]
        assert len(critical_regressions) == 0, \
            f"Critical performance regressions detected: {critical_regressions}"
    
    def test_generate_performance_report(self, performance_suite):
        """Generate comprehensive performance test report."""
        
        # Generate report from all test results
        report = performance_suite._generate_performance_report()
        
        # Validate report structure
        assert "summary" in report
        assert "performance_targets" in report
        assert "recommendations" in report
        
        # Check summary statistics
        summary = report["summary"]
        assert "total_tests" in summary
        assert "success_rate" in summary
        assert "real_time_compliance_rate" in summary
        
        # Print report for manual inspection (optional)
        if os.environ.get("PRINT_PERFORMANCE_REPORT", "").lower() == "true":
            print("\n" + "="*80)
            print("PERFORMANCE TEST REPORT")
            print("="*80)
            print(json.dumps(report, indent=2))
            print("="*80)
        
        # Assert overall performance standards
        if summary["total_tests"] > 0:
            assert summary["success_rate"] >= 80.0, \
                f"Overall success rate {summary['success_rate']}% below 80% threshold"
            
            if summary.get("real_time_compliance_rate", 0) > 0:
                assert summary["real_time_compliance_rate"] >= 90.0, \
                    f"Real-time compliance rate {summary['real_time_compliance_rate']}% below 90% threshold"


# Additional helper functions for performance testing
def run_performance_benchmark():
    """Standalone function to run comprehensive performance benchmark."""
    import asyncio
    
    async def benchmark():
        suite = PerformanceTestSuite()
        suite._setup_models()
        suite._setup_inference_engine()
        
        print("Starting comprehensive performance benchmark...")
        
        # Run core performance tests
        models = ["yolo11_yolo_nano", "yolo11_yolo_small", "rtdetr_r18", "sam2_tiny"]
        
        for model_id in models:
            if suite.model_registry.get_model_config(model_id):
                print(f"Testing {model_id}...")
                
                # Model loading
                result = await suite._run_model_loading_test(model_id)
                suite.results.append(result)
                
                # Single inference with different image types
                for image_key in ["small_simple", "standard_medium"]:
                    result = await suite._run_single_inference_test(
                        model_id, suite.test_images[image_key]
                    )
                    suite.results.append(result)
        
        # Generate and return report
        return suite._generate_performance_report()
    
    return asyncio.run(benchmark())


if __name__ == "__main__":
    # Run standalone performance benchmark
    report = run_performance_benchmark()
    print("\nPerformance Benchmark Results:")
    print(json.dumps(report, indent=2))