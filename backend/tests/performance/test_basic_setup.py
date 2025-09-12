"""
Basic setup tests to validate performance test infrastructure.
"""

import pytest
import psutil
import sys
import asyncio
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.lib.ml_models.models import ModelRegistry, ModelType, ModelVariant
from src.lib.ml_models.registry import get_model_registry
from src.lib.inference_engine.engine import InferenceEngine


class TestPerformanceSetup:
    """Test basic performance testing infrastructure setup."""
    
    def test_psutil_available(self):
        """Test that psutil is available for memory monitoring."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        assert memory_info.rss > 0
        assert hasattr(process, 'memory_info')
        
    def test_model_registry_basic_setup(self):
        """Test basic model registry setup."""
        registry = ModelRegistry()
        
        assert registry is not None
        assert hasattr(registry, 'register_model')
        assert hasattr(registry, 'list_models')
        assert hasattr(registry, 'load_model')
        
    def test_global_registry_initialization(self):
        """Test global model registry initialization."""
        registry = get_model_registry()
        
        assert registry is not None
        
        # Should have default models registered
        models = registry.list_models()
        assert len(models) > 0
        
        # Check for expected model types
        model_types = {model.model_type for model in models}
        expected_types = {ModelType.YOLO11, ModelType.YOLO12, ModelType.RT_DETR, ModelType.SAM2}
        
        assert expected_types.issubset(model_types)
        
    def test_inference_engine_basic_setup(self):
        """Test basic inference engine setup."""
        engine = InferenceEngine()
        
        assert engine is not None
        assert hasattr(engine, 'single_inference')
        assert hasattr(engine, 'start_batch_inference')
        assert engine.max_workers > 0
        
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test that async functionality works correctly."""
        
        async def dummy_async_function():
            await asyncio.sleep(0.001)  # Very short sleep
            return "async_works"
        
        result = await dummy_async_function()
        assert result == "async_works"
        
    def test_model_variants_available(self):
        """Test that all expected model variants are available."""
        registry = get_model_registry()
        models = registry.list_models()
        
        # Create mapping of models by type and variant
        models_by_type = {}
        for model in models:
            if model.model_type not in models_by_type:
                models_by_type[model.model_type] = set()
            models_by_type[model.model_type].add(model.variant)
        
        # Check YOLO variants
        expected_yolo_variants = {
            ModelVariant.YOLO_NANO,
            ModelVariant.YOLO_SMALL,
            ModelVariant.YOLO_MEDIUM,
            ModelVariant.YOLO_LARGE,
            ModelVariant.YOLO_EXTRA_LARGE
        }
        
        if ModelType.YOLO11 in models_by_type:
            yolo11_variants = models_by_type[ModelType.YOLO11]
            assert expected_yolo_variants.issubset(yolo11_variants), \
                f"Missing YOLO11 variants: {expected_yolo_variants - yolo11_variants}"
        
        # Check RT-DETR variants
        expected_rtdetr_variants = {
            ModelVariant.RT_DETR_R18,
            ModelVariant.RT_DETR_R34,
            ModelVariant.RT_DETR_R50,
            ModelVariant.RT_DETR_R101
        }
        
        if ModelType.RT_DETR in models_by_type:
            rtdetr_variants = models_by_type[ModelType.RT_DETR]
            assert expected_rtdetr_variants.issubset(rtdetr_variants), \
                f"Missing RT-DETR variants: {expected_rtdetr_variants - rtdetr_variants}"
        
        # Check SAM2 variants
        expected_sam2_variants = {
            ModelVariant.SAM2_TINY,
            ModelVariant.SAM2_SMALL,
            ModelVariant.SAM2_BASE_PLUS,
            ModelVariant.SAM2_LARGE
        }
        
        if ModelType.SAM2 in models_by_type:
            sam2_variants = models_by_type[ModelType.SAM2]
            assert expected_sam2_variants.issubset(sam2_variants), \
                f"Missing SAM2 variants: {expected_sam2_variants - sam2_variants}"
                
    def test_performance_markers_configured(self):
        """Test that performance pytest markers are properly configured."""
        import pytest
        
        # This test validates that our custom markers work
        # The actual marker validation happens in conftest.py
        assert hasattr(pytest.mark, 'performance')
        assert hasattr(pytest.mark, 'memory')
        assert hasattr(pytest.mark, 'benchmark')
        assert hasattr(pytest.mark, 'regression')


# Mark this entire test class as performance tests
pytestmark = pytest.mark.performance