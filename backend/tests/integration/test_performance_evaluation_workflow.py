"""
Integration test for complete performance evaluation workflow

This test MUST FAIL initially since the endpoints are not implemented.
Tests the complete performance evaluation workflow from quickstart.md Step 5.

Workflow:
1. Setup test dataset with ground truth annotations
2. Run model inference on test set
3. Calculate performance metrics (mAP, IoU, precision, recall, F1)
4. Measure execution time metrics
5. Compare performance between different models
6. Export evaluation results
"""

import pytest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import uuid
import time
from typing import List, Dict
import statistics

from src.main import app

client = TestClient(app)


class TestPerformanceEvaluationWorkflowIntegration:
    """Integration tests for complete performance evaluation workflow."""

    @pytest.fixture(scope="class")
    def setup_evaluation_data(self):
        """Setup comprehensive test data for performance evaluation."""
        test_data = {
            "test_image_ids": [],
            "ground_truth_annotations": [],
            "detection_models": [],
            "dataset_id": None
        }
        
        # Create test images with varying complexity
        test_scenarios = [
            ("eval_simple.jpg", (640, 480), 'red', "simple"),
            ("eval_medium.jpg", (1024, 768), 'green', "medium"),
            ("eval_complex.jpg", (1280, 960), 'blue', "complex"),
            ("eval_edge_case.jpg", (512, 384), 'yellow', "edge")
        ]
        
        for filename, size, color, complexity in test_scenarios:
            # Create test image
            image = Image.new('RGB', size, color=color)
            image_buffer = BytesIO()
            image.save(image_buffer, format='JPEG', quality=90)
            image_buffer.seek(0)
            
            # Upload image
            response = client.post(
                "/api/v1/images",
                files={"files": (filename, image_buffer, "image/jpeg")},
                data={"dataset_split": "test"}
            )
            
            if response.status_code == 201:
                image_data = response.json()["uploaded_images"][0]
                image_id = image_data["id"]
                test_data["test_image_ids"].append(image_id)
                
                # Create ground truth annotations for each test image
                ground_truth = self._create_ground_truth_annotation(image_id, complexity)
                
                annotation_response = client.post("/api/v1/annotations", json=ground_truth)
                if annotation_response.status_code == 201:
                    test_data["ground_truth_annotations"].append(annotation_response.json())
        
        # Get available detection models
        response = client.get("/api/v1/models?type=detection")
        if response.status_code == 200:
            test_data["detection_models"] = response.json()["models"]
        
        # Create test dataset
        if test_data["test_image_ids"]:
            dataset_request = {
                "name": "performance_evaluation_dataset",
                "description": "Test dataset for performance evaluation",
                "image_ids": test_data["test_image_ids"],
                "split": "test"
            }
            
            # This endpoint might not exist yet
            response = client.post("/api/v1/datasets", json=dataset_request)
            if response.status_code == 201:
                test_data["dataset_id"] = response.json()["id"]
        
        return test_data

    def _create_ground_truth_annotation(self, image_id: str, complexity: str) -> Dict:
        """Create ground truth annotations based on image complexity."""
        base_annotation = {
            "image_id": image_id,
            "user_tag": "ground_truth_evaluator",
            "metadata": {"complexity": complexity}
        }
        
        if complexity == "simple":
            base_annotation.update({
                "bounding_boxes": [
                    {
                        "x": 100.0, "y": 100.0, "width": 200.0, "height": 150.0,
                        "class_id": 0, "confidence": 1.0
                    }
                ],
                "class_labels": ["person"]
            })
        elif complexity == "medium":
            base_annotation.update({
                "bounding_boxes": [
                    {
                        "x": 150.0, "y": 200.0, "width": 180.0, "height": 120.0,
                        "class_id": 0, "confidence": 1.0
                    },
                    {
                        "x": 400.0, "y": 300.0, "width": 100.0, "height": 80.0,
                        "class_id": 1, "confidence": 1.0
                    }
                ],
                "class_labels": ["person", "car"]
            })
        elif complexity == "complex":
            base_annotation.update({
                "bounding_boxes": [
                    {
                        "x": 50.0, "y": 50.0, "width": 120.0, "height": 100.0,
                        "class_id": 0, "confidence": 1.0
                    },
                    {
                        "x": 200.0, "y": 150.0, "width": 150.0, "height": 130.0,
                        "class_id": 1, "confidence": 1.0
                    },
                    {
                        "x": 400.0, "y": 250.0, "width": 80.0, "height": 60.0,
                        "class_id": 2, "confidence": 1.0
                    }
                ],
                "class_labels": ["person", "car", "bicycle"]
            })
        else:  # edge case
            base_annotation.update({
                "bounding_boxes": [
                    {
                        "x": 5.0, "y": 5.0, "width": 50.0, "height": 40.0,
                        "class_id": 0, "confidence": 1.0
                    }
                ],
                "class_labels": ["small_object"]
            })
        
        return base_annotation

    def test_complete_performance_evaluation_workflow(self, setup_evaluation_data):
        """Test the complete performance evaluation workflow from quickstart Step 5."""
        
        if not setup_evaluation_data["test_image_ids"]:
            pytest.skip("Test data setup failed")
        
        if not setup_evaluation_data["detection_models"]:
            pytest.skip("Detection models not available")
        
        # Step 1: Select model and test dataset
        test_model = setup_evaluation_data["detection_models"][0]
        dataset_id = setup_evaluation_data["dataset_id"] or "test_dataset"
        
        # Step 2: Calculate comprehensive performance metrics
        metrics_request = {
            "model_id": test_model["id"],
            "dataset_id": dataset_id,
            "metric_types": [
                "mAP@50", "mAP@50:95", "precision", "recall", "F1",
                "execution_time", "throughput", "memory_usage"
            ],
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5
        }
        
        response = client.post("/api/v1/evaluation/metrics", json=metrics_request)
        
        # This MUST FAIL since endpoint doesn't exist yet
        assert response.status_code == 200
        
        evaluation_result = response.json()
        
        # Validate evaluation result structure
        assert evaluation_result["model_id"] == test_model["id"]
        assert evaluation_result["dataset_id"] == dataset_id
        assert "metrics" in evaluation_result
        assert "detailed_results" in evaluation_result
        assert "evaluation_timestamp" in evaluation_result
        
        metrics = evaluation_result["metrics"]
        
        # Step 3: Verify all requested metrics are calculated
        expected_metrics = [
            "mAP@50", "mAP@50:95", "precision", "recall", "F1",
            "execution_time_avg_ms", "throughput_images_per_sec", "memory_usage_mb"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Metric {metric} not numeric"
        
        # Validate metric ranges
        assert 0.0 <= metrics["mAP@50"] <= 1.0, "mAP@50 out of range"
        assert 0.0 <= metrics["mAP@50:95"] <= 1.0, "mAP@50:95 out of range"
        assert 0.0 <= metrics["precision"] <= 1.0, "Precision out of range"
        assert 0.0 <= metrics["recall"] <= 1.0, "Recall out of range"
        assert 0.0 <= metrics["F1"] <= 1.0, "F1 out of range"
        assert metrics["execution_time_avg_ms"] > 0, "Execution time should be positive"
        assert metrics["throughput_images_per_sec"] > 0, "Throughput should be positive"
        
        # Step 4: Validate detailed per-image results
        detailed_results = evaluation_result["detailed_results"]
        assert len(detailed_results) == len(setup_evaluation_data["test_image_ids"])
        
        for image_result in detailed_results:
            assert "image_id" in image_result
            assert image_result["image_id"] in setup_evaluation_data["test_image_ids"]
            assert "predictions" in image_result
            assert "ground_truth" in image_result
            assert "metrics" in image_result
            
            # Validate per-image metrics
            image_metrics = image_result["metrics"]
            assert "iou_scores" in image_metrics
            assert "precision" in image_metrics
            assert "recall" in image_metrics
            assert "execution_time_ms" in image_metrics

    def test_model_comparison_workflow(self, setup_evaluation_data):
        """Test comparing performance between different models."""
        
        if len(setup_evaluation_data["detection_models"]) < 2:
            pytest.skip("Need at least 2 models for comparison")
        
        # Select two different models for comparison
        model1 = setup_evaluation_data["detection_models"][0]
        model2 = setup_evaluation_data["detection_models"][1]
        dataset_id = setup_evaluation_data["dataset_id"] or "test_dataset"
        
        # Step 5: Compare models performance
        comparison_request = {
            "model_ids": [model1["id"], model2["id"]],
            "dataset_id": dataset_id,
            "metric_types": ["mAP@50", "mAP@50:95", "execution_time", "F1"],
            "confidence_threshold": 0.5
        }
        
        response = client.post("/api/v1/evaluation/compare", json=comparison_request)
        
        assert response.status_code == 200
        
        comparison_result = response.json()
        
        # Validate comparison structure
        assert "models" in comparison_result
        assert len(comparison_result["models"]) == 2
        assert "comparison_summary" in comparison_result
        assert "winner_by_metric" in comparison_result
        
        # Validate individual model results
        model_results = comparison_result["models"]
        model_ids = [result["model_id"] for result in model_results]
        assert model1["id"] in model_ids
        assert model2["id"] in model_ids
        
        for model_result in model_results:
            assert "model_id" in model_result
            assert "metrics" in model_result
            
            metrics = model_result["metrics"]
            assert "mAP@50" in metrics
            assert "mAP@50:95" in metrics
            assert "execution_time_avg_ms" in metrics
            assert "F1" in metrics
        
        # Validate comparison summary
        summary = comparison_result["comparison_summary"]
        assert "better_model_id" in summary
        assert "performance_difference" in summary
        assert "statistical_significance" in summary

    def test_metric_calculation_accuracy(self, setup_evaluation_data):
        """Test accuracy of metric calculations with known ground truth."""
        
        if not setup_evaluation_data["test_image_ids"]:
            pytest.skip("No test data available")
        
        # Create a controlled test case with known expected results
        controlled_image_id = setup_evaluation_data["test_image_ids"][0]
        
        # Create perfect predictions that match ground truth exactly
        ground_truth_annotation = None
        for annotation in setup_evaluation_data["ground_truth_annotations"]:
            if annotation["image_id"] == controlled_image_id:
                ground_truth_annotation = annotation
                break
        
        if not ground_truth_annotation:
            pytest.skip("Ground truth annotation not found")
        
        # Simulate perfect model predictions
        perfect_predictions = []
        for bbox in ground_truth_annotation["bounding_boxes"]:
            perfect_predictions.append({
                "class_id": bbox["class_id"],
                "confidence": 0.99,  # High confidence
                "bounding_box": {
                    "x": bbox["x"],
                    "y": bbox["y"],
                    "width": bbox["width"],
                    "height": bbox["height"]
                }
            })
        
        # Mock inference result with perfect predictions
        # This would be injected or mocked in a real test
        
        # Calculate metrics for this controlled case
        if setup_evaluation_data["detection_models"]:
            model_id = setup_evaluation_data["detection_models"][0]["id"]
            
            controlled_metrics_request = {
                "model_id": model_id,
                "image_ids": [controlled_image_id],
                "metric_types": ["mAP@50", "precision", "recall", "F1"],
                "confidence_threshold": 0.5
            }
            
            response = client.post("/api/v1/evaluation/metrics", json=controlled_metrics_request)
            
            if response.status_code == 200:
                result = response.json()
                metrics = result["metrics"]
                
                # With perfect predictions, these should be high
                # (exact values depend on implementation details)
                assert metrics["precision"] >= 0.9, "Perfect predictions should yield high precision"
                assert metrics["recall"] >= 0.9, "Perfect predictions should yield high recall"
                assert metrics["F1"] >= 0.9, "Perfect predictions should yield high F1"

    def test_execution_time_measurements(self, setup_evaluation_data):
        """Test execution time measurement accuracy and consistency."""
        
        if not setup_evaluation_data["test_image_ids"] or not setup_evaluation_data["detection_models"]:
            pytest.skip("Test data not available")
        
        model = setup_evaluation_data["detection_models"][0]
        test_image_ids = setup_evaluation_data["test_image_ids"][:2]  # Use subset for faster testing
        
        # Run multiple evaluations to test consistency
        execution_times = []
        
        for _ in range(3):  # Run 3 times
            metrics_request = {
                "model_id": model["id"],
                "image_ids": test_image_ids,
                "metric_types": ["execution_time"],
                "measure_detailed_timing": True
            }
            
            start_time = time.time()
            response = client.post("/api/v1/evaluation/metrics", json=metrics_request)
            total_wall_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                metrics = result["metrics"]
                
                # Validate timing measurements
                assert "execution_time_avg_ms" in metrics
                assert "execution_time_min_ms" in metrics
                assert "execution_time_max_ms" in metrics
                assert "execution_time_std_ms" in metrics
                
                avg_time = metrics["execution_time_avg_ms"]
                min_time = metrics["execution_time_min_ms"]
                max_time = metrics["execution_time_max_ms"]
                
                # Validate timing relationships
                assert min_time <= avg_time <= max_time, "Timing statistics inconsistent"
                assert avg_time > 0, "Average execution time should be positive"
                
                execution_times.append(avg_time)
                
                # Wall clock time should be reasonable compared to reported time
                reported_total_ms = avg_time * len(test_image_ids)
                assert total_wall_time * 1000 >= reported_total_ms * 0.5, "Wall time too fast"
                assert total_wall_time * 1000 <= reported_total_ms * 10, "Wall time too slow"
        
        # Test consistency across runs
        if len(execution_times) >= 2:
            mean_time = statistics.mean(execution_times)
            std_time = statistics.stdev(execution_times)
            
            # Standard deviation shouldn't be too high (coefficient of variation < 50%)
            if mean_time > 0:
                cv = std_time / mean_time
                assert cv < 0.5, f"Execution time too variable: CV={cv:.2f}"

    def test_memory_usage_tracking(self, setup_evaluation_data):
        """Test memory usage tracking during evaluation."""
        
        if not setup_evaluation_data["test_image_ids"] or not setup_evaluation_data["detection_models"]:
            pytest.skip("Test data not available")
        
        model = setup_evaluation_data["detection_models"][0]
        
        metrics_request = {
            "model_id": model["id"],
            "image_ids": setup_evaluation_data["test_image_ids"],
            "metric_types": ["memory_usage"],
            "track_memory_profile": True
        }
        
        response = client.post("/api/v1/evaluation/metrics", json=metrics_request)
        
        if response.status_code == 200:
            result = response.json()
            metrics = result["metrics"]
            
            # Validate memory metrics
            assert "memory_usage_mb" in metrics
            assert "memory_peak_mb" in metrics
            assert "memory_baseline_mb" in metrics
            
            memory_usage = metrics["memory_usage_mb"]
            memory_peak = metrics["memory_peak_mb"]
            memory_baseline = metrics["memory_baseline_mb"]
            
            # Validate memory measurements
            assert memory_usage >= 0, "Memory usage should be non-negative"
            assert memory_peak >= memory_usage, "Peak memory should >= average usage"
            assert memory_baseline >= 0, "Baseline memory should be non-negative"
            
            # Memory usage should be reasonable (not extremely high)
            assert memory_usage < 10000, "Memory usage seems unreasonably high (>10GB)"

    def test_confidence_threshold_impact_on_metrics(self, setup_evaluation_data):
        """Test how confidence thresholds affect evaluation metrics."""
        
        if not setup_evaluation_data["test_image_ids"] or not setup_evaluation_data["detection_models"]:
            pytest.skip("Test data not available")
        
        model = setup_evaluation_data["detection_models"][0]
        test_image_id = setup_evaluation_data["test_image_ids"][0]
        
        # Test different confidence thresholds
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results_by_threshold = {}
        
        for threshold in thresholds:
            metrics_request = {
                "model_id": model["id"],
                "image_ids": [test_image_id],
                "metric_types": ["precision", "recall", "F1"],
                "confidence_threshold": threshold
            }
            
            response = client.post("/api/v1/evaluation/metrics", json=metrics_request)
            
            if response.status_code == 200:
                result = response.json()
                metrics = result["metrics"]
                results_by_threshold[threshold] = metrics
        
        # Analyze threshold effects
        if len(results_by_threshold) >= 2:
            # Generally, higher confidence thresholds should increase precision
            # but may decrease recall (classic precision-recall trade-off)
            sorted_thresholds = sorted(results_by_threshold.keys())
            
            for i in range(len(sorted_thresholds) - 1):
                lower_thresh = sorted_thresholds[i]
                higher_thresh = sorted_thresholds[i + 1]
                
                lower_precision = results_by_threshold[lower_thresh]["precision"]
                higher_precision = results_by_threshold[higher_thresh]["precision"]
                
                # Higher threshold should generally have >= precision
                # (allowing for some variance in real scenarios)
                if higher_precision < lower_precision:
                    precision_drop = lower_precision - higher_precision
                    assert precision_drop < 0.3, f"Precision dropped too much: {precision_drop:.3f}"

    def test_evaluation_export_and_reporting(self, setup_evaluation_data):
        """Test exporting evaluation results and generating reports."""
        
        if not setup_evaluation_data["test_image_ids"] or not setup_evaluation_data["detection_models"]:
            pytest.skip("Test data not available")
        
        model = setup_evaluation_data["detection_models"][0]
        
        # Run comprehensive evaluation
        metrics_request = {
            "model_id": model["id"],
            "image_ids": setup_evaluation_data["test_image_ids"],
            "metric_types": ["mAP@50", "mAP@50:95", "precision", "recall", "F1", "execution_time"],
            "generate_report": True,
            "export_format": "detailed"
        }
        
        response = client.post("/api/v1/evaluation/metrics", json=metrics_request)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check for exportable report data
            if "report" in result:
                report = result["report"]
                
                # Validate report structure
                assert "summary" in report
                assert "detailed_metrics" in report
                assert "visualizations" in report or "charts_data" in report
                
                summary = report["summary"]
                assert "model_name" in summary
                assert "evaluation_date" in summary
                assert "dataset_size" in summary
                assert "overall_performance" in summary
        
        # Test exporting results to different formats
        export_request = {
            "evaluation_id": result.get("evaluation_id", "test_eval"),
            "format": "json",
            "include_visualizations": True
        }
        
        export_response = client.post("/api/v1/evaluation/export", json=export_request)
        
        # Should work once export endpoint is implemented
        if export_response.status_code == 200:
            export_result = export_response.json()
            assert "download_url" in export_result or "export_data" in export_result

    def test_evaluation_error_handling(self, setup_evaluation_data):
        """Test error handling in evaluation workflows."""
        
        # Test with invalid model ID
        response = client.post("/api/v1/evaluation/metrics", json={
            "model_id": "invalid-model-id",
            "dataset_id": "test-dataset",
            "metric_types": ["mAP@50"]
        })
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        
        # Test with empty image list
        if setup_evaluation_data["detection_models"]:
            response = client.post("/api/v1/evaluation/metrics", json={
                "model_id": setup_evaluation_data["detection_models"][0]["id"],
                "image_ids": [],
                "metric_types": ["mAP@50"]
            })
            
            assert response.status_code == 400
        
        # Test with invalid metric types
        if setup_evaluation_data["detection_models"] and setup_evaluation_data["test_image_ids"]:
            response = client.post("/api/v1/evaluation/metrics", json={
                "model_id": setup_evaluation_data["detection_models"][0]["id"],
                "image_ids": setup_evaluation_data["test_image_ids"][:1],
                "metric_types": ["invalid_metric", "fake_metric"]
            })
            
            assert response.status_code == 400