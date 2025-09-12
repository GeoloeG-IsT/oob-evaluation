"""
Integration test for complete model inference workflow

This test MUST FAIL initially since the endpoints are not implemented.
Tests the complete model inference workflow from quickstart.md Step 4.

Workflow:
1. Select object detection model (YOLO11)
2. Run single image inference in real-time
3. View predictions overlaid on image
4. Initiate batch inference on multiple images
5. Monitor batch processing progress in real-time
6. Handle inference errors and edge cases
"""

import pytest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import uuid
import time
from typing import List, Dict
import concurrent.futures

from src.main import app

client = TestClient(app)


class TestModelInferenceWorkflowIntegration:
    """Integration tests for complete model inference workflow."""

    @pytest.fixture(scope="class")
    def setup_test_data(self):
        """Setup test images and models for inference testing."""
        test_data = {
            "image_ids": [],
            "detection_models": [],
            "segmentation_models": []
        }
        
        # Create and upload test images with different characteristics
        test_images = [
            ("inference_test_1.jpg", (640, 480), 'red'),
            ("inference_test_2.jpg", (1024, 768), 'green'),
            ("inference_test_3.jpg", (512, 384), 'blue'),
            ("inference_batch_1.jpg", (800, 600), 'yellow'),
            ("inference_batch_2.jpg", (960, 720), 'purple')
        ]
        
        for filename, size, color in test_images:
            image = Image.new('RGB', size, color=color)
            image_buffer = BytesIO()
            image.save(image_buffer, format='JPEG')
            image_buffer.seek(0)
            
            response = client.post(
                "/api/v1/images",
                files={"files": (filename, image_buffer, "image/jpeg")},
                data={"dataset_split": "test"}
            )
            
            if response.status_code == 201:
                image_data = response.json()["uploaded_images"][0]
                test_data["image_ids"].append(image_data["id"])
        
        # Get available models for inference
        response = client.get("/api/v1/models")
        if response.status_code == 200:
            models = response.json()["models"]
            for model in models:
                if model["type"] == "detection":
                    test_data["detection_models"].append(model)
                elif model["type"] == "segmentation":
                    test_data["segmentation_models"].append(model)
        
        return test_data

    def test_complete_model_inference_workflow(self, setup_test_data):
        """Test the complete model inference workflow from quickstart Step 4."""
        
        if not setup_test_data["image_ids"]:
            pytest.skip("Image upload not implemented yet")
        
        if not setup_test_data["detection_models"]:
            pytest.skip("Detection models not available")
        
        image_id = setup_test_data["image_ids"][0]
        yolo_model = None
        
        # Step 1: Select YOLO11 model for object detection
        for model in setup_test_data["detection_models"]:
            if "yolo11" in model["name"].lower() or "yolo" in model["name"].lower():
                yolo_model = model
                break
        
        if not yolo_model:
            pytest.skip("YOLO model not found")
        
        # Step 2: Run single inference in real-time
        start_time = time.time()
        
        inference_request = {
            "image_id": image_id,
            "model_id": yolo_model["id"],
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "max_detections": 100
        }
        
        response = client.post("/api/v1/inference/single", json=inference_request)
        
        # This MUST FAIL since endpoint doesn't exist yet
        assert response.status_code == 200
        
        inference_time = time.time() - start_time
        
        # Verify real-time performance (should complete in < 2 seconds)
        assert inference_time < 2.0, f"Inference took {inference_time:.2f}s, should be < 2s"
        
        inference_result = response.json()
        
        # Step 3: Validate prediction structure and overlay data
        assert inference_result["image_id"] == image_id
        assert inference_result["model_id"] == yolo_model["id"]
        assert "predictions" in inference_result
        assert "execution_time_ms" in inference_result
        assert "model_version" in inference_result
        
        predictions = inference_result["predictions"]
        
        # Validate prediction format for overlay display
        for prediction in predictions:
            assert "class_id" in prediction
            assert "confidence" in prediction
            assert prediction["confidence"] >= 0.5  # Above threshold
            assert "bounding_box" in prediction
            
            bbox = prediction["bounding_box"]
            assert "x" in bbox and "y" in bbox
            assert "width" in bbox and "height" in bbox
            assert bbox["width"] > 0 and bbox["height"] > 0

    def test_batch_inference_workflow(self, setup_test_data):
        """Test batch inference workflow with progress monitoring."""
        
        if len(setup_test_data["image_ids"]) < 3:
            pytest.skip("Not enough test images for batch processing")
        
        if not setup_test_data["detection_models"]:
            pytest.skip("Detection models not available")
        
        model = setup_test_data["detection_models"][0]
        batch_image_ids = setup_test_data["image_ids"][:3]
        
        # Step 4: Initiate batch inference
        batch_request = {
            "image_ids": batch_image_ids,
            "model_id": model["id"],
            "confidence_threshold": 0.4,
            "iou_threshold": 0.45,
            "batch_size": 2,
            "priority": "normal"
        }
        
        response = client.post("/api/v1/inference/batch", json=batch_request)
        
        assert response.status_code == 202  # Accepted for processing
        
        batch_response = response.json()
        assert "job_id" in batch_response
        assert "status" in batch_response
        assert batch_response["status"] == "queued"
        assert "total_images" in batch_response
        assert batch_response["total_images"] == len(batch_image_ids)
        
        job_id = batch_response["job_id"]
        
        # Step 5: Monitor batch processing progress in real-time
        max_wait_time = 30  # seconds
        start_time = time.time()
        final_status = None
        
        while time.time() - start_time < max_wait_time:
            response = client.get(f"/api/v1/inference/jobs/{job_id}")
            assert response.status_code == 200
            
            job_status = response.json()
            
            # Validate progress structure
            assert job_status["id"] == job_id
            assert "status" in job_status
            assert "processed_count" in job_status
            assert "total_count" in job_status
            assert "progress_percentage" in job_status
            assert "estimated_completion_time" in job_status
            
            current_status = job_status["status"]
            
            if current_status in ["completed", "failed"]:
                final_status = current_status
                break
            elif current_status == "processing":
                # Verify progress updates
                assert job_status["processed_count"] <= job_status["total_count"]
                assert 0 <= job_status["progress_percentage"] <= 100
            
            time.sleep(1)  # Wait before next progress check
        
        # Verify batch completion
        assert final_status == "completed", f"Batch job did not complete successfully: {final_status}"
        
        # Get final results
        response = client.get(f"/api/v1/inference/jobs/{job_id}")
        final_job = response.json()
        
        assert "results" in final_job
        assert len(final_job["results"]) == len(batch_image_ids)
        
        # Validate each result
        for result in final_job["results"]:
            assert result["image_id"] in batch_image_ids
            assert "predictions" in result
            assert "execution_time_ms" in result
            assert result["status"] == "completed"

    def test_different_model_types_inference(self, setup_test_data):
        """Test inference with different model types (detection vs segmentation)."""
        
        if not setup_test_data["image_ids"]:
            pytest.skip("No test images available")
        
        image_id = setup_test_data["image_ids"][0]
        
        # Test detection model inference
        if setup_test_data["detection_models"]:
            detection_model = setup_test_data["detection_models"][0]
            
            detection_request = {
                "image_id": image_id,
                "model_id": detection_model["id"],
                "confidence_threshold": 0.5
            }
            
            response = client.post("/api/v1/inference/single", json=detection_request)
            
            if response.status_code == 200:
                result = response.json()
                assert "predictions" in result
                
                # Detection models should return bounding boxes
                for prediction in result["predictions"]:
                    assert "bounding_box" in prediction
                    assert "class_id" in prediction
                    assert "confidence" in prediction
        
        # Test segmentation model inference
        if setup_test_data["segmentation_models"]:
            segmentation_model = setup_test_data["segmentation_models"][0]
            
            segmentation_request = {
                "image_id": image_id,
                "model_id": segmentation_model["id"],
                "confidence_threshold": 0.5
            }
            
            response = client.post("/api/v1/inference/single", json=segmentation_request)
            
            if response.status_code == 200:
                result = response.json()
                assert "predictions" in result
                
                # Segmentation models should return masks/polygons
                for prediction in result["predictions"]:
                    has_mask = "mask" in prediction
                    has_polygon = "polygon" in prediction
                    assert has_mask or has_polygon, "Segmentation prediction missing mask/polygon"

    def test_inference_parameter_variations(self, setup_test_data):
        """Test inference with different parameter configurations."""
        
        if not setup_test_data["image_ids"] or not setup_test_data["detection_models"]:
            pytest.skip("Test data not available")
        
        image_id = setup_test_data["image_ids"][0]
        model = setup_test_data["detection_models"][0]
        
        # Test different confidence thresholds
        confidence_thresholds = [0.3, 0.5, 0.8]
        results = {}
        
        for threshold in confidence_thresholds:
            request = {
                "image_id": image_id,
                "model_id": model["id"],
                "confidence_threshold": threshold,
                "max_detections": 50
            }
            
            response = client.post("/api/v1/inference/single", json=request)
            
            if response.status_code == 200:
                result = response.json()
                prediction_count = len(result["predictions"])
                results[threshold] = prediction_count
                
                # All predictions should meet confidence threshold
                for prediction in result["predictions"]:
                    assert prediction["confidence"] >= threshold
        
        # Higher confidence thresholds should generally yield fewer detections
        if len(results) >= 2:
            sorted_thresholds = sorted(results.keys())
            for i in range(len(sorted_thresholds) - 1):
                low_thresh = sorted_thresholds[i]
                high_thresh = sorted_thresholds[i + 1]
                assert results[high_thresh] <= results[low_thresh], \
                    f"Higher threshold {high_thresh} yielded more detections than {low_thresh}"

    def test_inference_error_handling(self, setup_test_data):
        """Test error handling in inference workflow."""
        
        if not setup_test_data["image_ids"]:
            pytest.skip("No test images available")
        
        image_id = setup_test_data["image_ids"][0]
        
        # Test invalid model ID
        response = client.post("/api/v1/inference/single", json={
            "image_id": image_id,
            "model_id": "invalid-model-id",
            "confidence_threshold": 0.5
        })
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        
        # Test invalid image ID
        if setup_test_data["detection_models"]:
            response = client.post("/api/v1/inference/single", json={
                "image_id": "invalid-image-id",
                "model_id": setup_test_data["detection_models"][0]["id"],
                "confidence_threshold": 0.5
            })
            
            assert response.status_code == 400
        
        # Test invalid confidence threshold
        if setup_test_data["detection_models"]:
            response = client.post("/api/v1/inference/single", json={
                "image_id": image_id,
                "model_id": setup_test_data["detection_models"][0]["id"],
                "confidence_threshold": 1.5  # Invalid - should be <= 1.0
            })
            
            assert response.status_code == 400

    def test_concurrent_inference_requests(self, setup_test_data):
        """Test handling of concurrent inference requests."""
        
        if len(setup_test_data["image_ids"]) < 2 or not setup_test_data["detection_models"]:
            pytest.skip("Insufficient test data for concurrent testing")
        
        model = setup_test_data["detection_models"][0]
        
        def run_inference(image_id):
            """Helper function for concurrent inference."""
            return client.post("/api/v1/inference/single", json={
                "image_id": image_id,
                "model_id": model["id"],
                "confidence_threshold": 0.5
            })
        
        # Run concurrent inferences
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for image_id in setup_test_data["image_ids"][:3]:
                future = executor.submit(run_inference, image_id)
                futures.append((future, image_id))
            
            results = []
            for future, image_id in futures:
                response = future.result()
                results.append((response, image_id))
        
        # All concurrent requests should succeed
        for response, image_id in results:
            assert response.status_code == 200
            result_data = response.json()
            assert result_data["image_id"] == image_id

    def test_inference_with_custom_model_parameters(self, setup_test_data):
        """Test inference with model-specific custom parameters."""
        
        if not setup_test_data["image_ids"] or not setup_test_data["detection_models"]:
            pytest.skip("Test data not available")
        
        image_id = setup_test_data["image_ids"][0]
        
        # Test YOLO-specific parameters
        for model in setup_test_data["detection_models"]:
            if "yolo" in model["name"].lower():
                yolo_request = {
                    "image_id": image_id,
                    "model_id": model["id"],
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45,
                    "model_parameters": {
                        "agnostic_nms": True,
                        "multi_label": True,
                        "classes": [0, 1, 2],  # Specific class IDs
                        "max_det": 50,
                        "augment": False
                    }
                }
                
                response = client.post("/api/v1/inference/single", json=yolo_request)
                
                if response.status_code == 200:
                    result = response.json()
                    # Verify class filtering worked
                    for prediction in result["predictions"]:
                        assert prediction["class_id"] in [0, 1, 2]
                break

    def test_inference_performance_metrics(self, setup_test_data):
        """Test collection of inference performance metrics."""
        
        if not setup_test_data["image_ids"] or not setup_test_data["detection_models"]:
            pytest.skip("Test data not available")
        
        image_id = setup_test_data["image_ids"][0]
        model = setup_test_data["detection_models"][0]
        
        # Run inference with performance tracking
        request = {
            "image_id": image_id,
            "model_id": model["id"],
            "confidence_threshold": 0.5,
            "track_performance": True
        }
        
        response = client.post("/api/v1/inference/single", json=request)
        
        if response.status_code == 200:
            result = response.json()
            
            # Verify performance metrics are collected
            assert "execution_time_ms" in result
            assert "preprocessing_time_ms" in result
            assert "inference_time_ms" in result
            assert "postprocessing_time_ms" in result
            assert "total_memory_mb" in result
            
            # Validate performance metrics
            assert result["execution_time_ms"] > 0
            assert result["preprocessing_time_ms"] >= 0
            assert result["inference_time_ms"] > 0
            assert result["postprocessing_time_ms"] >= 0
            
            # Total should be sum of components
            total_calculated = (
                result["preprocessing_time_ms"] +
                result["inference_time_ms"] +
                result["postprocessing_time_ms"]
            )
            
            # Allow small variance due to measurement overhead
            assert abs(result["execution_time_ms"] - total_calculated) < 50

    def test_large_batch_inference_scalability(self, setup_test_data):
        """Test scalability with larger batch inference."""
        
        if len(setup_test_data["image_ids"]) < 2:
            pytest.skip("Not enough images for batch testing")
        
        if not setup_test_data["detection_models"]:
            pytest.skip("No detection models available")
        
        model = setup_test_data["detection_models"][0]
        
        # Use all available images (simulate larger batch)
        all_image_ids = setup_test_data["image_ids"]
        
        batch_request = {
            "image_ids": all_image_ids,
            "model_id": model["id"],
            "confidence_threshold": 0.5,
            "batch_size": max(1, len(all_image_ids) // 2),  # Process in smaller chunks
            "priority": "low"  # Lower priority for large batches
        }
        
        response = client.post("/api/v1/inference/batch", json=batch_request)
        
        if response.status_code == 202:
            job_data = response.json()
            job_id = job_data["job_id"]
            
            # Monitor for reasonable time
            max_wait = 60  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                response = client.get(f"/api/v1/inference/jobs/{job_id}")
                if response.status_code == 200:
                    status_data = response.json()
                    
                    if status_data["status"] == "completed":
                        # Verify all images processed
                        assert len(status_data["results"]) == len(all_image_ids)
                        
                        # Check throughput metrics
                        total_time = status_data.get("total_processing_time_ms", 0)
                        if total_time > 0:
                            throughput = len(all_image_ids) / (total_time / 1000)  # images per second
                            assert throughput > 0, "Throughput calculation failed"
                        
                        break
                    
                time.sleep(2)

    def test_inference_result_validation(self, setup_test_data):
        """Test validation of inference results format and content."""
        
        if not setup_test_data["image_ids"] or not setup_test_data["detection_models"]:
            pytest.skip("Test data not available")
        
        image_id = setup_test_data["image_ids"][0]
        model = setup_test_data["detection_models"][0]
        
        response = client.post("/api/v1/inference/single", json={
            "image_id": image_id,
            "model_id": model["id"],
            "confidence_threshold": 0.3
        })
        
        if response.status_code == 200:
            result = response.json()
            
            # Validate top-level structure
            required_fields = [
                "image_id", "model_id", "predictions", "execution_time_ms",
                "model_version", "inference_timestamp"
            ]
            
            for field in required_fields:
                assert field in result, f"Missing required field: {field}"
            
            # Validate predictions structure
            for prediction in result["predictions"]:
                prediction_fields = [
                    "class_id", "confidence", "bounding_box"
                ]
                
                for field in prediction_fields:
                    assert field in prediction, f"Missing prediction field: {field}"
                
                # Validate bounding box structure
                bbox = prediction["bounding_box"]
                bbox_fields = ["x", "y", "width", "height"]
                
                for field in bbox_fields:
                    assert field in bbox, f"Missing bbox field: {field}"
                    assert isinstance(bbox[field], (int, float)), f"Bbox {field} not numeric"
                    if field in ["width", "height"]:
                        assert bbox[field] > 0, f"Bbox {field} must be positive"
                
                # Validate confidence range
                assert 0.0 <= prediction["confidence"] <= 1.0, "Confidence out of range"
                assert isinstance(prediction["class_id"], int), "Class ID must be integer"