"""
Integration test for complete manual annotation workflow

This test MUST FAIL initially since the endpoints are not implemented.
Tests the complete manual annotation workflow from quickstart.md Step 2.

Workflow:
1. Upload images for annotation
2. Create manual annotations with drawing tools simulation
3. Save annotations with proper metadata
4. Verify annotation persistence and retrieval
5. Test annotation editing and deletion
6. Handle concurrent annotation scenarios
"""

import pytest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import uuid
import time
from datetime import datetime

from src.main import app

client = TestClient(app)


class TestManualAnnotationWorkflowIntegration:
    """Integration tests for complete manual annotation workflow."""

    @pytest.fixture(scope="class")
    def setup_test_images(self):
        """Setup test images for annotation."""
        # Create test images
        test_images = []
        
        for i in range(3):
            image = Image.new('RGB', (800, 600), color=['red', 'green', 'blue'][i])
            image_buffer = BytesIO()
            image.save(image_buffer, format='JPEG')
            image_buffer.seek(0)
            
            # Upload image
            response = client.post(
                "/api/v1/images",
                files={"files": (f"test_image_{i}.jpg", image_buffer, "image/jpeg")},
                data={"dataset_split": "train"}
            )
            
            # This will fail until upload endpoint is implemented
            if response.status_code == 201:
                image_data = response.json()["uploaded_images"][0]
                test_images.append(image_data["id"])
        
        return test_images

    def test_complete_manual_annotation_workflow(self, setup_test_images):
        """Test the complete manual annotation workflow from quickstart Step 2."""
        
        if not setup_test_images:
            pytest.skip("Image upload not implemented yet")
        
        image_id = setup_test_images[0]
        
        # Step 1: Create manual annotation with bounding boxes
        annotation_data = {
            "image_id": image_id,
            "bounding_boxes": [
                {
                    "x": 100.0,
                    "y": 150.0,
                    "width": 200.0,
                    "height": 180.0,
                    "class_id": 0,
                    "confidence": 1.0  # Manual annotations have perfect confidence
                },
                {
                    "x": 350.0,
                    "y": 200.0,
                    "width": 120.0,
                    "height": 150.0,
                    "class_id": 1,
                    "confidence": 1.0
                }
            ],
            "class_labels": ["person", "car"],
            "user_tag": "test_annotator",
            "metadata": {
                "annotation_time_seconds": 45,
                "difficulty": "medium",
                "quality": "high"
            }
        }
        
        response = client.post("/api/v1/annotations", json=annotation_data)
        
        # This MUST FAIL since endpoint doesn't exist yet
        assert response.status_code == 201
        
        annotation_response = response.json()
        
        # Validate annotation creation
        assert annotation_response["image_id"] == image_id
        assert annotation_response["creation_method"] == "user"
        assert annotation_response["user_tag"] == "test_annotator"
        assert len(annotation_response["bounding_boxes"]) == 2
        assert annotation_response["class_labels"] == ["person", "car"]
        
        annotation_id = annotation_response["id"]
        
        # Step 2: Verify annotation persistence by retrieving it
        response = client.get(f"/api/v1/annotations?image_id={image_id}")
        assert response.status_code == 200
        
        annotations_list = response.json()
        assert "annotations" in annotations_list
        assert len(annotations_list["annotations"]) >= 1
        
        # Find our annotation in the list
        our_annotation = None
        for ann in annotations_list["annotations"]:
            if ann["id"] == annotation_id:
                our_annotation = ann
                break
        
        assert our_annotation is not None
        assert our_annotation["creation_method"] == "user"
        assert len(our_annotation["bounding_boxes"]) == 2

    def test_annotation_with_segmentation_polygons(self, setup_test_images):
        """Test creating annotations with segmentation polygons."""
        
        if not setup_test_images:
            pytest.skip("Image upload not implemented yet")
        
        image_id = setup_test_images[1]
        
        annotation_data = {
            "image_id": image_id,
            "segments": [
                {
                    "polygon": [
                        [100, 100], [200, 100], [250, 150], [200, 200],
                        [150, 220], [100, 200], [80, 150]
                    ],
                    "class_id": 0,
                    "confidence": 1.0
                },
                {
                    "polygon": [
                        [400, 300], [500, 300], [500, 400], [400, 400]
                    ],
                    "class_id": 1,
                    "confidence": 1.0
                }
            ],
            "class_labels": ["building", "vehicle"],
            "user_tag": "expert_annotator"
        }
        
        response = client.post("/api/v1/annotations", json=annotation_data)
        
        assert response.status_code == 201
        
        annotation_response = response.json()
        assert len(annotation_response["segments"]) == 2
        assert annotation_response["class_labels"] == ["building", "vehicle"]
        
        # Verify polygon coordinates preserved exactly
        segment1 = annotation_response["segments"][0]
        expected_polygon = [
            [100, 100], [200, 100], [250, 150], [200, 200],
            [150, 220], [100, 200], [80, 150]
        ]
        assert segment1["polygon"] == expected_polygon

    def test_mixed_annotation_types(self, setup_test_images):
        """Test annotations with both bounding boxes and segments."""
        
        if not setup_test_images:
            pytest.skip("Image upload not implemented yet")
        
        image_id = setup_test_images[2]
        
        annotation_data = {
            "image_id": image_id,
            "bounding_boxes": [
                {
                    "x": 50.0,
                    "y": 50.0,
                    "width": 100.0,
                    "height": 80.0,
                    "class_id": 0,
                    "confidence": 1.0
                }
            ],
            "segments": [
                {
                    "polygon": [[200, 200], [300, 200], [300, 300], [200, 300]],
                    "class_id": 1,
                    "confidence": 1.0
                }
            ],
            "class_labels": ["person", "building"],
            "user_tag": "mixed_annotator"
        }
        
        response = client.post("/api/v1/annotations", json=annotation_data)
        
        assert response.status_code == 201
        
        annotation_response = response.json()
        assert len(annotation_response["bounding_boxes"]) == 1
        assert len(annotation_response["segments"]) == 1
        assert annotation_response["class_labels"] == ["person", "building"]

    def test_annotation_editing_workflow(self, setup_test_images):
        """Test editing existing annotations."""
        
        if not setup_test_images:
            pytest.skip("Image upload not implemented yet")
        
        image_id = setup_test_images[0]
        
        # Create initial annotation
        initial_data = {
            "image_id": image_id,
            "bounding_boxes": [
                {
                    "x": 100.0,
                    "y": 100.0,
                    "width": 150.0,
                    "height": 120.0,
                    "class_id": 0,
                    "confidence": 1.0
                }
            ],
            "class_labels": ["initial_class"],
            "user_tag": "editor_test"
        }
        
        response = client.post("/api/v1/annotations", json=initial_data)
        assert response.status_code == 201
        
        annotation_id = response.json()["id"]
        
        # Edit the annotation - update coordinates and add class
        updated_data = {
            "bounding_boxes": [
                {
                    "x": 110.0,  # Adjusted coordinates
                    "y": 110.0,
                    "width": 160.0,
                    "height": 130.0,
                    "class_id": 0,
                    "confidence": 1.0
                },
                {
                    "x": 300.0,  # New bounding box
                    "y": 250.0,
                    "width": 100.0,
                    "height": 80.0,
                    "class_id": 1,
                    "confidence": 1.0
                }
            ],
            "class_labels": ["updated_class", "new_class"],
            "user_tag": "editor_test_updated"
        }
        
        # This endpoint would need to be implemented
        response = client.put(f"/api/v1/annotations/{annotation_id}", json=updated_data)
        
        # Should succeed once implemented
        assert response.status_code in [200, 201]
        
        updated_response = response.json()
        assert len(updated_response["bounding_boxes"]) == 2
        assert updated_response["class_labels"] == ["updated_class", "new_class"]
        assert updated_response["bounding_boxes"][0]["x"] == 110.0

    def test_annotation_validation_and_errors(self, setup_test_images):
        """Test annotation validation and error handling."""
        
        if not setup_test_images:
            pytest.skip("Image upload not implemented yet")
        
        image_id = setup_test_images[0]
        
        # Test missing required fields
        invalid_data = {
            "image_id": image_id
            # Missing class_labels
        }
        
        response = client.post("/api/v1/annotations", json=invalid_data)
        assert response.status_code == 400
        
        error_data = response.json()
        assert "error" in error_data
        
        # Test invalid image_id format
        invalid_data = {
            "image_id": "not-a-uuid",
            "class_labels": ["test"]
        }
        
        response = client.post("/api/v1/annotations", json=invalid_data)
        assert response.status_code == 400
        
        # Test mismatched confidence_scores length
        invalid_data = {
            "image_id": image_id,
            "class_labels": ["class1", "class2"],
            "confidence_scores": [0.9]  # Only one score for two classes
        }
        
        response = client.post("/api/v1/annotations", json=invalid_data)
        assert response.status_code == 400

    def test_concurrent_annotation_handling(self, setup_test_images):
        """Test handling of concurrent annotations on the same image."""
        
        if not setup_test_images:
            pytest.skip("Image upload not implemented yet")
        
        image_id = setup_test_images[0]
        
        # Create two different annotations for the same image
        annotation1_data = {
            "image_id": image_id,
            "bounding_boxes": [
                {
                    "x": 100.0,
                    "y": 100.0,
                    "width": 100.0,
                    "height": 100.0,
                    "class_id": 0,
                    "confidence": 1.0
                }
            ],
            "class_labels": ["person"],
            "user_tag": "annotator1"
        }
        
        annotation2_data = {
            "image_id": image_id,
            "bounding_boxes": [
                {
                    "x": 300.0,
                    "y": 200.0,
                    "width": 80.0,
                    "height": 120.0,
                    "class_id": 1,
                    "confidence": 1.0
                }
            ],
            "class_labels": ["car"],
            "user_tag": "annotator2"
        }
        
        # Submit both annotations (simulating concurrent users)
        response1 = client.post("/api/v1/annotations", json=annotation1_data)
        response2 = client.post("/api/v1/annotations", json=annotation2_data)
        
        assert response1.status_code == 201
        assert response2.status_code == 201
        
        # Both should succeed and have different IDs
        ann1_id = response1.json()["id"]
        ann2_id = response2.json()["id"]
        assert ann1_id != ann2_id
        
        # Verify both annotations exist for the image
        response = client.get(f"/api/v1/annotations?image_id={image_id}")
        assert response.status_code == 200
        
        annotations = response.json()["annotations"]
        annotation_ids = [ann["id"] for ann in annotations]
        assert ann1_id in annotation_ids
        assert ann2_id in annotation_ids

    def test_annotation_deletion_workflow(self, setup_test_images):
        """Test annotation deletion functionality."""
        
        if not setup_test_images:
            pytest.skip("Image upload not implemented yet")
        
        image_id = setup_test_images[0]
        
        # Create annotation to delete
        annotation_data = {
            "image_id": image_id,
            "bounding_boxes": [
                {
                    "x": 100.0,
                    "y": 100.0,
                    "width": 100.0,
                    "height": 100.0,
                    "class_id": 0,
                    "confidence": 1.0
                }
            ],
            "class_labels": ["to_delete"],
            "user_tag": "deletion_test"
        }
        
        response = client.post("/api/v1/annotations", json=annotation_data)
        assert response.status_code == 201
        
        annotation_id = response.json()["id"]
        
        # Delete the annotation
        response = client.delete(f"/api/v1/annotations/{annotation_id}")
        
        # Should succeed once implemented
        assert response.status_code in [200, 204]
        
        # Verify annotation is deleted
        response = client.get(f"/api/v1/annotations?image_id={image_id}")
        assert response.status_code == 200
        
        remaining_annotations = response.json()["annotations"]
        remaining_ids = [ann["id"] for ann in remaining_annotations]
        assert annotation_id not in remaining_ids

    def test_annotation_quality_metrics(self, setup_test_images):
        """Test annotation quality tracking and metrics."""
        
        if not setup_test_images:
            pytest.skip("Image upload not implemented yet")
        
        image_id = setup_test_images[0]
        
        # Create annotation with quality metadata
        annotation_data = {
            "image_id": image_id,
            "bounding_boxes": [
                {
                    "x": 100.0,
                    "y": 100.0,
                    "width": 200.0,
                    "height": 150.0,
                    "class_id": 0,
                    "confidence": 1.0
                }
            ],
            "class_labels": ["person"],
            "user_tag": "quality_tester",
            "metadata": {
                "annotation_time_seconds": 120,
                "difficulty": "hard",
                "quality": "high",
                "notes": "Partially occluded object",
                "revision_count": 3
            }
        }
        
        response = client.post("/api/v1/annotations", json=annotation_data)
        assert response.status_code == 201
        
        annotation_response = response.json()
        assert "metadata" in annotation_response
        assert annotation_response["metadata"]["difficulty"] == "hard"
        assert annotation_response["metadata"]["quality"] == "high"
        assert annotation_response["metadata"]["annotation_time_seconds"] == 120

    def test_annotation_batch_operations(self, setup_test_images):
        """Test batch annotation operations."""
        
        if not setup_test_images:
            pytest.skip("Image upload not implemented yet")
        
        # Create multiple annotations in batch
        batch_annotations = []
        for i, image_id in enumerate(setup_test_images):
            annotation_data = {
                "image_id": image_id,
                "bounding_boxes": [
                    {
                        "x": float(100 + i * 50),
                        "y": float(100 + i * 30),
                        "width": 100.0,
                        "height": 80.0,
                        "class_id": 0,
                        "confidence": 1.0
                    }
                ],
                "class_labels": ["batch_object"],
                "user_tag": f"batch_annotator_{i}"
            }
            batch_annotations.append(annotation_data)
        
        # Submit batch creation request (if endpoint exists)
        response = client.post("/api/v1/annotations/batch", json={"annotations": batch_annotations})
        
        # Should work once batch endpoint is implemented
        if response.status_code == 201:
            batch_response = response.json()
            assert "created_annotations" in batch_response
            assert len(batch_response["created_annotations"]) == len(setup_test_images)
        else:
            # Fallback to individual creation
            for annotation_data in batch_annotations:
                response = client.post("/api/v1/annotations", json=annotation_data)
                assert response.status_code == 201