# API Examples

This guide provides comprehensive examples for all major API operations.

## Image Management

### Upload Single Image

```bash
curl -X POST "http://localhost:8000/api/v1/images" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sample_image.jpg" \
  -F "dataset_split=train"
```

### Upload Multiple Images

```bash
curl -X POST "http://localhost:8000/api/v1/images" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "dataset_split=validation"
```

### List Images with Filters

```bash
# Get all training images
curl -X GET "http://localhost:8000/api/v1/images?dataset_split=train&limit=20"

# Get images with pagination
curl -X GET "http://localhost:8000/api/v1/images?limit=50&offset=100"
```

### Get Image Details

```bash
curl -X GET "http://localhost:8000/api/v1/images/{image_id}"
```

## Model Management

### List Available Models

```bash
# Get all models
curl -X GET "http://localhost:8000/api/v1/models"

# Filter by type
curl -X GET "http://localhost:8000/api/v1/models?type=detection"

# Filter by framework
curl -X GET "http://localhost:8000/api/v1/models?framework=YOLO11"
```

### Get Model Details

```bash
curl -X GET "http://localhost:8000/api/v1/models/{model_id}"
```

## Annotations

### Create Manual Annotation

```bash
curl -X POST "http://localhost:8000/api/v1/annotations" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "123e4567-e89b-12d3-a456-426614174000",
    "bounding_boxes": [
      {
        "x": 100,
        "y": 50,
        "width": 200,
        "height": 150,
        "class_id": 0,
        "confidence": 1.0
      }
    ],
    "segments": [
      {
        "polygon": [[100, 50], [300, 50], [300, 200], [100, 200]],
        "class_id": 0,
        "confidence": 1.0
      }
    ],
    "class_labels": ["person"],
    "user_tag": "manual_annotation",
    "metadata": {
      "annotator": "user123",
      "annotation_time": 45
    }
  }'
```

### Generate Assisted Annotation

```bash
curl -X POST "http://localhost:8000/api/v1/annotations/assisted" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "123e4567-e89b-12d3-a456-426614174000",
    "model_id": "456e7890-e89b-12d3-a456-426614174000",
    "confidence_threshold": 0.7
  }'
```

### List Annotations

```bash
# Get all annotations for an image
curl -X GET "http://localhost:8000/api/v1/annotations?image_id={image_id}"

# Get annotations by creation method
curl -X GET "http://localhost:8000/api/v1/annotations?creation_method=user"
```

## Inference

### Single Image Inference

```bash
curl -X POST "http://localhost:8000/api/v1/inference/single" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "123e4567-e89b-12d3-a456-426614174000",
    "model_id": "456e7890-e89b-12d3-a456-426614174000",
    "confidence_threshold": 0.5
  }'
```

### Batch Inference

```bash
curl -X POST "http://localhost:8000/api/v1/inference/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "image_ids": [
      "123e4567-e89b-12d3-a456-426614174000",
      "234e5678-e89b-12d3-a456-426614174001",
      "345e6789-e89b-12d3-a456-426614174002"
    ],
    "model_id": "456e7890-e89b-12d3-a456-426614174000",
    "confidence_threshold": 0.6
  }'
```

### Monitor Inference Job

```bash
curl -X GET "http://localhost:8000/api/v1/inference/jobs/{job_id}"
```

## Training

### Start Model Training

```bash
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "456e7890-e89b-12d3-a456-426614174000",
    "dataset_id": "789e0123-e89b-12d3-a456-426614174000",
    "hyperparameters": {
      "epochs": 50,
      "batch_size": 16,
      "learning_rate": 0.001,
      "patience": 10
    },
    "metadata": {
      "experiment_name": "custom_training_v1",
      "description": "Fine-tuning YOLO11 on custom dataset"
    }
  }'
```

### Monitor Training Job

```bash
curl -X GET "http://localhost:8000/api/v1/training/jobs/{job_id}"
```

## Performance Evaluation

### Calculate Metrics

```bash
curl -X POST "http://localhost:8000/api/v1/evaluation/metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "456e7890-e89b-12d3-a456-426614174000",
    "dataset_id": "789e0123-e89b-12d3-a456-426614174000",
    "metric_types": ["mAP", "mAP@50", "precision", "recall", "F1"],
    "iou_threshold": 0.5
  }'
```

### Compare Models

```bash
curl -X POST "http://localhost:8000/api/v1/evaluation/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "model_ids": [
      "456e7890-e89b-12d3-a456-426614174000",
      "567e8901-e89b-12d3-a456-426614174001",
      "678e9012-e89b-12d3-a456-426614174002"
    ],
    "dataset_id": "789e0123-e89b-12d3-a456-426614174000",
    "metric_types": ["mAP", "execution_time"]
  }'
```

## Model Deployment

### Deploy Model

```bash
curl -X POST "http://localhost:8000/api/v1/deployments" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "456e7890-e89b-12d3-a456-426614174000",
    "version": "v1.0.0",
    "configuration": {
      "replicas": 2,
      "cpu_limit": "2000m",
      "memory_limit": "4Gi",
      "gpu_required": true
    },
    "metadata": {
      "environment": "production",
      "description": "Production deployment of custom trained model"
    }
  }'
```

### List Deployments

```bash
curl -X GET "http://localhost:8000/api/v1/deployments"
```

### Update Deployment

```bash
curl -X PATCH "http://localhost:8000/api/v1/deployments/{deployment_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "inactive",
    "configuration": {
      "replicas": 1
    }
  }'
```

## Data Export

### Export Annotations

```bash
# Export in COCO format
curl -X POST "http://localhost:8000/api/v1/export/annotations" \
  -H "Content-Type: application/json" \
  -d '{
    "image_ids": [
      "123e4567-e89b-12d3-a456-426614174000",
      "234e5678-e89b-12d3-a456-426614174001"
    ],
    "format": "COCO",
    "include_predictions": true,
    "model_id": "456e7890-e89b-12d3-a456-426614174000"
  }' \
  --output annotations_export.zip

# Export in YOLO format
curl -X POST "http://localhost:8000/api/v1/export/annotations" \
  -H "Content-Type: application/json" \
  -d '{
    "image_ids": [
      "123e4567-e89b-12d3-a456-426614174000"
    ],
    "format": "YOLO",
    "include_predictions": false
  }' \
  --output yolo_annotations.zip
```

## Python SDK Examples

### Basic Usage

```python
import requests
import json

BASE_URL = "http://localhost:8000"

class MLEvaluationClient:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def upload_image(self, file_path, dataset_split="train"):
        url = f"{self.base_url}/api/v1/images"
        with open(file_path, 'rb') as f:
            files = {"files": f}
            data = {"dataset_split": dataset_split}
            response = self.session.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    
    def run_inference(self, image_id, model_id, confidence_threshold=0.5):
        url = f"{self.base_url}/api/v1/inference/single"
        data = {
            "image_id": image_id,
            "model_id": model_id,
            "confidence_threshold": confidence_threshold
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

# Usage example
client = MLEvaluationClient()

# Upload image
result = client.upload_image("sample_image.jpg", "test")
image_id = result["uploaded_images"][0]["id"]

# Run inference
inference_result = client.run_inference(image_id, "model-id-here")
print(f"Found {len(inference_result['predictions'])} objects")
```

For more examples and detailed integration patterns, see the [Quick Start Guide](quick-start.md).
