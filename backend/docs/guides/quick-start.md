# Quick Start Guide

This guide will help you get started with the ML Evaluation Platform API.

## Base URL

- Development: `http://localhost:8000`
- Production: `https://api.ml-eval.cloud`

## Authentication

Currently, the API does not require authentication for development purposes.

## Common Workflows

### 1. Upload Images

First, upload some images to work with:

```bash
curl -X POST "http://localhost:8000/api/v1/images" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "dataset_split=train"
```

### 2. List Available Models

Check what models are available:

```bash
curl -X GET "http://localhost:8000/api/v1/models"
```

### 3. Run Inference

Run inference on an uploaded image:

```bash
curl -X POST "http://localhost:8000/api/v1/inference/single" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "your-image-id",
    "model_id": "your-model-id",
    "confidence_threshold": 0.5
  }'
```

### 4. Create Manual Annotations

Add manual annotations to images:

```bash
curl -X POST "http://localhost:8000/api/v1/annotations" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "your-image-id",
    "bounding_boxes": [{
      "x": 100,
      "y": 100,
      "width": 200,
      "height": 150,
      "class_id": 1,
      "confidence": 1.0
    }],
    "class_labels": ["person"]
  }'
```

### 5. Start Model Training

Fine-tune a model with your annotations:

```bash
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "your-base-model-id",
    "dataset_id": "your-dataset-id",
    "hyperparameters": {
      "epochs": 100,
      "batch_size": 16,
      "learning_rate": 0.001
    }
  }'
```

## Next Steps

- Check out the [Examples Guide](examples.md) for more detailed examples
- See [Error Handling](error-handling.md) for information about error responses
- Read the full [API Reference](../api/README.md) for complete endpoint documentation

## Support

For questions and support, please refer to the project documentation or open an issue in the repository.
