#!/bin/bash

# ML Evaluation Platform API - cURL Examples
# 
# This script contains example API calls using cURL for all major endpoints.
# Replace placeholder values (marked with {}) with actual IDs from your system.

BASE_URL="http://localhost:8000"

echo "=== ML Evaluation Platform API Examples ==="
echo

# 1. UPLOAD IMAGES
echo "1. Uploading sample images..."
curl -X POST "$BASE_URL/api/v1/images" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sample_image1.jpg" \
  -F "files=@sample_image2.jpg" \
  -F "dataset_split=train"

echo
echo "---"

# 2. LIST IMAGES
echo "2. Listing uploaded images..."
curl -X GET "$BASE_URL/api/v1/images?limit=10"

echo
echo "---"

# 3. GET IMAGE DETAILS
echo "3. Getting image details..."
curl -X GET "$BASE_URL/api/v1/images/{image_id}"

echo
echo "---"

# 4. LIST MODELS
echo "4. Listing available models..."
curl -X GET "$BASE_URL/api/v1/models"

echo
echo "---"

# 5. CREATE MANUAL ANNOTATION
echo "5. Creating manual annotation..."
curl -X POST "$BASE_URL/api/v1/annotations" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "{image_id}",
    "bounding_boxes": [{
      "x": 100,
      "y": 100,
      "width": 200,
      "height": 150,
      "class_id": 1,
      "confidence": 1.0
    }],
    "class_labels": ["person"],
    "user_tag": "manual_example"
  }'

echo
echo "---"

# 6. GENERATE ASSISTED ANNOTATION
echo "6. Generating assisted annotation..."
curl -X POST "$BASE_URL/api/v1/annotations/assisted" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "{image_id}",
    "model_id": "{model_id}",
    "confidence_threshold": 0.7
  }'

echo
echo "---"

# 7. SINGLE IMAGE INFERENCE
echo "7. Running single image inference..."
curl -X POST "$BASE_URL/api/v1/inference/single" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "{image_id}",
    "model_id": "{model_id}",
    "confidence_threshold": 0.5
  }'

echo
echo "---"

# 8. BATCH INFERENCE
echo "8. Starting batch inference job..."
curl -X POST "$BASE_URL/api/v1/inference/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "image_ids": ["{image_id_1}", "{image_id_2}"],
    "model_id": "{model_id}",
    "confidence_threshold": 0.6
  }'

echo
echo "---"

# 9. START TRAINING JOB
echo "9. Starting model training job..."
curl -X POST "$BASE_URL/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "{base_model_id}",
    "dataset_id": "{dataset_id}",
    "hyperparameters": {
      "epochs": 50,
      "batch_size": 16,
      "learning_rate": 0.001,
      "patience": 10
    }
  }'

echo
echo "---"

# 10. CALCULATE PERFORMANCE METRICS
echo "10. Calculating performance metrics..."
curl -X POST "$BASE_URL/api/v1/evaluation/metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "{model_id}",
    "dataset_id": "{dataset_id}",
    "metric_types": ["mAP", "mAP@50", "precision", "recall", "F1"],
    "iou_threshold": 0.5
  }'

echo
echo "---"

# 11. COMPARE MODELS
echo "11. Comparing model performance..."
curl -X POST "$BASE_URL/api/v1/evaluation/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "model_ids": ["{model_id_1}", "{model_id_2}"],
    "dataset_id": "{dataset_id}",
    "metric_types": ["mAP", "execution_time"]
  }'

echo
echo "---"

# 12. DEPLOY MODEL
echo "12. Deploying model..."
curl -X POST "$BASE_URL/api/v1/deployments" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "{model_id}",
    "version": "v1.0.0",
    "configuration": {
      "replicas": 1,
      "cpu_limit": "1000m",
      "memory_limit": "2Gi",
      "gpu_required": false
    }
  }'

echo
echo "---"

# 13. LIST DEPLOYMENTS
echo "13. Listing deployments..."
curl -X GET "$BASE_URL/api/v1/deployments"

echo
echo "---"

# 14. EXPORT ANNOTATIONS
echo "14. Exporting annotations..."
curl -X POST "$BASE_URL/api/v1/export/annotations" \
  -H "Content-Type: application/json" \
  -d '{
    "image_ids": ["{image_id}"],
    "format": "COCO",
    "include_predictions": false
  }' \
  --output exported_annotations.zip

echo
echo "=== Examples completed ==="
echo "Remember to replace {placeholder} values with actual IDs from your system!"
