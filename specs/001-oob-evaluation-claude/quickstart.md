# ML Evaluation Platform - Quickstart Guide

## Overview

This quickstart guide demonstrates the complete workflow of the ML Evaluation Platform, from image upload through model deployment. Follow these steps to validate all core functionality.

## Prerequisites

- Platform deployed and accessible
- Sample images for testing (preferably a mix of formats: JPEG, PNG, TIFF)
- Browser for web interface testing
- API client for endpoint testing (curl, Postman, etc.)

## Quickstart Workflow

### Step 1: Upload and Organize Images

**Objective**: Upload images and organize them into train/validation/test splits

**Actions**:
1. Navigate to the web interface at `http://localhost:3000`
2. Go to the "Upload Images" section
3. Select multiple test images (aim for at least 20 images)
4. Upload 60% to training split, 20% to validation, 20% to test
5. Verify images appear in the correct folders

**Expected Results**:
- All images uploaded successfully
- Images correctly organized by dataset split
- Image metadata displayed (filename, size, dimensions, format)
- Image thumbnails visible in web interface

**API Validation**:
```bash
# Upload images via API
curl -X POST "http://localhost:8000/api/v1/images" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@test_image1.jpg" \
  -F "files=@test_image2.png" \
  -F "dataset_split=train"

# Verify upload
curl "http://localhost:8000/api/v1/images?dataset_split=train&limit=10"
```

### Step 2: Manual Annotation

**Objective**: Create manual annotations using drawing tools

**Actions**:
1. Select an image from the training set
2. Use annotation tools to draw bounding boxes around objects
3. Assign class labels to each annotation
4. Save annotations with user tag
5. Repeat for several images

**Expected Results**:
- Drawing tools responsive and accurate
- Bounding boxes properly saved with coordinates
- Class labels correctly associated
- Annotations visible with distinct colors from model predictions

**API Validation**:
```bash
# Create manual annotation
curl -X POST "http://localhost:8000/api/v1/annotations" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "uuid-of-uploaded-image",
    "bounding_boxes": [{
      "x": 100, "y": 100, "width": 200, "height": 150,
      "class_id": 0, "confidence": 1.0
    }],
    "class_labels": ["person"],
    "user_tag": "test_user"
  }'
```

### Step 3: Model Selection and Assisted Annotation

**Objective**: Use pre-trained models for annotation assistance

**Actions**:
1. Navigate to "Models" section
2. Verify available models are listed (YOLO11, YOLO12, RT-DETR, SAM2)
3. Select SAM2 model for assisted annotation
4. Apply assisted annotation to an image
5. Review and accept/reject model suggestions

**Expected Results**:
- All model variants displayed with metadata
- Assisted annotation generates reasonable predictions
- Model annotations displayed in different color
- User can accept/reject suggestions

**API Validation**:
```bash
# List available models
curl "http://localhost:8000/api/v1/models?type=segmentation"

# Generate assisted annotation
curl -X POST "http://localhost:8000/api/v1/annotations/assisted" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "uuid-of-uploaded-image",
    "model_id": "uuid-of-sam2-model",
    "confidence_threshold": 0.5
  }'
```

### Step 4: Model Inference

**Objective**: Run inference with object detection models

**Actions**:
1. Select YOLO11 model for object detection
2. Run inference on a test image
3. View predictions overlaid on image
4. Initiate batch inference on multiple images
5. Monitor batch processing progress

**Expected Results**:
- Single inference completes in real-time
- Predictions accurately displayed with bounding boxes
- Batch inference job created and monitored
- Progress updates shown in real-time

**API Validation**:
```bash
# Single inference
curl -X POST "http://localhost:8000/api/v1/inference/single" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "uuid-of-test-image",
    "model_id": "uuid-of-yolo11-model",
    "confidence_threshold": 0.5
  }'

# Batch inference
curl -X POST "http://localhost:8000/api/v1/inference/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "image_ids": ["uuid1", "uuid2", "uuid3"],
    "model_id": "uuid-of-yolo11-model"
  }'
```

### Step 5: Performance Evaluation

**Objective**: Calculate and compare model performance metrics

**Actions**:
1. Navigate to "Evaluation" section
2. Select a model and test dataset
3. Calculate performance metrics (mAP, IoU, precision, recall, F1)
4. View execution time measurements
5. Compare performance between different models

**Expected Results**:
- Metrics calculated and displayed accurately
- Execution times measured for each inference
- Model comparison shows performance differences
- Results exportable for further analysis

**API Validation**:
```bash
# Calculate metrics
curl -X POST "http://localhost:8000/api/v1/evaluation/metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "uuid-of-model",
    "dataset_id": "uuid-of-test-dataset",
    "metric_types": ["mAP@50", "mAP@50:95", "precision", "recall", "F1", "execution_time"]
  }'

# Compare models
curl -X POST "http://localhost:8000/api/v1/evaluation/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "model_ids": ["uuid1", "uuid2"],
    "dataset_id": "uuid-of-test-dataset",
    "metric_types": ["mAP@50", "execution_time"]
  }'
```

### Step 6: Model Training/Fine-tuning

**Objective**: Train a model using user annotations

**Actions**:
1. Ensure sufficient annotations exist (minimum 10+ annotated images)
2. Navigate to "Training" section
3. Select base model (e.g., YOLO11s) and training dataset
4. Configure hyperparameters (epochs, batch size, learning rate)
5. Start training job
6. Monitor training progress and logs in real-time

**Expected Results**:
- Training job initiated successfully
- Real-time progress updates displayed
- Training logs show epoch progress and metrics
- Trained model available after completion

**API Validation**:
```bash
# Start training
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "uuid-of-base-model",
    "dataset_id": "uuid-of-training-dataset",
    "hyperparameters": {
      "epochs": 50,
      "batch_size": 16,
      "learning_rate": 0.001,
      "patience": 10
    }
  }'

# Monitor training progress
curl "http://localhost:8000/api/v1/training/jobs/{job_id}"
```

### Step 7: Model Deployment

**Objective**: Deploy trained model as REST API endpoint

**Actions**:
1. Select completed trained model
2. Configure deployment settings (replicas, resources)
3. Deploy model as API endpoint
4. Test deployed endpoint with sample image
5. Monitor deployment performance metrics

**Expected Results**:
- Model deployed successfully
- API endpoint accessible and responsive
- Predictions returned correctly via deployed API
- Performance metrics tracked

**API Validation**:
```bash
# Deploy model
curl -X POST "http://localhost:8000/api/v1/deployments" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "uuid-of-trained-model",
    "version": "1.0.0",
    "configuration": {
      "replicas": 1,
      "cpu_limit": "1000m",
      "memory_limit": "2Gi",
      "gpu_required": false
    }
  }'

# Test deployed endpoint
curl -X POST "{deployed_endpoint_url}/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### Step 8: Data Export

**Objective**: Export annotations in standard formats

**Actions**:
1. Navigate to "Export" section
2. Select images with annotations
3. Choose export format (COCO, YOLO, Pascal VOC)
4. Include/exclude model predictions
5. Download exported data

**Expected Results**:
- Export completes successfully
- Downloaded file contains properly formatted annotations
- Both user and model annotations included as requested
- File structure matches standard format specifications

**API Validation**:
```bash
# Export annotations
curl -X POST "http://localhost:8000/api/v1/export/annotations" \
  -H "Content-Type: application/json" \
  -d '{
    "image_ids": ["uuid1", "uuid2", "uuid3"],
    "format": "COCO",
    "include_predictions": true,
    "model_id": "uuid-of-model"
  }' \
  --output annotations_export.zip
```

## Error Handling Validation

### Test Error Scenarios

**Unsupported File Formats**:
1. Upload a .txt file as image
2. Verify appropriate error message displayed
3. Confirm no corrupt data stored

**Memory Issues** (Large Files):
1. Upload extremely large image (if available)
2. Verify system handles gracefully
3. Check for proper progress indication or chunked processing

**Training Failures**:
1. Start training with insufficient data
2. Verify error reporting and cleanup
3. Confirm system state remains consistent

**Annotation Conflicts**:
1. Simulate concurrent annotation of same image
2. Verify conflict resolution
3. Ensure data integrity maintained

## Success Criteria

✅ **Complete Workflow**: All 8 steps executed successfully without errors

✅ **Performance Requirements**: 
- Real-time inference (<2 seconds per image)
- Batch processing handles multiple images efficiently
- Training monitoring provides real-time updates

✅ **Data Integrity**: 
- All uploads, annotations, and training results properly stored
- No data loss during operations
- Consistent state across all operations

✅ **API Functionality**: 
- All endpoints respond correctly
- Proper error handling and status codes
- Data formats match API specification

✅ **User Experience**: 
- Interface responsive and intuitive
- Real-time updates work correctly
- Error messages helpful and actionable

✅ **Model Integration**: 
- All model variants (YOLO11/12, RT-DETR, SAM2) functional
- Performance metrics accurate
- Model deployment successful

## Troubleshooting

### Common Issues

**Images Not Uploading**:
- Check file format support
- Verify network connectivity
- Check server disk space

**Model Inference Slow**:
- Verify GPU availability
- Check system resources
- Consider smaller model variants

**Training Not Starting**:
- Ensure sufficient annotations
- Check dataset configuration
- Verify model compatibility

**API Endpoints Not Responding**:
- Check service status
- Verify port configuration
- Review server logs

### Getting Help

- Check server logs: `docker-compose logs -f api`
- Review API documentation: `http://localhost:8000/docs`
- Monitor system resources: `docker stats`
- Validate data integrity: Check database directly

This quickstart guide validates all major functionality and serves as both a testing protocol and user onboarding experience.