# Data Model: ML Evaluation Platform

## Entity Overview

The ML evaluation platform manages seven core entities that support the complete workflow from image upload through model deployment:

1. **Image** - Uploaded image files with metadata
2. **Annotation** - Object/segment annotations with coordinates and labels
3. **Model** - Available AI models for detection/segmentation
4. **Dataset** - Collections of images organized by splits
5. **Training Job** - Model training/fine-tuning tasks
6. **Inference Job** - Batch inference tasks
7. **Performance Metric** - Evaluation results and metrics
8. **Deployment** - Deployed model instances with endpoints

## Entity Definitions

### Image

Represents uploaded image files with associated metadata and dataset organization.

**Fields:**
- `id` (UUID, PK) - Unique identifier
- `filename` (String, required) - Original filename
- `file_path` (String, required) - Storage path/URL
- `file_size` (Integer, required) - File size in bytes
- `format` (String, required) - Image format (JPEG, PNG, TIFF, etc.)
- `width` (Integer, required) - Image width in pixels
- `height` (Integer, required) - Image height in pixels
- `dataset_split` (Enum, required) - train/validation/test
- `upload_timestamp` (DateTime, required) - When uploaded
- `metadata` (JSONB, optional) - Additional image metadata

**Relationships:**
- One-to-many with Annotation
- Many-to-many with Dataset
- One-to-many with Inference Job (as input)

**Validation Rules:**
- filename must not be empty
- file_size must be positive
- format must be supported image type
- width and height must be positive
- dataset_split must be one of: train, validation, test

### Annotation

Represents object/segment annotations with coordinates, labels, and creation metadata.

**Fields:**
- `id` (UUID, PK) - Unique identifier
- `image_id` (UUID, FK to Image, required) - Associated image
- `bounding_boxes` (JSONB, optional) - Object detection boxes
- `segments` (JSONB, optional) - Segmentation masks/polygons
- `class_labels` (Array[String], required) - Object class names
- `confidence_scores` (Array[Float], optional) - Prediction confidence
- `creation_method` (Enum, required) - user/model
- `model_id` (UUID, FK to Model, optional) - If created by model
- `user_tag` (String, optional) - User identifier for manual annotations
- `created_at` (DateTime, required) - Creation timestamp
- `metadata` (JSONB, optional) - Additional annotation data

**Relationships:**
- Many-to-one with Image
- Many-to-one with Model (if model-generated)

**Validation Rules:**
- Either bounding_boxes or segments must be present
- class_labels array must not be empty
- confidence_scores length must match class_labels length if present
- creation_method must be 'user' or 'model'
- If creation_method is 'model', model_id must be present

**State Transitions:**
- Draft → Validated → Finalized (for user annotations)
- Generated → Reviewed → Accepted/Rejected (for model annotations)

### Model

Represents available AI models with metadata, performance metrics, and status.

**Fields:**
- `id` (UUID, PK) - Unique identifier
- `name` (String, required) - Model name
- `type` (Enum, required) - detection/segmentation
- `variant` (String, required) - Model size variant (nano, small, medium, large, xl)
- `version` (String, required) - Model version
- `framework` (String, required) - Model framework (YOLO11, YOLO12, RT-DETR, SAM2)
- `model_path` (String, required) - Storage path for model files
- `training_status` (Enum, required) - pre-trained/training/trained/failed
- `performance_metrics` (JSONB, optional) - mAP, accuracy, speed metrics
- `created_at` (DateTime, required) - Creation timestamp
- `updated_at` (DateTime, required) - Last update timestamp
- `metadata` (JSONB, optional) - Framework-specific configuration

**Relationships:**
- One-to-many with Annotation (model-generated)
- One-to-many with Training Job
- One-to-many with Inference Job
- One-to-many with Deployment

**Validation Rules:**
- name must be unique within type and variant
- type must be 'detection' or 'segmentation'
- framework must be supported (YOLO11, YOLO12, RT-DETR, SAM2)
- model_path must be accessible
- training_status transitions: pre-trained/training → trained/failed

### Dataset

Represents collections of images organized by train/validation/test splits.

**Fields:**
- `id` (UUID, PK) - Unique identifier
- `name` (String, required) - Dataset name
- `description` (Text, optional) - Dataset description
- `train_count` (Integer, default=0) - Number of training images
- `validation_count` (Integer, default=0) - Number of validation images
- `test_count` (Integer, default=0) - Number of test images
- `created_at` (DateTime, required) - Creation timestamp
- `updated_at` (DateTime, required) - Last update timestamp
- `metadata` (JSONB, optional) - Dataset configuration

**Relationships:**
- Many-to-many with Image
- One-to-many with Training Job

**Validation Rules:**
- name must be unique
- At least one split count must be > 0
- Split counts must match actual image assignments

### Training Job

Represents model training/fine-tuning tasks with status and configuration.

**Fields:**
- `id` (UUID, PK) - Unique identifier
- `base_model_id` (UUID, FK to Model, required) - Base model for training
- `dataset_id` (UUID, FK to Dataset, required) - Training dataset
- `status` (Enum, required) - queued/running/completed/failed
- `progress_percentage` (Float, default=0.0) - Training progress
- `hyperparameters` (JSONB, required) - Training configuration
- `execution_logs` (Text, optional) - Training logs
- `start_time` (DateTime, optional) - Training start time
- `end_time` (DateTime, optional) - Training completion time
- `result_model_id` (UUID, FK to Model, optional) - Resulting trained model
- `created_at` (DateTime, required) - Job creation timestamp
- `metadata` (JSONB, optional) - Additional training metadata

**Relationships:**
- Many-to-one with Model (base model)
- Many-to-one with Dataset
- One-to-one with Model (result model)

**Validation Rules:**
- progress_percentage must be between 0.0 and 100.0
- status transitions: queued → running → completed/failed
- If status is completed, result_model_id must be present
- hyperparameters must contain required training parameters

**State Transitions:**
- Queued → Running (when resources available)
- Running → Completed (on successful training)
- Running → Failed (on training error)
- Any → Cancelled (user cancellation)

### Inference Job

Represents batch inference tasks on multiple images.

**Fields:**
- `id` (UUID, PK) - Unique identifier
- `model_id` (UUID, FK to Model, required) - Model for inference
- `target_images` (Array[UUID], required) - Image IDs for processing
- `status` (Enum, required) - queued/running/completed/failed
- `progress_percentage` (Float, default=0.0) - Processing progress
- `results` (JSONB, optional) - Inference results
- `execution_logs` (Text, optional) - Processing logs
- `start_time` (DateTime, optional) - Processing start time
- `end_time` (DateTime, optional) - Processing completion time
- `created_at` (DateTime, required) - Job creation timestamp
- `metadata` (JSONB, optional) - Additional inference metadata

**Relationships:**
- Many-to-one with Model
- Many-to-many with Image (via target_images)
- One-to-many with Annotation (generated annotations)

**Validation Rules:**
- target_images array must not be empty
- progress_percentage must be between 0.0 and 100.0
- status transitions: queued → running → completed/failed
- All target_images must reference valid Image entities

### Performance Metric

Represents evaluation results with accuracy scores and execution times.

**Fields:**
- `id` (UUID, PK) - Unique identifier
- `model_id` (UUID, FK to Model, required) - Evaluated model
- `dataset_id` (UUID, FK to Dataset, optional) - Test dataset
- `metric_type` (String, required) - mAP, IoU, precision, recall, F1, execution_time
- `metric_value` (Float, required) - Metric value
- `threshold` (Float, optional) - IoU threshold for mAP calculations
- `class_name` (String, optional) - Class-specific metrics
- `evaluation_timestamp` (DateTime, required) - When metric was calculated
- `metadata` (JSONB, optional) - Additional metric details

**Relationships:**
- Many-to-one with Model
- Many-to-one with Dataset (optional)

**Validation Rules:**
- metric_type must be supported metric
- metric_value must be non-negative
- threshold must be between 0.0 and 1.0 if present
- For mAP metrics, threshold should be specified

### Deployment

Represents deployed model instances with API endpoints and monitoring.

**Fields:**
- `id` (UUID, PK) - Unique identifier
- `model_id` (UUID, FK to Model, required) - Deployed model
- `endpoint_url` (String, required) - API endpoint URL
- `version` (String, required) - Deployment version
- `status` (Enum, required) - deploying/active/inactive/failed
- `configuration` (JSONB, required) - Deployment settings
- `performance_monitoring` (JSONB, optional) - Usage and performance metrics
- `created_at` (DateTime, required) - Deployment timestamp
- `updated_at` (DateTime, required) - Last update timestamp
- `metadata` (JSONB, optional) - Additional deployment data

**Relationships:**
- Many-to-one with Model

**Validation Rules:**
- endpoint_url must be valid URL format
- version must follow semantic versioning
- status transitions: deploying → active/failed, active ↔ inactive
- configuration must contain required deployment parameters

**State Transitions:**
- Deploying → Active (on successful deployment)
- Deploying → Failed (on deployment error)
- Active ↔ Inactive (manual control)
- Any → Deploying (on redeployment)

## Database Schema Considerations

### Indexes
- Image: filename, dataset_split, upload_timestamp
- Annotation: image_id, model_id, creation_method, created_at
- Model: name, type, framework, training_status
- Training Job: status, created_at, base_model_id
- Inference Job: status, created_at, model_id
- Performance Metric: model_id, metric_type, evaluation_timestamp
- Deployment: model_id, status, endpoint_url

### Constraints
- Unique constraints on model name+version combinations
- Foreign key constraints with appropriate cascade behaviors
- Check constraints for enum values and numeric ranges
- JSON schema validation for JSONB fields

### Partitioning Strategy
- Performance Metric table partitioned by evaluation_timestamp (monthly)
- Training/Inference Job logs archived after completion
- Image files stored with content-based addressing

This data model supports all 40 functional requirements while maintaining referential integrity and providing efficient query patterns for the ML evaluation workflow.