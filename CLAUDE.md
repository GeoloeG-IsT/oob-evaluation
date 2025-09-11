# Claude Code Context: ML Evaluation Platform

## Project Overview
A web application for evaluating Object Detection and Segmentation models (YOLO11/12, RT-DETR, SAM2) with capabilities for image upload, annotation, model inference, training/fine-tuning, performance evaluation, and model deployment.

## Tech Stack
- **Frontend**: Next.js 14, React 18, TypeScript
- **Backend**: FastAPI, Python 3.11+
- **Database**: Supabase/PostgreSQL with pgvector
- **Task Queue**: Celery with Redis
- **Containerization**: Docker + Docker Compose
- **Deployment**: GCP Cloud Run
- **ML Frameworks**: YOLO11/12, RT-DETR, SAM2

## Project Structure
```
frontend/           # Next.js React application
backend/           # FastAPI Python application
celery/            # Async task processing
docker/            # Container configurations
.env, .env.local, .env.development
```

## Core Entities
1. **Image** - Uploaded files with metadata and dataset splits
2. **Annotation** - Object/segment annotations (user or model-generated)
3. **Model** - AI models (detection/segmentation) with variants
4. **Dataset** - Collections organized by train/val/test splits
5. **Training Job** - Model fine-tuning tasks with progress tracking
6. **Inference Job** - Batch processing tasks
7. **Performance Metric** - Evaluation results (mAP, IoU, etc.)
8. **Deployment** - Model API endpoints with monitoring

## Key Features
- **Upload & Organization**: Images organized in train/val/test folders
- **Manual Annotation**: Drawing tools for bounding boxes and segments
- **Assisted Annotation**: Pre-trained model suggestions (SAM2)
- **Model Inference**: Real-time and batch processing
- **Performance Evaluation**: mAP, IoU, precision, recall, F1, execution time
- **Model Training**: Fine-tuning with progress monitoring
- **Model Deployment**: REST API endpoints with versioning
- **Data Export**: Standard formats (COCO, YOLO, Pascal VOC)

## Model Variants Supported
- **YOLO11**: nano, small, medium, large, extra-large
- **YOLO12**: nano, small, medium, large, extra-large  
- **RT-DETR**: R18, R34, R50, R101, RF-DETR Nano/Small/Medium
- **SAM2**: Tiny, Small, Base+, Large

## API Endpoints (Key)
- `POST /api/v1/images` - Upload images
- `POST /api/v1/annotations` - Create manual annotations
- `POST /api/v1/annotations/assisted` - Generate assisted annotations
- `POST /api/v1/inference/single` - Single image inference
- `POST /api/v1/inference/batch` - Batch inference
- `POST /api/v1/training/jobs` - Start model training
- `POST /api/v1/evaluation/metrics` - Calculate performance metrics
- `POST /api/v1/deployments` - Deploy model endpoints

## Development Workflow
1. **TDD Approach**: Contract → Integration → E2E → Unit tests
2. **Libraries**: Every feature implemented as library with CLI
3. **Real Dependencies**: Actual PostgreSQL, file system (no mocks)
4. **Version Control**: Semantic versioning with BUILD increments

## Performance Requirements
- Real-time inference monitoring
- Batch processing support
- Unlimited concurrent users
- Unlimited file sizes
- Indefinite data retention
- No authentication required

## Recent Changes
1. Added model deployment functionality (FR-036 to FR-040)
2. Completed specification with 40 functional requirements
3. Generated API contracts and data model
4. Created quickstart validation guide

## Constitutional Principles
- **Simplicity**: 3 projects max (frontend, backend, celery)
- **Testing**: RED-GREEN-Refactor cycle enforced
- **Architecture**: Features as libraries with CLI interfaces
- **Observability**: Structured logging with unified streams
- **Versioning**: 1.0.0 with BUILD increments on changes

## Key Constraints
- Support unlimited image formats including large TIFFs
- Handle memory-intensive ML operations efficiently
- Provide real-time progress monitoring for long-running tasks
- Maintain data integrity across concurrent operations
- Error handling for unsupported formats, memory issues, training failures

## Next Steps
Ready for task generation and implementation following TDD principles with contract tests, data model creation, and feature library development.