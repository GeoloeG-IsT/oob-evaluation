# ML Evaluation Platform Backend

FastAPI backend for the ML Evaluation Platform supporting object detection and segmentation model evaluation.

## Features

- RESTful API for ML model evaluation
- Support for YOLO11/12, RT-DETR, SAM2 models
- Image upload and annotation management
- Model training and fine-tuning
- Performance evaluation and metrics
- Model deployment as API endpoints
- Async task processing with Celery

## Setup

### Development

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp ../.env.development .env

# Run database migrations
alembic upgrade head

# Start development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests
pytest -m contract      # Contract tests

# Run with coverage
pytest --cov=src --cov-report=html
```

## Project Structure

```
backend/
├── src/
│   ├── models/          # Database models
│   ├── services/        # Business logic
│   ├── api/             # API endpoints
│   ├── lib/             # ML libraries
│   ├── cli/             # CLI commands
│   └── utils/           # Utilities
├── tests/
│   ├── contract/        # API contract tests
│   ├── integration/     # Integration tests
│   └── unit/            # Unit tests
└── migrations/          # Database migrations
```

## Architecture

- **FastAPI**: Web framework with automatic OpenAPI documentation
- **SQLAlchemy**: ORM for database operations
- **Alembic**: Database migration management
- **Celery**: Async task processing
- **PostgreSQL**: Database with pgvector for vector storage
- **Structured Logging**: JSON-formatted logs with correlation IDs

## ML Model Support

- **YOLO11/12**: Object detection with variants (nano, small, medium, large, xl)
- **RT-DETR**: Vision transformer-based detection
- **SAM2**: Advanced segmentation for images and videos
- **Custom Models**: Support for fine-tuned models
