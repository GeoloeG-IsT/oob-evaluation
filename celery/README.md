# ML Evaluation Platform - Celery Workers

Asynchronous task processing for ML operations including model training, batch inference, evaluation, and deployment.

## Features

- **Training Worker**: Model fine-tuning with progress tracking
- **Inference Worker**: Batch image processing 
- **Evaluation Worker**: Performance metrics calculation
- **Deployment Worker**: Model endpoint deployment

## Setup

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp ../.env.local .env

# Start workers (separate terminals)
celery -A src.celery_app worker --loglevel=info --queue=training
celery -A src.celery_app worker --loglevel=info --queue=inference  
celery -A src.celery_app worker --loglevel=info --queue=evaluation
celery -A src.celery_app worker --loglevel=info --queue=deployment

# Start flower monitoring (optional)
celery -A src.celery_app flower
```

### Production

```bash
# Start all workers with concurrency
celery -A src.celery_app worker --loglevel=info --concurrency=4

# Or start specific queues
celery -A src.celery_app worker --loglevel=info --queues=training,inference
```

## Project Structure

```
celery/
├── src/
│   ├── workers/         # Task workers
│   ├── tasks/           # Task definitions
│   └── celery_app.py    # Celery configuration
├── tests/
│   └── test_workers.py  # Worker tests
└── requirements.txt     # Dependencies
```

## Task Queues

- **training**: Long-running model training tasks
- **inference**: Batch image processing tasks
- **evaluation**: Performance calculation tasks  
- **deployment**: Model deployment tasks

## Monitoring

Access Flower monitoring at `http://localhost:5555` when running flower.