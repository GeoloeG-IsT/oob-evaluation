# ML Evaluation Platform - Standalone CLI Tools

This directory contains the main entry point CLI files for the ML Evaluation Platform. These standalone CLI tools provide unified command-line interfaces for all platform operations.

## Available CLI Tools

### 1. ML Models CLI (`ml_models_cli.py`)
**Purpose**: Manage machine learning models, run predictions, and perform benchmarking.

**Key Features**:
- List and manage available models
- Load/unload models from memory  
- Run single and batch predictions
- Benchmark model performance
- Compare multiple models
- System information and monitoring

**Usage Examples**:
```bash
# List all available models
python ml_models_cli.py list --type yolo11

# Load a model and show info
python ml_models_cli.py load yolo11_nano --info

# Run prediction on single image
python ml_models_cli.py predict yolo11_nano image.jpg --confidence 0.7

# Run batch prediction
python ml_models_cli.py batch-predict yolo11_nano "images/*.jpg" --output results.json

# Benchmark model performance
python ml_models_cli.py benchmark yolo11_nano --images "test/*.jpg" --iterations 10

# Compare models
python ml_models_cli.py compare yolo11_nano yolo12_small --metric speed
```

### 2. Inference CLI (`inference_cli.py`)
**Purpose**: Run inference operations including single, batch, and pipeline processing.

**Key Features**:
- Single image inference
- Batch processing with progress tracking
- Pipeline processing with directory watching
- Inference server for REST API access
- Performance monitoring and metrics export
- Log streaming and job management

**Usage Examples**:
```bash
# Single inference
python inference_cli.py single yolo11_nano image.jpg --confidence 0.7

# Batch processing
python inference_cli.py batch yolo11_nano "images/*.jpg" --async --output results.json

# Pipeline with directory watching
python inference_cli.py pipeline yolo11_nano input_dir output_dir --watch

# Start inference server
python inference_cli.py serve yolo11_nano --port 8080 --max-concurrent 10

# Monitor performance (live)
python inference_cli.py monitor --live --model-id yolo11_nano

# Stream inference logs
python inference_cli.py logs --interval 2

# Export performance metrics
python inference_cli.py export --output metrics.json --time-window 60
```

### 3. Training CLI (`training_cli.py`)
**Purpose**: Manage model training, hyperparameter optimization, and experiment tracking.

**Key Features**:
- Model training with comprehensive configuration
- Resume training from checkpoints
- Schedule training jobs
- Hyperparameter optimization
- Experiment management and comparison
- Model export to multiple formats
- Training monitoring and progress tracking

**Usage Examples**:
```bash
# Start training
python training_cli.py train --model yolo11_nano --dataset data/ --epochs 100 --batch-size 16

# Resume training from checkpoint
python training_cli.py resume job_12345 --epochs 50 --lr 0.0001

# Schedule training for later
python training_cli.py schedule --model yolo11_nano --dataset data/ --schedule-time "2024-01-15T22:00:00"

# Hyperparameter optimization
python training_cli.py optimize --model yolo11_small --dataset data/ --trials 20 --parallel 4

# Create training experiment
python training_cli.py experiment --name "lr_comparison" --model yolo11_nano --dataset data/ --vary-lr

# Export trained model
python training_cli.py export job_12345 --format onnx --output model.onnx --optimize

# Compare experiments
python training_cli.py compare exp1.json exp2.json exp3.json --metric map50

# Live training monitoring
python training_cli.py monitor --live --experiment my_experiment
```

### 4. Annotation CLI (`annotation_cli.py`)
**Purpose**: Manage annotations including assistance, validation, conversion, and workflow automation.

**Key Features**:
- AI-assisted annotation with multiple models
- Batch annotation processing
- Format conversion between COCO, YOLO, Pascal VOC
- Annotation validation and quality checking
- Workflow automation and configuration
- Annotation merging and splitting
- Visualization and comprehensive auditing

**Usage Examples**:
```bash
# AI-assisted annotation
python annotation_cli.py assist --model sam2_base --image image.jpg --mode segmentation --output result.json

# Batch annotation processing
python annotation_cli.py batch "images/*.jpg" --assistant sam2 --output results/ --max-concurrent 8

# Convert annotation formats
python annotation_cli.py convert --input data.coco --output data.yolo --from coco --to yolo --show-stats

# Validate annotations
python annotation_cli.py validate annotations.json --format coco --quality-check --show-details

# Run automated workflow
python annotation_cli.py workflow --config workflow.json --input images/ --output annotations/

# Create workflow configuration
python annotation_cli.py create-workflow --output workflow.json --include-assistance --include-validation

# Merge multiple annotation files
python annotation_cli.py merge file1.json file2.json file3.json --output merged.json --show-stats

# Split annotations for train/val/test
python annotation_cli.py split annotations.json --output-dir splits/ --train-ratio 0.7 --val-ratio 0.2

# Create visualizations
python annotation_cli.py visualize annotations.json --images-dir images/ --output-dir viz/ --create-overlay --create-statistics

# Comprehensive audit
python annotation_cli.py audit annotations.json --output audit_report.json --check-duplicates
```

## Global Options

All CLI tools support these common options:

- `--verbose, -v`: Enable verbose output for debugging
- `--config CONFIG_FILE`: Load configuration from JSON file
- `--format {json,table}`: Output format (default: table)
- `--help`: Show detailed help for commands and options

## Configuration Files

You can create JSON configuration files to set default parameters:

```json
{
  "default_confidence": 0.7,
  "default_batch_size": 16,
  "output_directory": "./outputs",
  "enable_logging": true,
  "log_level": "INFO"
}
```

Use with: `python {cli_tool}.py --config config.json {command}`

## Integration with Library CLI

These standalone CLI tools import and extend the functionality from the library CLI modules:
- `/backend/src/lib/ml_models/cli.py`
- `/backend/src/lib/inference_engine/cli.py`  
- `/backend/src/lib/training_pipeline/cli.py`
- `/backend/src/lib/annotation_tools/cli.py`

The standalone versions add production features like:
- Enhanced error handling and logging
- Configuration file support
- Advanced workflow automation
- Performance monitoring
- Batch operations
- Server deployment capabilities

## Error Handling

All CLI tools include comprehensive error handling:
- Graceful keyboard interrupt (Ctrl+C) handling
- Detailed error messages with stack traces (in verbose mode)
- Input validation and helpful usage suggestions
- Automatic cleanup of resources on exit

## Performance Considerations

- Use `--async` flags for long-running operations
- Adjust `--max-concurrent` based on system resources
- Monitor memory usage for large batch operations  
- Use `--timeout` settings to prevent hung operations
- Enable progress tracking for visibility into long tasks

## Support and Troubleshooting

- Use `--verbose` flag for detailed debugging information
- Check log files in the output directories
- Verify model and dataset paths are correct
- Ensure sufficient system resources (CPU, memory, GPU)
- Review configuration file syntax and parameters