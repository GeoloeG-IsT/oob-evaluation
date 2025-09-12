#!/usr/bin/env python3
"""
Training Pipeline CLI - Main entry point for model training operations.

This CLI provides a unified command-line interface for training machine learning models
in the ML Evaluation Platform. It imports and extends the library CLI functionality
with additional features for production training workflows.

Usage:
    python -m backend.src.cli.training_cli --help
    python training_cli.py train --model yolo11_nano --dataset /path/to/data --epochs 100
    python training_cli.py resume job_id_here --epochs 50
    python training_cli.py optimize --model yolo11_small --trials 20 --parallel 4
    python training_cli.py monitor --live --experiment my_experiment
    python training_cli.py export job_id_here --format onnx
"""

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import signal
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Import library CLI functionality
    from backend.src.lib.training_pipeline.cli import (
        train_model as lib_train_model,
        show_job_status as lib_show_job_status,
        list_jobs as lib_list_jobs,
        start_hyperparameter_optimization as lib_start_optimization,
        show_monitoring_dashboard as lib_show_monitoring_dashboard,
        validate_dataset_command as lib_validate_dataset,
        create_training_config
    )
    
    # Import additional library components
    from backend.src.lib.training_pipeline.pipeline import (
        TrainingPipeline,
        TrainingConfig,
        HyperParameters,
        TrainingStatus,
        TrainingMetrics
    )
    from backend.src.lib.training_pipeline.jobs import TrainingJobManager, JobPriority
    from backend.src.lib.training_pipeline.datasets import DatasetManager, DatasetConfig, DatasetFormat
    from backend.src.lib.training_pipeline.optimizers import HyperParameterOptimizer, OptimizationConfig
    from backend.src.lib.training_pipeline.monitoring import TrainingMonitor
    from backend.src.lib.training_pipeline.export import ModelExporter, ExportFormat, ExportConfig
    from backend.src.lib.ml_models import ModelType, ModelVariant
    
    logger.info("Successfully imported Training Pipeline library components")
    
except ImportError as e:
    logger.error(f"Failed to import Training Pipeline library: {e}")
    print(f"Error: Failed to import Training Pipeline library: {e}")
    print("Please ensure the backend.src.lib.training_pipeline package is properly installed.")
    sys.exit(1)


class TrainingCLI:
    """Enhanced CLI for Training Pipeline with production features."""
    
    def __init__(self):
        self.verbose = False
        self.config_file = None
        self.shutdown_event = threading.Event()
        self.job_manager = TrainingJobManager()
        self.monitor = TrainingMonitor()
        
    def set_verbosity(self, verbose: bool):
        """Set verbosity level."""
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled")
    
    def load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file if provided."""
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
                return config
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                return {}
        return {}
    
    def resume_training(self, args) -> None:
        """Resume training from checkpoint."""
        print(f"Resuming training job: {args.job_id}")
        
        try:
            # Get job details
            job_details = self.job_manager.get_job_details(args.job_id)
            if not job_details:
                print(f"Job {args.job_id} not found")
                sys.exit(1)
            
            # Check if job can be resumed
            if job_details.get('status') not in [TrainingStatus.PAUSED, TrainingStatus.FAILED]:
                print(f"Job cannot be resumed from status: {job_details.get('status')}")
                sys.exit(1)
            
            # Create new training config based on existing job
            original_config = job_details.get('training_config')
            if not original_config:
                print("Original training configuration not found")
                sys.exit(1)
            
            # Update config with new parameters
            config = TrainingConfig(**original_config)
            
            if args.epochs:
                config.hyperparameters.epochs = args.epochs
            if args.lr:
                config.hyperparameters.learning_rate = args.lr
            if args.batch_size:
                config.hyperparameters.batch_size = args.batch_size
            
            # Set resume checkpoint
            checkpoint_path = job_details.get('latest_checkpoint')
            if checkpoint_path:
                config.resume_from_checkpoint = checkpoint_path
                print(f"Resuming from checkpoint: {checkpoint_path}")
            
            # Create new job with updated config
            new_job_id = self.job_manager.create_and_submit_job(
                config,
                JobPriority.HIGH  # High priority for resumed jobs
            )
            
            print(f"Created new job for resumed training: {new_job_id}")
            
            if not args.async_mode:
                self._wait_for_completion(new_job_id)
        
        except Exception as e:
            logger.error(f"Resume training failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Resume training failed: {str(e)}")
            sys.exit(1)
    
    def schedule_training(self, args) -> None:
        """Schedule training to run at specified time."""
        print(f"Scheduling training for: {args.schedule_time}")
        
        try:
            # Parse schedule time
            schedule_time = datetime.fromisoformat(args.schedule_time)
            current_time = datetime.now()
            
            if schedule_time <= current_time:
                print("Scheduled time must be in the future")
                sys.exit(1)
            
            # Create training configuration
            config = create_training_config(args)
            
            # Schedule the job
            job_id = self.job_manager.schedule_job(
                config,
                schedule_time,
                JobPriority.NORMAL
            )
            
            print(f"Training job scheduled: {job_id}")
            print(f"Will start at: {schedule_time}")
            
            delay_seconds = (schedule_time - current_time).total_seconds()
            print(f"Delay: {delay_seconds / 60:.1f} minutes")
        
        except Exception as e:
            logger.error(f"Schedule training failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Schedule training failed: {str(e)}")
            sys.exit(1)
    
    def export_model(self, args) -> None:
        """Export trained model to specified format."""
        print(f"Exporting model from job: {args.job_id}")
        
        try:
            # Get job details
            job_details = self.job_manager.get_job_details(args.job_id)
            if not job_details:
                print(f"Job {args.job_id} not found")
                sys.exit(1)
            
            if job_details.get('status') != TrainingStatus.COMPLETED:
                print(f"Job must be completed to export model. Current status: {job_details.get('status')}")
                sys.exit(1)
            
            # Get best model path
            best_model_path = job_details.get('best_model_path')
            if not best_model_path or not Path(best_model_path).exists():
                print("Best model file not found")
                sys.exit(1)
            
            # Create export configuration
            export_config = ExportConfig(
                source_model_path=best_model_path,
                output_path=args.output,
                format=ExportFormat(args.format.lower()),
                optimize_for_inference=args.optimize,
                quantize=args.quantize,
                include_preprocessing=args.include_preprocessing,
                batch_size=args.batch_size or 1,
                input_shape=args.input_shape or [640, 640]
            )
            
            # Export model
            exporter = ModelExporter()
            export_result = exporter.export_model(export_config)
            
            if export_result.success:
                print(f"Model exported successfully to: {export_result.output_path}")
                print(f"Export format: {args.format}")
                print(f"File size: {export_result.file_size_mb:.1f} MB")
                
                if export_result.benchmark_results:
                    print(f"Inference time: {export_result.benchmark_results.get('avg_inference_time_ms', 0):.2f}ms")
            else:
                print(f"Export failed: {export_result.error_message}")
                sys.exit(1)
        
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Model export failed: {str(e)}")
            sys.exit(1)
    
    def create_experiment(self, args) -> None:
        """Create a new training experiment with multiple configurations."""
        print(f"Creating experiment: {args.experiment_name}")
        
        try:
            experiment_config = {
                "experiment_name": args.experiment_name,
                "description": args.description or f"Experiment created on {datetime.now()}",
                "base_model": args.model,
                "dataset": args.dataset,
                "variations": []
            }
            
            # Create variations based on parameters
            base_config = create_training_config(args)
            
            if args.vary_lr:
                lr_values = [0.0001, 0.001, 0.01] if not args.lr_values else [float(x) for x in args.lr_values.split(',')]
                for lr in lr_values:
                    config = base_config.copy()
                    config.hyperparameters.learning_rate = lr
                    config.experiment_name = f"{args.experiment_name}_lr_{lr}"
                    experiment_config["variations"].append(config)
            
            if args.vary_batch_size:
                batch_sizes = [8, 16, 32] if not args.batch_size_values else [int(x) for x in args.batch_size_values.split(',')]
                for batch_size in batch_sizes:
                    config = base_config.copy()
                    config.hyperparameters.batch_size = batch_size
                    config.experiment_name = f"{args.experiment_name}_bs_{batch_size}"
                    experiment_config["variations"].append(config)
            
            if not experiment_config["variations"]:
                # Single configuration experiment
                experiment_config["variations"].append(base_config)
            
            # Submit all jobs
            job_ids = []
            for i, config in enumerate(experiment_config["variations"]):
                job_id = self.job_manager.create_and_submit_job(
                    config,
                    JobPriority.NORMAL
                )
                job_ids.append(job_id)
                print(f"Created job {i+1}/{len(experiment_config['variations'])}: {job_id}")
            
            # Save experiment metadata
            experiment_file = Path(args.output_dir) / f"{args.experiment_name}_experiment.json"
            experiment_file.parent.mkdir(parents=True, exist_ok=True)
            
            experiment_config["job_ids"] = job_ids
            experiment_config["created_at"] = datetime.now().isoformat()
            
            with open(experiment_file, 'w') as f:
                json.dump(experiment_config, f, indent=2, default=str)
            
            print(f"Experiment created with {len(job_ids)} jobs")
            print(f"Experiment metadata saved to: {experiment_file}")
        
        except Exception as e:
            logger.error(f"Create experiment failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Create experiment failed: {str(e)}")
            sys.exit(1)
    
    def compare_experiments(self, args) -> None:
        """Compare results from multiple experiments."""
        print("Comparing experiment results...")
        
        try:
            experiment_results = {}
            
            # Load experiment files
            for exp_file in args.experiment_files:
                exp_path = Path(exp_file)
                if not exp_path.exists():
                    print(f"Experiment file not found: {exp_file}")
                    continue
                
                with open(exp_path, 'r') as f:
                    exp_data = json.load(f)
                
                exp_name = exp_data.get('experiment_name', exp_path.stem)
                experiment_results[exp_name] = {
                    "metadata": exp_data,
                    "results": []
                }
                
                # Get results for each job
                for job_id in exp_data.get('job_ids', []):
                    job_details = self.job_manager.get_job_details(job_id)
                    if job_details and job_details.get('status') == TrainingStatus.COMPLETED:
                        experiment_results[exp_name]["results"].append(job_details)
            
            # Display comparison
            if args.format == "json":
                print(json.dumps(experiment_results, indent=2, default=str))
            else:
                self._display_experiment_comparison(experiment_results, args.metric)
        
        except Exception as e:
            logger.error(f"Compare experiments failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Compare experiments failed: {str(e)}")
            sys.exit(1)
    
    def _wait_for_completion(self, job_id: str):
        """Wait for job completion with progress updates."""
        print("Waiting for training to complete...")
        
        while True:
            progress = self.job_manager.get_job_progress(job_id)
            if not progress:
                print("Job not found")
                break
            
            print(f"\rProgress: {progress.progress_percentage:.1f}% "
                  f"(Epoch {progress.current_epoch}/{progress.total_epochs}) "
                  f"Loss: {progress.latest_train_loss:.4f} "
                  f"mAP: {progress.latest_map50:.4f}", end="")
            
            if progress.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                print(f"\nTraining {progress.status.lower()}")
                
                if progress.status == TrainingStatus.FAILED:
                    job_details = self.job_manager.get_job_details(job_id)
                    if job_details and job_details.get('error_message'):
                        print(f"Error: {job_details['error_message']}")
                break
            
            time.sleep(2)
    
    def _display_experiment_comparison(self, results: Dict[str, Any], metric: str):
        """Display experiment comparison results."""
        print("\nExperiment Comparison Results")
        print("=" * 80)
        
        # Extract comparison data
        comparison_data = []
        
        for exp_name, exp_data in results.items():
            for result in exp_data["results"]:
                best_metrics = result.get("best_metrics")
                if best_metrics:
                    comparison_data.append({
                        "experiment": exp_name,
                        "job_id": result.get("job_id", "")[:8],
                        "config": result.get("experiment_name", ""),
                        "map50": best_metrics.get("map50", 0),
                        "map50_95": best_metrics.get("map50_95", 0),
                        "training_time": result.get("training_time_hours", 0),
                        "final_loss": result.get("final_train_loss", 0)
                    })
        
        # Sort by selected metric
        if metric == "map50":
            comparison_data.sort(key=lambda x: x["map50"], reverse=True)
        elif metric == "map50_95":
            comparison_data.sort(key=lambda x: x["map50_95"], reverse=True)
        elif metric == "training_time":
            comparison_data.sort(key=lambda x: x["training_time"])
        else:
            comparison_data.sort(key=lambda x: x["final_loss"])
        
        # Display table
        print(f"{'Experiment':<20} {'Job':<8} {'Config':<25} {'mAP@0.5':<8} {'mAP@0.5:0.95':<12} {'Time (h)':<8} {'Loss':<8}")
        print("-" * 95)
        
        for data in comparison_data:
            print(f"{data['experiment']:<20} {data['job_id']:<8} {data['config']:<25} "
                  f"{data['map50']:<8.3f} {data['map50_95']:<12.3f} "
                  f"{data['training_time']:<8.1f} {data['final_loss']:<8.4f}")


def main():
    """Main CLI entry point with enhanced functionality."""
    # Create CLI instance
    cli = TrainingCLI()
    
    # Main parser
    parser = argparse.ArgumentParser(
        description="Training Pipeline CLI - Unified command-line interface for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --model yolo11_nano --dataset data/ --epochs 100 --batch-size 16
  %(prog)s resume job_12345 --epochs 50 --lr 0.0001
  %(prog)s optimize --model yolo11_small --dataset data/ --trials 20 --parallel 4
  %(prog)s schedule --model yolo11_nano --dataset data/ --schedule-time "2024-01-15T22:00:00"
  %(prog)s export job_12345 --format onnx --output model.onnx --optimize
  %(prog)s experiment --name "lr_comparison" --model yolo11_nano --dataset data/ --vary-lr
        """
    )
    
    # Global arguments
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--format", choices=["json", "table"], default="table",
                       help="Output format (default: table)")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command (uses library function but enhanced)
    train_parser = subparsers.add_parser("train", help="Start model training")
    train_parser.add_argument("--model", required=True,
                             help="Model to train (e.g., yolo11_nano, yolo12_small)")
    train_parser.add_argument("--dataset", required=True,
                             help="Path to dataset")
    train_parser.add_argument("--dataset-format", default="coco",
                             choices=["coco", "yolo", "pascal_voc", "custom"],
                             help="Dataset format")
    train_parser.add_argument("--epochs", type=int, default=100,
                             help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16,
                             help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001,
                             help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=0.0005,
                             help="Weight decay")
    train_parser.add_argument("--optimizer", default="adam",
                             choices=["adam", "sgd", "rmsprop"],
                             help="Optimizer")
    train_parser.add_argument("--patience", type=int, default=10,
                             help="Early stopping patience")
    train_parser.add_argument("--num-classes", type=int, default=80,
                             help="Number of classes")
    train_parser.add_argument("--output-dir", default="./training_outputs",
                             help="Output directory")
    train_parser.add_argument("--experiment-name",
                             help="Experiment name")
    train_parser.add_argument("--device", default="auto",
                             choices=["auto", "cpu", "cuda", "mps"],
                             help="Training device")
    train_parser.add_argument("--resume", help="Resume from checkpoint")
    train_parser.add_argument("--no-mixed-precision", action="store_true",
                             help="Disable mixed precision training")
    train_parser.add_argument("--no-augment", action="store_true",
                             help="Disable data augmentation")
    train_parser.add_argument("--no-checkpoints", action="store_true",
                             help="Disable checkpoint saving")
    train_parser.add_argument("--tensorboard", action="store_true",
                             help="Enable TensorBoard logging")
    train_parser.add_argument("--validate-dataset", action="store_true",
                             help="Validate dataset before training")
    train_parser.add_argument("--force", action="store_true",
                             help="Force training despite validation errors")
    train_parser.add_argument("--use-scheduler", action="store_true",
                             help="Use job scheduler")
    train_parser.add_argument("--monitor", action="store_true",
                             help="Enable training monitoring")
    train_parser.add_argument("--async", dest="async_mode", action="store_true",
                             help="Submit job and exit (don't wait)")
    
    # Resume command (new)
    resume_parser = subparsers.add_parser("resume", help="Resume training from checkpoint")
    resume_parser.add_argument("job_id", help="Job ID to resume")
    resume_parser.add_argument("--epochs", type=int, help="Override epoch count")
    resume_parser.add_argument("--lr", type=float, help="Override learning rate")
    resume_parser.add_argument("--batch-size", type=int, help="Override batch size")
    resume_parser.add_argument("--async", dest="async_mode", action="store_true",
                              help="Submit job and exit")
    
    # Schedule command (new)
    schedule_parser = subparsers.add_parser("schedule", help="Schedule training job")
    schedule_parser.add_argument("--schedule-time", required=True,
                                help="Schedule time (ISO format: 2024-01-15T22:00:00)")
    schedule_parser.add_argument("--model", required=True, help="Model to train")
    schedule_parser.add_argument("--dataset", required=True, help="Dataset path")
    schedule_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    schedule_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    schedule_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    schedule_parser.add_argument("--dataset-format", default="coco", help="Dataset format")
    schedule_parser.add_argument("--num-classes", type=int, default=80, help="Number of classes")
    schedule_parser.add_argument("--output-dir", default="./training_outputs", help="Output directory")
    schedule_parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    schedule_parser.add_argument("--optimizer", default="adam", help="Optimizer")
    schedule_parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    schedule_parser.add_argument("--device", default="auto", help="Training device")
    
    # Export command (new)
    export_parser = subparsers.add_parser("export", help="Export trained model")
    export_parser.add_argument("job_id", help="Training job ID")
    export_parser.add_argument("--output", required=True, help="Output file path")
    export_parser.add_argument("--format", required=True,
                              choices=["onnx", "torchscript", "tflite", "coreml"],
                              help="Export format")
    export_parser.add_argument("--optimize", action="store_true",
                              help="Optimize for inference")
    export_parser.add_argument("--quantize", action="store_true",
                              help="Apply quantization")
    export_parser.add_argument("--include-preprocessing", action="store_true",
                              help="Include preprocessing in export")
    export_parser.add_argument("--batch-size", type=int, default=1,
                              help="Batch size for export")
    export_parser.add_argument("--input-shape", nargs=2, type=int,
                              help="Input shape (height width)")
    
    # Experiment command (new)
    experiment_parser = subparsers.add_parser("experiment", help="Create training experiment")
    experiment_parser.add_argument("--name", dest="experiment_name", required=True,
                                  help="Experiment name")
    experiment_parser.add_argument("--description", help="Experiment description")
    experiment_parser.add_argument("--model", required=True, help="Base model")
    experiment_parser.add_argument("--dataset", required=True, help="Dataset path")
    experiment_parser.add_argument("--output-dir", default="./experiments",
                                  help="Output directory")
    experiment_parser.add_argument("--vary-lr", action="store_true",
                                  help="Create variations with different learning rates")
    experiment_parser.add_argument("--lr-values", help="Comma-separated LR values")
    experiment_parser.add_argument("--vary-batch-size", action="store_true",
                                  help="Create variations with different batch sizes")
    experiment_parser.add_argument("--batch-size-values", help="Comma-separated batch sizes")
    experiment_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    experiment_parser.add_argument("--dataset-format", default="coco", help="Dataset format")
    experiment_parser.add_argument("--num-classes", type=int, default=80, help="Number of classes")
    experiment_parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    experiment_parser.add_argument("--optimizer", default="adam", help="Optimizer")
    experiment_parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    experiment_parser.add_argument("--device", default="auto", help="Training device")
    
    # Compare command (new)
    compare_parser = subparsers.add_parser("compare", help="Compare experiment results")
    compare_parser.add_argument("experiment_files", nargs="+",
                               help="Experiment JSON files to compare")
    compare_parser.add_argument("--metric", default="map50",
                               choices=["map50", "map50_95", "training_time", "loss"],
                               help="Primary comparison metric")
    
    # Status, jobs, optimize, monitor, validate commands (use library functions)
    status_parser = subparsers.add_parser("status", help="Show job status")
    status_parser.add_argument("job_id", help="Job ID")
    
    jobs_parser = subparsers.add_parser("jobs", help="List training jobs")
    jobs_parser.add_argument("--status", help="Filter by status")
    
    opt_parser = subparsers.add_parser("optimize", help="Hyperparameter optimization")
    opt_parser.add_argument("--model", required=True, help="Model to optimize")
    opt_parser.add_argument("--dataset", required=True, help="Path to dataset")
    opt_parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    opt_parser.add_argument("--parallel", type=int, default=2, help="Parallel trials")
    opt_parser.add_argument("--timeout", type=float, default=24.0, help="Timeout (hours)")
    opt_parser.add_argument("--optimizer-type", default="random_search",
                           choices=["random_search", "grid_search", "bayesian", "genetic"],
                           help="Optimization algorithm")
    opt_parser.add_argument("--objective", default="map50_95",
                           choices=["map50", "map50_95", "train_loss", "val_loss"],
                           help="Objective metric")
    opt_parser.add_argument("--optimize-lr", action="store_true", help="Optimize learning rate")
    opt_parser.add_argument("--optimize-batch", action="store_true", help="Optimize batch size")
    opt_parser.add_argument("--optimize-weight-decay", action="store_true", help="Optimize weight decay")
    opt_parser.add_argument("--export-results", action="store_true", help="Export results")
    opt_parser.add_argument("--epochs", type=int, default=50, help="Epochs per trial")
    opt_parser.add_argument("--dataset-format", default="coco", help="Dataset format")
    opt_parser.add_argument("--num-classes", type=int, default=80, help="Number of classes")
    opt_parser.add_argument("--output-dir", default="./training_outputs", help="Output directory")
    opt_parser.add_argument("--batch-size", type=int, default=16, help="Base batch size")
    opt_parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate")
    opt_parser.add_argument("--weight-decay", type=float, default=0.0005, help="Base weight decay")
    opt_parser.add_argument("--optimizer", default="adam", help="Base optimizer")
    opt_parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    opt_parser.add_argument("--device", default="auto", help="Training device")
    
    monitor_parser = subparsers.add_parser("monitor", help="Show monitoring dashboard")
    monitor_parser.add_argument("--live", action="store_true", help="Live monitoring mode")
    monitor_parser.add_argument("--refresh-interval", type=int, default=5,
                               help="Refresh interval for live mode (seconds)")
    monitor_parser.add_argument("--experiment", help="Filter by experiment name")
    
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("dataset_path", help="Path to dataset")
    validate_parser.add_argument("--format", default="coco",
                                choices=["coco", "yolo", "pascal_voc", "custom"],
                                help="Dataset format")
    validate_parser.add_argument("--num-classes", type=int, help="Number of classes")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up CLI
    cli.set_verbosity(args.verbose)
    config = cli.load_config(args.config)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Route to appropriate command
        if args.command == "train":
            asyncio.run(lib_train_model(args))
        
        elif args.command == "resume":
            cli.resume_training(args)
        
        elif args.command == "schedule":
            cli.schedule_training(args)
        
        elif args.command == "export":
            cli.export_model(args)
        
        elif args.command == "experiment":
            cli.create_experiment(args)
        
        elif args.command == "compare":
            cli.compare_experiments(args)
        
        elif args.command == "status":
            lib_show_job_status(args)
        
        elif args.command == "jobs":
            lib_list_jobs(args)
        
        elif args.command == "optimize":
            lib_start_optimization(args)
        
        elif args.command == "monitor":
            if hasattr(args, 'live') and args.live:
                # Live monitoring implementation
                try:
                    while True:
                        import os
                        os.system('cls' if os.name == 'nt' else 'clear')
                        lib_show_monitoring_dashboard(args)
                        print(f"\nRefreshing every {args.refresh_interval}s... (Press Ctrl+C to exit)")
                        time.sleep(args.refresh_interval)
                except KeyboardInterrupt:
                    print("\nMonitoring stopped")
            else:
                lib_show_monitoring_dashboard(args)
        
        elif args.command == "validate":
            lib_validate_dataset(args)
        
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        cli.shutdown_event.set()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            traceback.print_exc()
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()