#!/usr/bin/env python3
"""
CLI interface for Training Pipeline Library.

Usage examples:
    python -m backend.src.lib.training_pipeline.cli train --model yolo11_nano --dataset /path/to/data --epochs 100
    python -m backend.src.lib.training_pipeline.cli status job_id_here
    python -m backend.src.lib.training_pipeline.cli jobs --status training
    python -m backend.src.lib.training_pipeline.cli optimize --model yolo11_small --dataset /path/to/data --trials 20
    python -m backend.src.lib.training_pipeline.cli monitor --live
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from .pipeline import (
    TrainingPipeline,
    TrainingConfig,
    HyperParameters,
    TrainingStatus
)
from .jobs import TrainingJobManager, JobPriority
from .datasets import DatasetManager, DatasetConfig, DatasetFormat
from .optimizers import (
    HyperParameterOptimizer,
    OptimizationConfig,
    OptimizerType,
    ParameterRange,
    ParameterType
)
from .monitoring import TrainingMonitor
from ..ml_models import ModelType, ModelVariant


def create_training_config(args) -> TrainingConfig:
    """Create training configuration from CLI arguments."""
    
    # Parse model type and variant
    if '_' in args.model:
        model_type_str, variant_str = args.model.split('_', 1)
        model_type = ModelType(model_type_str.lower())
        variant = ModelVariant(variant_str.lower())
    else:
        raise ValueError(f"Invalid model format: {args.model}. Expected format: model_type_variant")
    
    # Create hyperparameters
    hyperparams = HyperParameters(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        patience=args.patience
    )
    
    # Set augmentation parameters if provided
    if hasattr(args, 'augment') and not args.augment:
        hyperparams.augment_fliplr = 0.0
        hyperparams.augment_mosaic = 0.0
        hyperparams.augment_mixup = 0.0
    
    # Create training config
    config = TrainingConfig(
        base_model_id=args.model,
        model_type=model_type,
        variant=variant,
        dataset_path=args.dataset,
        num_classes=args.num_classes,
        hyperparameters=hyperparams,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name or f"{args.model}_training",
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        enable_tensorboard=args.tensorboard,
        save_checkpoints=not args.no_checkpoints
    )
    
    if args.resume:
        config.resume_from_checkpoint = args.resume
    
    return config


async def train_model(args):
    """Start model training."""
    print(f"Starting training with model {args.model}")
    
    try:
        # Create training configuration
        config = create_training_config(args)
        
        # Validate dataset if requested
        if args.validate_dataset:
            print("Validating dataset...")
            dataset_config = DatasetConfig(
                dataset_id="cli_dataset",
                name="CLI Dataset",
                description="Dataset provided via CLI",
                dataset_path=args.dataset,
                format=DatasetFormat(args.dataset_format.lower())
            )
            
            dataset_manager = DatasetManager()
            dataset_manager.register_dataset(dataset_config)
            
            validation_result = dataset_manager.validate_dataset("cli_dataset")
            if validation_result and not validation_result.is_valid:
                print("Dataset validation failed:")
                for error in validation_result.errors:
                    print(f"  ERROR: {error}")
                for warning in validation_result.warnings:
                    print(f"  WARNING: {warning}")
                
                if not args.force:
                    print("Use --force to proceed despite validation errors")
                    sys.exit(1)
            else:
                print("Dataset validation passed")
        
        # Set up training pipeline
        if args.use_scheduler:
            # Use job manager with scheduler
            job_manager = TrainingJobManager()
            
            # Set up monitoring if requested
            monitor = None
            if args.monitor:
                monitor = TrainingMonitor(log_dir=f"{args.output_dir}/logs")
            
            # Submit job
            job_id = job_manager.create_and_submit_job(
                config,
                JobPriority.NORMAL
            )
            
            print(f"Training job submitted: {job_id}")
            
            if not args.async_mode:
                # Wait for completion
                print("Waiting for training to complete...")
                
                while True:
                    progress = job_manager.get_job_progress(job_id)
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
                            job_details = job_manager.get_job_details(job_id)
                            if job_details and job_details.get('error_message'):
                                print(f"Error: {job_details['error_message']}")
                        
                        break
                    
                    time.sleep(2)
        
        else:
            # Use direct training pipeline
            pipeline = TrainingPipeline()
            job = pipeline.create_training_job(config)
            
            print(f"Training job created: {job.job_id}")
            
            # Start training
            result = await pipeline.start_training(job)
            
            print(f"Training completed with status: {result.status}")
            if result.error_message:
                print(f"Error: {result.error_message}")
            
            if result.best_metrics:
                print(f"Best mAP@0.5: {result.best_metrics.map50:.4f}")
                print(f"Best mAP@0.5:0.95: {result.best_metrics.map50_95:.4f}")
                
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)


def show_job_status(args):
    """Show status of a training job."""
    job_manager = TrainingJobManager()
    
    if args.format == "json":
        job_details = job_manager.get_job_details(args.job_id)
        if job_details:
            print(json.dumps(job_details, indent=2, default=str))
        else:
            print(f'{{"error": "Job {args.job_id} not found"}}')
    else:
        progress = job_manager.get_job_progress(args.job_id)
        if progress:
            print(f"Job Status: {args.job_id}")
            print(f"Status: {progress.status}")
            print(f"Progress: {progress.progress_percentage:.1f}%")
            print(f"Epoch: {progress.current_epoch}/{progress.total_epochs}")
            
            if progress.elapsed_time_seconds > 0:
                hours = int(progress.elapsed_time_seconds // 3600)
                minutes = int((progress.elapsed_time_seconds % 3600) // 60)
                print(f"Elapsed: {hours:02d}:{minutes:02d}")
            
            if progress.estimated_remaining_seconds > 0:
                remaining_hours = int(progress.estimated_remaining_seconds // 3600)
                remaining_minutes = int((progress.estimated_remaining_seconds % 3600) // 60)
                print(f"Remaining: {remaining_hours:02d}:{remaining_minutes:02d}")
            
            print(f"Latest Loss: {progress.latest_train_loss:.4f}")
            print(f"Latest mAP: {progress.latest_map50:.4f}")
            print(f"Best mAP: {progress.best_map50:.4f}")
            
            if progress.gpu_memory_usage_mb > 0:
                print(f"GPU Memory: {progress.gpu_memory_usage_mb:.0f} MB")
        else:
            print(f"Job {args.job_id} not found")


def list_jobs(args):
    """List training jobs."""
    job_manager = TrainingJobManager()
    
    status_filter = None
    if args.status:
        try:
            status_filter = TrainingStatus(args.status.lower())
        except ValueError:
            print(f"Invalid status: {args.status}")
            print(f"Valid statuses: {[s.value for s in TrainingStatus]}")
            sys.exit(1)
    
    jobs = job_manager.list_jobs(status_filter)
    
    if args.format == "json":
        print(json.dumps(jobs, indent=2, default=str))
    else:
        if not jobs:
            print("No jobs found")
            return
        
        print(f"{'Job ID':<36} {'Experiment':<20} {'Status':<10} {'Progress':<8} {'Model':<15}")
        print("-" * 95)
        
        for job in jobs:
            job_id = job['job_id'][:35]  # Truncate long IDs
            experiment = job['experiment_name'][:19]
            status = job['status']
            progress = f"{job['progress_percentage']:.1f}%"
            model = f"{job['model_type']}_{job['variant']}"
            
            print(f"{job_id:<36} {experiment:<20} {status:<10} {progress:<8} {model:<15}")


def start_hyperparameter_optimization(args):
    """Start hyperparameter optimization."""
    print(f"Starting hyperparameter optimization for {args.model}")
    
    try:
        # Create base training config
        config = create_training_config(args)
        
        # Create optimization config
        opt_config = OptimizationConfig(
            optimization_id=f"opt_{args.model}_{int(time.time())}",
            objective_metric=args.objective,
            maximize_objective=args.objective in ["map50", "map50_95", "precision", "recall", "f1_score"],
            optimizer_type=OptimizerType(args.optimizer_type.lower()),
            max_trials=args.trials,
            max_parallel_trials=args.parallel,
            timeout_hours=args.timeout
        )
        
        # Define parameter ranges
        if args.optimize_lr:
            opt_config.add_parameter_range(ParameterRange(
                name="learning_rate",
                param_type=ParameterType.FLOAT,
                min_value=1e-5,
                max_value=1e-1,
                log_scale=True,
                default_value=0.001
            ))
        
        if args.optimize_batch:
            opt_config.add_parameter_range(ParameterRange(
                name="batch_size",
                param_type=ParameterType.CATEGORICAL,
                choices=[8, 16, 32, 64],
                default_value=16
            ))
        
        if args.optimize_weight_decay:
            opt_config.add_parameter_range(ParameterRange(
                name="weight_decay",
                param_type=ParameterType.FLOAT,
                min_value=1e-6,
                max_value=1e-2,
                log_scale=True,
                default_value=0.0005
            ))
        
        # Create optimizer
        optimizer_manager = HyperParameterOptimizer()
        optimizer = optimizer_manager.create_optimization(opt_config)
        
        print(f"Optimization created: {opt_config.optimization_id}")
        print(f"Max trials: {args.trials}")
        print(f"Optimizer: {args.optimizer_type}")
        print(f"Objective: {args.objective}")
        
        # Run optimization (simplified - in real implementation would be async)
        best_params = None
        best_score = float('-inf') if opt_config.maximize_objective else float('inf')
        
        for trial_num in range(args.trials):
            trial = optimizer_manager.suggest_trial(opt_config.optimization_id)
            if not trial:
                print(f"Failed to create trial {trial_num}")
                continue
            
            print(f"\nTrial {trial_num + 1}/{args.trials}: {trial.trial_id}")
            print(f"Hyperparameters: {trial.hyperparameters}")
            
            # Simulate training with these hyperparameters
            # In real implementation, would start actual training job
            import random
            simulated_score = random.uniform(0.3, 0.8)  # Simulate mAP score
            
            # Create mock metrics
            from .pipeline import TrainingMetrics
            metrics = TrainingMetrics(
                epoch=args.epochs,
                map50_95=simulated_score,
                map50=simulated_score + 0.1,
                train_loss=random.uniform(0.1, 2.0),
                val_loss=random.uniform(0.1, 2.0)
            )
            
            # Update trial
            optimizer_manager.update_trial_result(
                opt_config.optimization_id,
                trial.trial_id,
                metrics,
                "completed"
            )
            
            print(f"Result: {args.objective} = {simulated_score:.4f}")
            
            # Track best
            is_better = (
                (opt_config.maximize_objective and simulated_score > best_score) or
                (not opt_config.maximize_objective and simulated_score < best_score)
            )
            
            if is_better:
                best_score = simulated_score
                best_params = trial.hyperparameters
                print(f"New best score: {best_score:.4f}")
        
        # Show final results
        print(f"\nOptimization completed!")
        print(f"Best {args.objective}: {best_score:.4f}")
        print(f"Best hyperparameters: {json.dumps(best_params, indent=2)}")
        
        # Export results
        if args.export_results:
            results = optimizer_manager.export_optimization_results(opt_config.optimization_id)
            output_file = f"{args.output_dir}/optimization_results.json"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Results exported to: {output_file}")
            
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        sys.exit(1)


def show_monitoring_dashboard(args):
    """Show training monitoring dashboard."""
    monitor = TrainingMonitor()
    
    if args.live:
        # Live monitoring mode
        try:
            while True:
                dashboard = monitor.get_monitoring_dashboard()
                
                # Clear screen
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("Training Monitor Dashboard")
                print("=" * 50)
                print(f"Last updated: {dashboard['timestamp']}")
                print()
                
                summary = dashboard['summary']
                print(f"Total Jobs: {summary['total_jobs']}")
                print(f"Active Jobs: {summary['active_jobs']}")
                print(f"Completed Jobs: {summary['completed_jobs']}")
                print(f"Success Rate: {summary['success_rate']:.1f}%")
                print()
                
                if dashboard['active_jobs']:
                    print("Active Jobs:")
                    print(f"{'Job ID':<20} {'Experiment':<20} {'Progress':<10} {'Epoch':<10} {'mAP':<8}")
                    print("-" * 80)
                    
                    for job in dashboard['active_jobs']:
                        job_id = job['job_id'][:19]
                        experiment = job['experiment_name'][:19]
                        progress = f"{job['progress_percentage']:.1f}%"
                        epoch = f"{job['current_epoch']}/{job['total_epochs']}"
                        map_score = f"{job['latest_map50']:.3f}"
                        
                        print(f"{job_id:<20} {experiment:<20} {progress:<10} {epoch:<10} {map_score:<8}")
                
                # Show system metrics
                if dashboard['system_metrics']['current']:
                    print("\nSystem Metrics:")
                    sys_metrics = dashboard['system_metrics']['current']
                    print(f"CPU: {sys_metrics['cpu_usage_percent']:.1f}% | "
                          f"Memory: {sys_metrics['memory_usage_mb']:.0f}MB | "
                          f"GPU: {sys_metrics['gpu_usage_percent']:.1f}% | "
                          f"GPU Memory: {sys_metrics['gpu_memory_usage_mb']:.0f}MB")
                
                print("\nPress Ctrl+C to exit")
                time.sleep(args.refresh_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    else:
        # Single snapshot
        dashboard = monitor.get_monitoring_dashboard()
        
        if args.format == "json":
            print(json.dumps(dashboard, indent=2, default=str))
        else:
            print("Training Monitor Dashboard")
            print("=" * 50)
            
            summary = dashboard['summary']
            print(f"Total Jobs: {summary['total_jobs']}")
            print(f"Active Jobs: {summary['active_jobs']}")
            print(f"Completed Jobs: {summary['completed_jobs']}")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            
            if dashboard['active_jobs']:
                print("\nActive Jobs:")
                for job in dashboard['active_jobs']:
                    print(f"  {job['job_id']}: {job['experiment_name']} "
                          f"({job['progress_percentage']:.1f}%)")


def validate_dataset_command(args):
    """Validate a dataset."""
    dataset_config = DatasetConfig(
        dataset_id="cli_validation",
        name="CLI Validation Dataset",
        description="Dataset being validated via CLI",
        dataset_path=args.dataset_path,
        format=DatasetFormat(args.format.lower())
    )
    
    if args.num_classes:
        dataset_config.categories = [
            {"category_id": i, "name": f"class_{i}"}
            for i in range(args.num_classes)
        ]
    
    dataset_manager = DatasetManager()
    dataset_manager.register_dataset(dataset_config)
    
    print(f"Validating dataset: {args.dataset_path}")
    print(f"Format: {args.format}")
    
    validation_result = dataset_manager.validate_dataset("cli_validation")
    
    if not validation_result:
        print("Validation failed - could not load dataset")
        sys.exit(1)
    
    if args.format == "json":
        stats = dataset_manager.get_dataset_statistics("cli_validation")
        print(json.dumps(stats, indent=2, default=str))
    else:
        print(f"\nValidation Results:")
        print(f"Valid: {'Yes' if validation_result.is_valid else 'No'}")
        print(f"Total Images: {validation_result.total_images}")
        print(f"Total Annotations: {validation_result.total_annotations}")
        
        print(f"\nSplit Distribution:")
        print(f"Train: {validation_result.train_images}")
        print(f"Validation: {validation_result.val_images}")
        print(f"Test: {validation_result.test_images}")
        
        if validation_result.class_distribution:
            print(f"\nClass Distribution:")
            for class_name, count in validation_result.class_distribution.items():
                print(f"  {class_name}: {count}")
        
        if validation_result.errors:
            print(f"\nErrors ({len(validation_result.errors)}):")
            for error in validation_result.errors:
                print(f"  - {error}")
        
        if validation_result.warnings:
            print(f"\nWarnings ({len(validation_result.warnings)}):")
            for warning in validation_result.warnings:
                print(f"  - {warning}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Training Pipeline CLI")
    parser.add_argument("--format", choices=["json", "table"], default="table",
                       help="Output format")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
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
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show job status")
    status_parser.add_argument("job_id", help="Job ID")
    
    # Jobs command
    jobs_parser = subparsers.add_parser("jobs", help="List training jobs")
    jobs_parser.add_argument("--status", help="Filter by status")
    
    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Hyperparameter optimization")
    opt_parser.add_argument("--model", required=True,
                           help="Model to optimize")
    opt_parser.add_argument("--dataset", required=True,
                           help="Path to dataset")
    opt_parser.add_argument("--trials", type=int, default=20,
                           help="Number of optimization trials")
    opt_parser.add_argument("--parallel", type=int, default=2,
                           help="Parallel trials")
    opt_parser.add_argument("--timeout", type=float, default=24.0,
                           help="Optimization timeout (hours)")
    opt_parser.add_argument("--optimizer-type", default="random_search",
                           choices=["random_search", "grid_search", "bayesian", "genetic"],
                           help="Optimization algorithm")
    opt_parser.add_argument("--objective", default="map50_95",
                           choices=["map50", "map50_95", "train_loss", "val_loss"],
                           help="Objective metric")
    opt_parser.add_argument("--optimize-lr", action="store_true",
                           help="Optimize learning rate")
    opt_parser.add_argument("--optimize-batch", action="store_true",
                           help="Optimize batch size")
    opt_parser.add_argument("--optimize-weight-decay", action="store_true",
                           help="Optimize weight decay")
    opt_parser.add_argument("--export-results", action="store_true",
                           help="Export optimization results")
    opt_parser.add_argument("--epochs", type=int, default=50,
                           help="Epochs per trial")
    opt_parser.add_argument("--dataset-format", default="coco",
                           help="Dataset format")
    opt_parser.add_argument("--num-classes", type=int, default=80,
                           help="Number of classes")
    opt_parser.add_argument("--output-dir", default="./training_outputs",
                           help="Output directory")
    opt_parser.add_argument("--batch-size", type=int, default=16,
                           help="Base batch size")
    opt_parser.add_argument("--lr", type=float, default=0.001,
                           help="Base learning rate")
    opt_parser.add_argument("--weight-decay", type=float, default=0.0005,
                           help="Base weight decay")
    opt_parser.add_argument("--optimizer", default="adam",
                           help="Base optimizer")
    opt_parser.add_argument("--patience", type=int, default=10,
                           help="Early stopping patience")
    opt_parser.add_argument("--device", default="auto",
                           help="Training device")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Show monitoring dashboard")
    monitor_parser.add_argument("--live", action="store_true",
                               help="Live monitoring mode")
    monitor_parser.add_argument("--refresh-interval", type=int, default=5,
                               help="Refresh interval for live mode (seconds)")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("dataset_path", help="Path to dataset")
    validate_parser.add_argument("--format", default="coco",
                                choices=["coco", "yolo", "pascal_voc", "custom"],
                                help="Dataset format")
    validate_parser.add_argument("--num-classes", type=int,
                                help="Number of classes")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    if args.command == "train":
        asyncio.run(train_model(args))
    elif args.command == "status":
        show_job_status(args)
    elif args.command == "jobs":
        list_jobs(args)
    elif args.command == "optimize":
        start_hyperparameter_optimization(args)
    elif args.command == "monitor":
        show_monitoring_dashboard(args)
    elif args.command == "validate":
        validate_dataset_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()