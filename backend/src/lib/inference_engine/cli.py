#!/usr/bin/env python3
"""
CLI interface for Inference Engine Library.

Usage examples:
    python -m backend.src.lib.inference_engine.cli single yolo11_nano /path/to/image.jpg
    python -m backend.src.lib.inference_engine.cli batch yolo11_nano /path/to/images/*.jpg
    python -m backend.src.lib.inference_engine.cli status job_id_here
    python -m backend.src.lib.inference_engine.cli monitor --live
"""
import argparse
import asyncio
import glob
import json
import sys
import time
from pathlib import Path
from typing import List

from .engine import InferenceEngine, InferenceRequest, BatchInferenceJob, InferenceStatus
from .processors import InferenceJobManager, ProcessingOptions
from .formatters import get_formatter
from .monitoring import PerformanceMonitor


# Global instances for CLI
job_manager = InferenceJobManager(max_engines=1)  # Single engine for CLI
monitor = PerformanceMonitor()

# Register monitoring callback
def monitoring_callback(metrics):
    """Callback to collect metrics during CLI operations."""
    pass  # Metrics are automatically collected

monitor.register_callback(monitoring_callback)


async def single_inference(args):
    """Run single image inference."""
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Prepare parameters
    parameters = {}
    if hasattr(args, 'confidence') and args.confidence:
        parameters['confidence'] = args.confidence
    if hasattr(args, 'iou_threshold') and args.iou_threshold:
        parameters['iou_threshold'] = args.iou_threshold
    if hasattr(args, 'points') and args.points:
        # Parse points for SAM2: "x1,y1;x2,y2"
        points = []
        for point_str in args.points.split(";"):
            x, y = map(int, point_str.split(","))
            points.append([x, y])
        parameters['points'] = points
    
    # Processing options
    options = ProcessingOptions(
        timeout_seconds=args.timeout,
        retry_count=args.retry_count,
        validate_images=not args.no_validation
    )
    
    print(f"Running inference on {args.image_path} with model {args.model_id}...")
    
    try:
        start_time = time.time()
        result = await job_manager.process_single_image(
            args.model_id,
            args.image_path,
            parameters,
            options
        )
        
        # Record metrics
        monitor.record_inference_result(result, args.image_path)
        
        # Format output
        formatter = get_formatter(args.format)
        formatted_result = formatter.format_single_result(result)
        
        if args.format == "json":
            print(json.dumps(formatted_result, indent=2, default=str))
        else:
            print(f"\nInference completed in {time.time() - start_time:.2f}s")
            print(f"Status: {result.status}")
            
            if result.status == InferenceStatus.COMPLETED:
                print(f"Predictions: {len(result.predictions)}")
                print(f"Inference Time: {result.performance_metrics.inference_time_ms:.1f}ms")
                print(f"Total Time: {result.performance_metrics.total_time_ms:.1f}ms")
                
                if args.show_predictions:
                    print("\nPredictions:")
                    for i, pred in enumerate(result.predictions):
                        print(f"  {i+1}. {pred}")
            else:
                print(f"Error: {result.error_message}")
                
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        sys.exit(1)


def batch_inference(args):
    """Run batch inference."""
    # Expand image paths
    image_paths = []
    for pattern in args.image_patterns:
        if "*" in pattern or "?" in pattern:
            expanded = glob.glob(pattern, recursive=True)
            image_paths.extend(expanded)
        else:
            image_paths.append(pattern)
    
    # Filter existing files
    existing_paths = [p for p in image_paths if Path(p).exists()]
    
    if not existing_paths:
        print("Error: No valid image files found")
        sys.exit(1)
    
    print(f"Found {len(existing_paths)} images to process")
    
    # Prepare parameters
    parameters = {}
    if hasattr(args, 'confidence') and args.confidence:
        parameters['confidence'] = args.confidence
    if hasattr(args, 'iou_threshold') and args.iou_threshold:
        parameters['iou_threshold'] = args.iou_threshold
    
    # Processing options
    options = ProcessingOptions(
        max_concurrent=args.max_concurrent,
        timeout_seconds=args.timeout,
        retry_count=args.retry_count,
        validate_images=not args.no_validation
    )
    
    print(f"Starting batch inference with model {args.model_id}...")
    
    try:
        job = job_manager.process_batch_images(
            args.model_id,
            existing_paths,
            parameters,
            options
        )
        
        print(f"Started batch job: {job.job_id}")
        
        if not args.async_mode:
            # Wait for completion and show progress
            while not job.is_complete:
                time.sleep(1)
                # Refresh job status
                updated_job = job_manager.engines[0].get_batch_job_status(job.job_id)
                if updated_job:
                    job = updated_job
                    progress = job.progress_percentage
                    print(f"\rProgress: {progress:.1f}% ({job.completed_images}/{job.total_images})", end="")
            
            print(f"\nBatch job completed!")
            print(f"Total images: {job.total_images}")
            print(f"Successful: {job.completed_images}")
            print(f"Failed: {job.failed_images}")
            
            # Record metrics
            monitor.record_batch_job_metrics(job)
            
            # Format and show results
            if args.show_results:
                formatter = get_formatter(args.format)
                formatted_results = formatter.format_batch_results(job)
                
                if args.format == "json":
                    print(json.dumps(formatted_results, indent=2, default=str))
                else:
                    print("\nSample results:")
                    for i, result in enumerate(job.results[:5]):  # Show first 5
                        if result.status == InferenceStatus.COMPLETED:
                            print(f"  Image {i+1}: {len(result.predictions)} predictions")
        else:
            print(f"Job {job.job_id} started in async mode")
            print(f"Use 'status {job.job_id}' to check progress")
            
    except Exception as e:
        print(f"Error during batch inference: {str(e)}")
        sys.exit(1)


def job_status(args):
    """Check status of a batch job."""
    found_job = None
    
    # Search across all engines
    for engine in job_manager.engines:
        job = engine.get_batch_job_status(args.job_id)
        if job:
            found_job = job
            break
    
    if not found_job:
        print(f"Job {args.job_id} not found")
        sys.exit(1)
    
    if args.format == "json":
        job_data = {
            "job_id": found_job.job_id,
            "model_id": found_job.model_id,
            "status": found_job.status,
            "progress_percentage": found_job.progress_percentage,
            "total_images": found_job.total_images,
            "completed_images": found_job.completed_images,
            "failed_images": found_job.failed_images,
            "created_at": found_job.created_at,
            "started_at": found_job.started_at,
            "completed_at": found_job.completed_at,
            "error_message": found_job.error_message
        }
        print(json.dumps(job_data, indent=2, default=str))
    else:
        print(f"Job Status: {args.job_id}")
        print(f"Model: {found_job.model_id}")
        print(f"Status: {found_job.status}")
        print(f"Progress: {found_job.progress_percentage:.1f}%")
        print(f"Images: {found_job.completed_images + found_job.failed_images}/{found_job.total_images}")
        print(f"Success: {found_job.completed_images}")
        print(f"Failed: {found_job.failed_images}")
        
        if found_job.error_message:
            print(f"Error: {found_job.error_message}")
        
        print(f"Created: {found_job.created_at}")
        if found_job.started_at:
            print(f"Started: {found_job.started_at}")
        if found_job.completed_at:
            print(f"Completed: {found_job.completed_at}")


def list_jobs(args):
    """List active and recent jobs."""
    all_active = job_manager.get_all_active_jobs()
    
    if args.format == "json":
        jobs_data = []
        for job in all_active:
            jobs_data.append({
                "job_id": job.job_id,
                "model_id": job.model_id,
                "status": job.status,
                "progress_percentage": job.progress_percentage,
                "total_images": job.total_images,
                "completed_images": job.completed_images,
                "failed_images": job.failed_images,
                "created_at": job.created_at
            })
        print(json.dumps(jobs_data, indent=2))
    else:
        if not all_active:
            print("No active jobs")
            return
        
        print(f"{'Job ID':<36} {'Model':<15} {'Status':<10} {'Progress':<8} {'Images'}")
        print("-" * 85)
        
        for job in all_active:
            progress = f"{job.progress_percentage:.1f}%"
            images = f"{job.completed_images + job.failed_images}/{job.total_images}"
            print(f"{job.job_id:<36} {job.model_id:<15} {job.status:<10} {progress:<8} {images}")


def show_monitor_stats(args):
    """Show performance monitoring statistics."""
    if args.model_id:
        stats = monitor.get_model_stats(args.model_id)
        if not stats:
            print(f"No statistics found for model: {args.model_id}")
            sys.exit(1)
        
        if args.format == "json":
            stats_dict = {
                "model_id": stats.model_id,
                "total_inferences": stats.total_inferences,
                "successful_inferences": stats.successful_inferences,
                "failed_inferences": stats.failed_inferences,
                "success_rate": stats.success_rate,
                "average_inference_time_ms": stats.average_inference_time_ms,
                "min_inference_time_ms": stats.min_inference_time_ms,
                "max_inference_time_ms": stats.max_inference_time_ms,
                "average_total_time_ms": stats.average_total_time_ms,
                "average_throughput_fps": stats.average_throughput_fps,
                "total_predictions": stats.total_predictions,
                "average_predictions_per_image": stats.average_predictions_per_image,
                "confidence_distribution": stats.confidence_distribution,
                "last_updated": stats.last_updated
            }
            print(json.dumps(stats_dict, indent=2))
        else:
            print(f"Performance Statistics: {stats.model_id}")
            print("-" * 50)
            print(f"Total Inferences: {stats.total_inferences}")
            print(f"Success Rate: {stats.success_rate:.1f}%")
            print(f"Avg Inference Time: {stats.average_inference_time_ms:.1f}ms")
            print(f"Avg Total Time: {stats.average_total_time_ms:.1f}ms")
            print(f"Avg Throughput: {stats.average_throughput_fps:.1f} FPS")
            print(f"Avg Predictions/Image: {stats.average_predictions_per_image:.1f}")
            
            if stats.confidence_distribution:
                print("\nConfidence Distribution:")
                for bucket, count in stats.confidence_distribution.items():
                    print(f"  {bucket}: {count}")
    else:
        # Show system overview
        summary = monitor.get_performance_summary(args.time_window)
        
        if args.format == "json":
            print(json.dumps(summary, indent=2))
        else:
            print(f"Performance Summary (Last {args.time_window} minutes)")
            print("-" * 60)
            
            if summary.get("total_inferences", 0) == 0:
                print("No inference data available")
                return
            
            print(f"Total Inferences: {summary['summary']['total_inferences']}")
            print(f"Success Rate: {summary['summary']['success_rate']:.1f}%")
            print(f"Requests/Minute: {summary['summary']['requests_per_minute']:.1f}")
            
            if "performance" in summary:
                perf = summary["performance"]
                print(f"Avg Inference Time: {perf.get('average_inference_time_ms', 0):.1f}ms")
                print(f"P95 Inference Time: {perf.get('p95_inference_time_ms', 0):.1f}ms")
            
            if summary.get("model_breakdown"):
                print("\nModel Breakdown:")
                for model_id, stats in summary["model_breakdown"].items():
                    success_rate = (stats["success"] / stats["count"] * 100) if stats["count"] > 0 else 0
                    print(f"  {model_id}: {stats['count']} requests ({success_rate:.1f}% success)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Inference Engine CLI")
    parser.add_argument("--format", choices=["json", "table"], default="table",
                       help="Output format")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single inference command
    single_parser = subparsers.add_parser("single", help="Run single image inference")
    single_parser.add_argument("model_id", help="Model ID")
    single_parser.add_argument("image_path", help="Path to image file")
    single_parser.add_argument("--confidence", type=float, default=0.5,
                              help="Confidence threshold (default: 0.5)")
    single_parser.add_argument("--iou-threshold", type=float, default=0.5,
                              help="IoU threshold for NMS (default: 0.5)")
    single_parser.add_argument("--points", help="Points for SAM2 (format: x1,y1;x2,y2)")
    single_parser.add_argument("--timeout", type=int, default=30,
                              help="Timeout in seconds (default: 30)")
    single_parser.add_argument("--retry-count", type=int, default=2,
                              help="Number of retries (default: 2)")
    single_parser.add_argument("--no-validation", action="store_true",
                              help="Skip image validation")
    single_parser.add_argument("--show-predictions", action="store_true",
                              help="Show detailed predictions")
    
    # Batch inference command
    batch_parser = subparsers.add_parser("batch", help="Run batch inference")
    batch_parser.add_argument("model_id", help="Model ID")
    batch_parser.add_argument("image_patterns", nargs="+", 
                             help="Image file paths or patterns (supports wildcards)")
    batch_parser.add_argument("--confidence", type=float, default=0.5,
                             help="Confidence threshold (default: 0.5)")
    batch_parser.add_argument("--iou-threshold", type=float, default=0.5,
                             help="IoU threshold for NMS (default: 0.5)")
    batch_parser.add_argument("--max-concurrent", type=int, default=4,
                             help="Maximum concurrent processes (default: 4)")
    batch_parser.add_argument("--timeout", type=int, default=30,
                             help="Timeout per image in seconds (default: 30)")
    batch_parser.add_argument("--retry-count", type=int, default=2,
                             help="Number of retries per image (default: 2)")
    batch_parser.add_argument("--no-validation", action="store_true",
                             help="Skip image validation")
    batch_parser.add_argument("--async", dest="async_mode", action="store_true",
                             help="Run in async mode (don't wait for completion)")
    batch_parser.add_argument("--show-results", action="store_true",
                             help="Show detailed results after completion")
    
    # Job status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("job_id", help="Batch job ID")
    
    # List jobs command
    list_parser = subparsers.add_parser("jobs", help="List active jobs")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Show performance statistics")
    monitor_parser.add_argument("--model-id", help="Show stats for specific model")
    monitor_parser.add_argument("--time-window", type=int, default=60,
                               help="Time window in minutes for system stats (default: 60)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    if args.command == "single":
        asyncio.run(single_inference(args))
    elif args.command == "batch":
        batch_inference(args)
    elif args.command == "status":
        job_status(args)
    elif args.command == "jobs":
        list_jobs(args)
    elif args.command == "monitor":
        show_monitor_stats(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()