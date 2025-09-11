#!/usr/bin/env python3
"""
Inference Engine CLI - Main entry point for inference operations.

This CLI provides a unified command-line interface for running inference tasks
in the ML Evaluation Platform. It imports and extends the library CLI functionality
with additional features for production inference workflows.

Usage:
    python -m backend.src.cli.inference_cli --help
    python inference_cli.py single yolo11_nano image.jpg
    python inference_cli.py batch yolo11_nano "images/*.jpg" --async
    python inference_cli.py pipeline yolo11_nano "input/" "output/" --watch
    python inference_cli.py serve yolo11_nano --port 8080
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
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Import library CLI functionality
    from backend.src.lib.inference_engine.cli import (
        single_inference as lib_single_inference,
        batch_inference as lib_batch_inference,
        job_status as lib_job_status,
        list_jobs as lib_list_jobs,
        show_monitor_stats as lib_show_monitor_stats,
        job_manager,
        monitor
    )
    
    # Import additional library components
    from backend.src.lib.inference_engine.engine import InferenceEngine, InferenceStatus
    from backend.src.lib.inference_engine.processors import InferenceJobManager, ProcessingOptions
    from backend.src.lib.inference_engine.formatters import get_formatter
    from backend.src.lib.inference_engine.monitoring import PerformanceMonitor
    from backend.src.lib.inference_engine.server import InferenceServer
    
    logger.info("Successfully imported Inference Engine library components")
    
except ImportError as e:
    logger.error(f"Failed to import Inference Engine library: {e}")
    print(f"Error: Failed to import Inference Engine library: {e}")
    print("Please ensure the backend.src.lib.inference_engine package is properly installed.")
    sys.exit(1)


class InferenceCLI:
    """Enhanced CLI for Inference Engine with production features."""
    
    def __init__(self):
        self.verbose = False
        self.config_file = None
        self.shutdown_event = threading.Event()
        self.server_instance = None
        
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
    
    async def run_pipeline(self, args) -> None:
        """Run inference pipeline with directory watching."""
        print(f"Starting inference pipeline: {args.input_dir} -> {args.output_dir}")
        
        try:
            input_dir = Path(args.input_dir)
            output_dir = Path(args.output_dir)
            
            if not input_dir.exists():
                print(f"Error: Input directory not found: {args.input_dir}")
                sys.exit(1)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Processing options
            options = ProcessingOptions(
                max_concurrent=args.max_concurrent,
                timeout_seconds=args.timeout,
                retry_count=args.retry_count,
                validate_images=not args.no_validation
            )
            
            # Parameters
            parameters = {}
            if args.confidence:
                parameters['confidence'] = args.confidence
            if args.iou_threshold:
                parameters['iou_threshold'] = args.iou_threshold
            
            # Process existing images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
            existing_images = []
            
            for ext in image_extensions:
                existing_images.extend(input_dir.glob(f"*{ext}"))
                existing_images.extend(input_dir.glob(f"*{ext.upper()}"))
            
            if existing_images:
                print(f"Processing {len(existing_images)} existing images...")
                await self._process_image_batch(
                    [str(p) for p in existing_images],
                    args.model_id,
                    parameters,
                    options,
                    output_dir,
                    args.format
                )
            
            # Watch for new images if requested
            if args.watch:
                print("Watching for new images... (Press Ctrl+C to stop)")
                await self._watch_directory(
                    input_dir,
                    output_dir,
                    args.model_id,
                    parameters,
                    options,
                    args.format
                )
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Pipeline failed: {str(e)}")
            sys.exit(1)
    
    async def _watch_directory(self, input_dir: Path, output_dir: Path,
                             model_id: str, parameters: Dict[str, Any],
                             options: ProcessingOptions, output_format: str):
        """Watch directory for new images."""
        import watchdog.observers
        import watchdog.events
        
        processed_files = set()
        
        class ImageHandler(watchdog.events.FileSystemEventHandler):
            def __init__(self, cli_instance):
                self.cli = cli_instance
                
            def on_created(self, event):
                if event.is_directory:
                    return
                
                file_path = Path(event.src_path)
                if file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}:
                    if file_path not in processed_files:
                        processed_files.add(file_path)
                        print(f"New image detected: {file_path.name}")
                        
                        # Process the new image
                        asyncio.create_task(
                            self.cli._process_single_image(
                                str(file_path), model_id, parameters, options, output_dir, output_format
                            )
                        )
        
        observer = watchdog.observers.Observer()
        observer.schedule(ImageHandler(self), str(input_dir), recursive=False)
        observer.start()
        
        try:
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)
        finally:
            observer.stop()
            observer.join()
    
    async def _process_single_image(self, image_path: str, model_id: str,
                                   parameters: Dict[str, Any], options: ProcessingOptions,
                                   output_dir: Path, output_format: str):
        """Process a single image."""
        try:
            result = await job_manager.process_single_image(
                model_id, image_path, parameters, options
            )
            
            if result.status == InferenceStatus.COMPLETED:
                # Save results
                image_name = Path(image_path).stem
                output_file = output_dir / f"{image_name}_predictions.json"
                
                formatter = get_formatter(output_format)
                formatted_result = formatter.format_single_result(result)
                
                with open(output_file, 'w') as f:
                    json.dump(formatted_result, f, indent=2, default=str)
                
                print(f"Processed: {Path(image_path).name} -> {output_file.name}")
            else:
                print(f"Failed to process: {Path(image_path).name} - {result.error_message}")
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            print(f"Error processing {Path(image_path).name}: {str(e)}")
    
    async def _process_image_batch(self, image_paths: List[str], model_id: str,
                                  parameters: Dict[str, Any], options: ProcessingOptions,
                                  output_dir: Path, output_format: str):
        """Process a batch of images."""
        job = job_manager.process_batch_images(model_id, image_paths, parameters, options)
        
        print(f"Started batch job: {job.job_id}")
        
        # Wait for completion with progress updates
        while not job.is_complete:
            await asyncio.sleep(2)
            # Refresh job status
            updated_job = job_manager.engines[0].get_batch_job_status(job.job_id)
            if updated_job:
                job = updated_job
                progress = job.progress_percentage
                print(f"\rProgress: {progress:.1f}% ({job.completed_images}/{job.total_images})", end="")
        
        print(f"\nBatch completed: {job.completed_images} successful, {job.failed_images} failed")
        
        # Save results
        formatter = get_formatter(output_format)
        formatted_results = formatter.format_batch_results(job)
        
        batch_output = output_dir / "batch_results.json"
        with open(batch_output, 'w') as f:
            json.dump(formatted_results, f, indent=2, default=str)
        
        print(f"Results saved to: {batch_output}")
    
    def start_server(self, args) -> None:
        """Start inference server."""
        print(f"Starting inference server for model: {args.model_id}")
        
        try:
            # Create server configuration
            server_config = {
                "model_id": args.model_id,
                "host": args.host,
                "port": args.port,
                "max_concurrent_requests": args.max_concurrent,
                "request_timeout": args.timeout,
                "enable_monitoring": True,
                "enable_metrics": True,
                "log_level": "DEBUG" if args.verbose else "INFO"
            }
            
            # Create and configure server
            self.server_instance = InferenceServer(server_config)
            
            # Set up signal handlers
            def signal_handler(signum, frame):
                print("\nShutting down server...")
                self.shutdown_event.set()
                if self.server_instance:
                    self.server_instance.shutdown()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Start server
            print(f"Server starting on http://{args.host}:{args.port}")
            print("Available endpoints:")
            print(f"  POST /predict - Single image inference")
            print(f"  POST /batch - Batch image inference")
            print(f"  GET /health - Health check")
            print(f"  GET /metrics - Performance metrics")
            print("Press Ctrl+C to stop")
            
            self.server_instance.start()
            
            # Keep server running
            while not self.shutdown_event.is_set():
                time.sleep(1)
        
        except Exception as e:
            logger.error(f"Server failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Server failed: {str(e)}")
            sys.exit(1)
    
    def stream_logs(self, args) -> None:
        """Stream inference logs in real-time."""
        print("Streaming inference logs... (Press Ctrl+C to stop)")
        
        try:
            import time
            
            # Set up log streaming
            last_check = time.time()
            
            while True:
                # Get recent log entries
                summary = monitor.get_performance_summary(time_window=1)
                
                if summary.get("recent_requests"):
                    for request in summary["recent_requests"]:
                        timestamp = request.get("timestamp", "")
                        model_id = request.get("model_id", "")
                        status = request.get("status", "")
                        duration = request.get("duration_ms", 0)
                        
                        print(f"[{timestamp}] {model_id} - {status} ({duration:.1f}ms)")
                
                time.sleep(args.interval)
        
        except KeyboardInterrupt:
            print("\nLog streaming stopped")
        except Exception as e:
            logger.error(f"Log streaming failed: {e}")
            print(f"Log streaming failed: {str(e)}")
    
    def export_metrics(self, args) -> None:
        """Export performance metrics to file."""
        print(f"Exporting metrics to: {args.output}")
        
        try:
            # Gather comprehensive metrics
            metrics = {}
            
            # System summary
            metrics["system_summary"] = monitor.get_performance_summary(args.time_window)
            
            # Model-specific stats
            if args.model_id:
                metrics["model_stats"] = monitor.get_model_stats(args.model_id)
            else:
                metrics["all_model_stats"] = {}
                # Get stats for all models (if available)
                for model_id in job_manager.get_active_models():
                    model_stats = monitor.get_model_stats(model_id)
                    if model_stats:
                        metrics["all_model_stats"][model_id] = model_stats
            
            # Job history
            active_jobs = job_manager.get_all_active_jobs()
            metrics["active_jobs"] = [
                {
                    "job_id": job.job_id,
                    "model_id": job.model_id,
                    "status": job.status,
                    "progress": job.progress_percentage,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at
                }
                for job in active_jobs
            ]
            
            # Export to file
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            print(f"Metrics exported to: {args.output}")
            
            # Show summary
            summary = metrics.get("system_summary", {}).get("summary", {})
            print(f"\nSummary (Last {args.time_window} minutes):")
            print(f"  Total Requests: {summary.get('total_inferences', 0)}")
            print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")
            print(f"  Requests/Minute: {summary.get('requests_per_minute', 0):.1f}")
        
        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Metrics export failed: {str(e)}")
            sys.exit(1)


def main():
    """Main CLI entry point with enhanced functionality."""
    # Create CLI instance
    cli = InferenceCLI()
    
    # Main parser
    parser = argparse.ArgumentParser(
        description="Inference Engine CLI - Unified command-line interface for inference operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s single yolo11_nano image.jpg --confidence 0.7
  %(prog)s batch yolo11_nano "images/*.jpg" --async --output results.json
  %(prog)s pipeline yolo11_nano input_dir output_dir --watch
  %(prog)s serve yolo11_nano --port 8080 --max-concurrent 10
  %(prog)s monitor --live --model-id yolo11_nano
  %(prog)s logs --interval 2
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
                             help="Image file paths or patterns")
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
    batch_parser.add_argument("--output", help="Output file path")
    
    # Pipeline command (new)
    pipeline_parser = subparsers.add_parser("pipeline", help="Run inference pipeline")
    pipeline_parser.add_argument("model_id", help="Model ID")
    pipeline_parser.add_argument("input_dir", help="Input directory")
    pipeline_parser.add_argument("output_dir", help="Output directory")
    pipeline_parser.add_argument("--watch", action="store_true",
                                help="Watch for new files")
    pipeline_parser.add_argument("--confidence", type=float, default=0.5,
                                help="Confidence threshold")
    pipeline_parser.add_argument("--iou-threshold", type=float, default=0.5,
                                help="IoU threshold")
    pipeline_parser.add_argument("--max-concurrent", type=int, default=4,
                                help="Maximum concurrent processes")
    pipeline_parser.add_argument("--timeout", type=int, default=30,
                                help="Timeout per image")
    pipeline_parser.add_argument("--retry-count", type=int, default=2,
                                help="Number of retries")
    pipeline_parser.add_argument("--no-validation", action="store_true",
                                help="Skip image validation")
    
    # Server command (new)
    server_parser = subparsers.add_parser("serve", help="Start inference server")
    server_parser.add_argument("model_id", help="Model ID to serve")
    server_parser.add_argument("--host", default="127.0.0.1",
                              help="Server host (default: 127.0.0.1)")
    server_parser.add_argument("--port", type=int, default=8000,
                              help="Server port (default: 8000)")
    server_parser.add_argument("--max-concurrent", type=int, default=10,
                              help="Max concurrent requests (default: 10)")
    server_parser.add_argument("--timeout", type=int, default=60,
                              help="Request timeout (default: 60)")
    
    # Job status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("job_id", help="Batch job ID")
    
    # List jobs command
    jobs_parser = subparsers.add_parser("jobs", help="List active jobs")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Show performance statistics")
    monitor_parser.add_argument("--model-id", help="Show stats for specific model")
    monitor_parser.add_argument("--time-window", type=int, default=60,
                               help="Time window in minutes (default: 60)")
    monitor_parser.add_argument("--live", action="store_true",
                               help="Live monitoring mode")
    monitor_parser.add_argument("--refresh-interval", type=int, default=5,
                               help="Refresh interval for live mode (default: 5)")
    
    # Logs command (new)
    logs_parser = subparsers.add_parser("logs", help="Stream inference logs")
    logs_parser.add_argument("--interval", type=int, default=1,
                            help="Refresh interval in seconds (default: 1)")
    
    # Export metrics command (new)
    export_parser = subparsers.add_parser("export", help="Export performance metrics")
    export_parser.add_argument("--output", required=True,
                              help="Output file path")
    export_parser.add_argument("--model-id", help="Export for specific model")
    export_parser.add_argument("--time-window", type=int, default=60,
                              help="Time window in minutes (default: 60)")
    
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
        if args.command == "single":
            asyncio.run(lib_single_inference(args))
        
        elif args.command == "batch":
            lib_batch_inference(args)
        
        elif args.command == "pipeline":
            asyncio.run(cli.run_pipeline(args))
        
        elif args.command == "serve":
            cli.start_server(args)
        
        elif args.command == "status":
            lib_job_status(args)
        
        elif args.command == "jobs":
            lib_list_jobs(args)
        
        elif args.command == "monitor":
            if hasattr(args, 'live') and args.live:
                # Live monitoring implementation
                try:
                    while True:
                        import os
                        os.system('cls' if os.name == 'nt' else 'clear')
                        lib_show_monitor_stats(args)
                        print(f"\nRefreshing every {args.refresh_interval}s... (Press Ctrl+C to exit)")
                        time.sleep(args.refresh_interval)
                except KeyboardInterrupt:
                    print("\nMonitoring stopped")
            else:
                lib_show_monitor_stats(args)
        
        elif args.command == "logs":
            cli.stream_logs(args)
        
        elif args.command == "export":
            cli.export_metrics(args)
        
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