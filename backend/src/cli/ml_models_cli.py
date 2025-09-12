#!/usr/bin/env python3
"""
ML Models CLI - Main entry point for ML model operations.

This CLI provides a unified command-line interface for managing machine learning models
in the ML Evaluation Platform. It imports and extends the library CLI functionality
with additional features for production use.

Usage:
    python -m backend.src.cli.ml_models_cli --help
    python ml_models_cli.py list --type yolo11
    python ml_models_cli.py load yolo11_nano --info
    python ml_models_cli.py predict yolo11_nano /path/to/image.jpg --confidence 0.7
    python ml_models_cli.py benchmark yolo11_nano --images /path/to/test/*.jpg
    python ml_models_cli.py compare yolo11_nano yolo12_small --metric map50
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Import library CLI functionality
    from backend.src.lib.ml_models.cli import (
        list_models as lib_list_models,
        show_model_info as lib_show_model_info,
        load_model as lib_load_model,
        unload_model as lib_unload_model,
        predict_image as lib_predict_image,
        list_variants as lib_list_variants
    )
    
    # Import additional library components for extended functionality
    from backend.src.lib.ml_models.factory import ModelFactory
    from backend.src.lib.ml_models.models import ModelType, ModelVariant
    from backend.src.lib.ml_models.registry import get_model_registry
    from backend.src.lib.ml_models.benchmarking import ModelBenchmarker, BenchmarkConfig
    
    logger.info("Successfully imported ML Models library components")
    
except ImportError as e:
    logger.error(f"Failed to import ML Models library: {e}")
    print(f"Error: Failed to import ML Models library: {e}")
    print("Please ensure the backend.src.lib.ml_models package is properly installed.")
    sys.exit(1)


class MLModelsCLI:
    """Enhanced CLI for ML Models with additional production features."""
    
    def __init__(self):
        self.verbose = False
        self.config_file = None
        
    def set_verbosity(self, verbose: bool):
        """Set verbosity level."""
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled")
    
    def load_config(self, config_file: Optional[str]):
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
    
    def benchmark_model(self, args) -> None:
        """Run comprehensive model benchmarks."""
        print(f"Starting benchmark for model: {args.model_id}")
        
        try:
            # Prepare benchmark configuration
            config = BenchmarkConfig(
                model_id=args.model_id,
                test_images=args.images,
                iterations=args.iterations,
                warmup_iterations=args.warmup,
                batch_sizes=args.batch_sizes,
                confidence_thresholds=args.confidence_thresholds,
                iou_thresholds=args.iou_thresholds,
                enable_profiling=args.profile,
                output_dir=args.output_dir
            )
            
            # Create benchmarker
            benchmarker = ModelBenchmarker()
            
            # Run benchmark
            print("Running benchmark tests...")
            start_time = time.time()
            
            results = benchmarker.run_comprehensive_benchmark(config)
            
            total_time = time.time() - start_time
            print(f"Benchmark completed in {total_time:.2f}s")
            
            # Display results
            if args.format == "json":
                print(json.dumps(results, indent=2, default=str))
            else:
                self._display_benchmark_results(results)
            
            # Save results if output specified
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                print(f"Benchmark results saved to: {args.output}")
        
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Benchmark failed: {str(e)}")
            sys.exit(1)
    
    def compare_models(self, args) -> None:
        """Compare performance between multiple models."""
        print(f"Comparing models: {', '.join(args.model_ids)}")
        
        try:
            comparison_results = {}
            
            for model_id in args.model_ids:
                print(f"Benchmarking {model_id}...")
                
                # Run quick benchmark for each model
                config = BenchmarkConfig(
                    model_id=model_id,
                    test_images=args.images,
                    iterations=args.iterations,
                    warmup_iterations=2,
                    batch_sizes=[1],
                    confidence_thresholds=[0.5],
                    iou_thresholds=[0.5]
                )
                
                benchmarker = ModelBenchmarker()
                results = benchmarker.run_comprehensive_benchmark(config)
                comparison_results[model_id] = results
            
            # Display comparison
            if args.format == "json":
                print(json.dumps(comparison_results, indent=2, default=str))
            else:
                self._display_model_comparison(comparison_results, args.metric)
        
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Model comparison failed: {str(e)}")
            sys.exit(1)
    
    def batch_predict(self, args) -> None:
        """Run batch prediction on multiple images."""
        print(f"Running batch prediction with model: {args.model_id}")
        
        try:
            # Expand image paths
            import glob
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
            
            # Prepare prediction parameters
            kwargs = {}
            if args.confidence:
                kwargs["confidence"] = args.confidence
            if args.iou_threshold:
                kwargs["iou_threshold"] = args.iou_threshold
            
            # Process images
            results = []
            for i, image_path in enumerate(existing_paths):
                print(f"Processing image {i+1}/{len(existing_paths)}: {Path(image_path).name}")
                
                try:
                    result = ModelFactory.predict(args.model_id, image_path, **kwargs)
                    results.append({
                        "image_path": image_path,
                        "result": result,
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "image_path": image_path,
                        "error": str(e),
                        "status": "failed"
                    })
            
            # Display summary
            successful = len([r for r in results if r["status"] == "success"])
            failed = len(results) - successful
            
            print(f"\nBatch prediction completed:")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            
            if args.format == "json":
                print(json.dumps(results, indent=2, default=str))
            
            # Save results if output specified
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                print(f"Results saved to: {args.output}")
        
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Batch prediction failed: {str(e)}")
            sys.exit(1)
    
    def show_system_info(self, args) -> None:
        """Show system information and model registry status."""
        print("ML Models System Information")
        print("=" * 50)
        
        try:
            # Show registry info
            registry = get_model_registry()
            print(f"Model Registry: {len(registry.get_all_models())} models registered")
            
            # Show loaded models
            loaded_models = ModelFactory.get_loaded_models()
            print(f"Loaded Models: {len(loaded_models)}")
            
            for model_id in loaded_models:
                model_info = ModelFactory.get_model_info(model_id)
                if model_info:
                    print(f"  - {model_id}: {model_info.get('name', 'Unknown')}")
            
            # Show system resources
            import psutil
            import torch
            
            print(f"\nSystem Resources:")
            print(f"  CPU Cores: {psutil.cpu_count()}")
            print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
            print(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
            
            if torch.cuda.is_available():
                print(f"  CUDA Available: Yes")
                print(f"  CUDA Devices: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    print(f"    Device {i}: {props.name} ({memory_gb:.1f} GB)")
            else:
                print(f"  CUDA Available: No")
            
            # Show model types and variants available
            print(f"\nSupported Model Types:")
            for model_type in ModelType:
                variants = ModelFactory.get_model_variants(model_type)
                print(f"  {model_type.value.upper()}: {[v.value for v in variants]}")
                
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Error getting system info: {str(e)}")
    
    def _display_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Display benchmark results in human-readable format."""
        print("\nBenchmark Results")
        print("=" * 50)
        
        if "summary" in results:
            summary = results["summary"]
            print(f"Model: {summary.get('model_id', 'Unknown')}")
            print(f"Total Tests: {summary.get('total_tests', 0)}")
            print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
            print(f"Average Inference Time: {summary.get('avg_inference_time_ms', 0):.2f}ms")
            print(f"Average Throughput: {summary.get('avg_throughput_fps', 0):.2f} FPS")
        
        if "performance_metrics" in results:
            metrics = results["performance_metrics"]
            print(f"\nPerformance Metrics:")
            print(f"  Min Time: {metrics.get('min_time_ms', 0):.2f}ms")
            print(f"  Max Time: {metrics.get('max_time_ms', 0):.2f}ms")
            print(f"  P50 Time: {metrics.get('p50_time_ms', 0):.2f}ms")
            print(f"  P95 Time: {metrics.get('p95_time_ms', 0):.2f}ms")
            print(f"  P99 Time: {metrics.get('p99_time_ms', 0):.2f}ms")
    
    def _display_model_comparison(self, results: Dict[str, Any], metric: str) -> None:
        """Display model comparison results."""
        print("\nModel Comparison Results")
        print("=" * 60)
        
        # Extract key metrics for comparison
        comparison_data = []
        for model_id, result in results.items():
            summary = result.get("summary", {})
            comparison_data.append({
                "model": model_id,
                "avg_time": summary.get("avg_inference_time_ms", 0),
                "throughput": summary.get("avg_throughput_fps", 0),
                "success_rate": summary.get("success_rate", 0)
            })
        
        # Sort by selected metric
        if metric == "speed":
            comparison_data.sort(key=lambda x: x["avg_time"])
        elif metric == "throughput":
            comparison_data.sort(key=lambda x: x["throughput"], reverse=True)
        else:
            comparison_data.sort(key=lambda x: x["success_rate"], reverse=True)
        
        # Display table
        print(f"{'Model':<20} {'Avg Time (ms)':<15} {'Throughput (FPS)':<18} {'Success Rate':<12}")
        print("-" * 70)
        
        for data in comparison_data:
            print(f"{data['model']:<20} {data['avg_time']:<15.2f} "
                  f"{data['throughput']:<18.2f} {data['success_rate']:<12.1f}%")


def main():
    """Main CLI entry point with enhanced functionality."""
    # Create CLI instance
    cli = MLModelsCLI()
    
    # Main parser
    parser = argparse.ArgumentParser(
        description="ML Models CLI - Unified command-line interface for ML model operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list --type yolo11 --format json
  %(prog)s info yolo11_nano
  %(prog)s load yolo11_nano --info
  %(prog)s predict yolo11_nano image.jpg --confidence 0.7
  %(prog)s benchmark yolo11_nano --images "test/*.jpg" --iterations 10
  %(prog)s compare yolo11_nano yolo12_small --metric speed
  %(prog)s batch-predict yolo11_nano "images/*.jpg" --output results.json
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
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--type", help="Filter by model type")
    list_parser.add_argument("--loaded-only", action="store_true",
                            help="Show only loaded models")
    
    # Model info command
    info_parser = subparsers.add_parser("info", help="Show detailed model information")
    info_parser.add_argument("model_id", help="Model ID")
    
    # Load model command
    load_parser = subparsers.add_parser("load", help="Load model into memory")
    load_parser.add_argument("model_id", help="Model ID")
    load_parser.add_argument("--info", action="store_true",
                            help="Show model info after loading")
    
    # Unload model command
    unload_parser = subparsers.add_parser("unload", help="Unload model from memory")
    unload_parser.add_argument("model_id", help="Model ID")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference on single image")
    predict_parser.add_argument("model_id", help="Model ID")
    predict_parser.add_argument("image_path", help="Path to image file")
    predict_parser.add_argument("--confidence", type=float, default=0.5,
                               help="Confidence threshold (default: 0.5)")
    predict_parser.add_argument("--iou-threshold", type=float, default=0.5,
                               help="IoU threshold for NMS (default: 0.5)")
    predict_parser.add_argument("--points", help="Points for SAM2 (format: x1,y1;x2,y2)")
    
    # Batch predict command
    batch_predict_parser = subparsers.add_parser("batch-predict", 
                                                help="Run batch prediction on multiple images")
    batch_predict_parser.add_argument("model_id", help="Model ID")
    batch_predict_parser.add_argument("image_patterns", nargs="+",
                                     help="Image file paths or patterns")
    batch_predict_parser.add_argument("--confidence", type=float, default=0.5,
                                     help="Confidence threshold")
    batch_predict_parser.add_argument("--iou-threshold", type=float, default=0.5,
                                     help="IoU threshold for NMS")
    batch_predict_parser.add_argument("--output", help="Output file path")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run model benchmark")
    benchmark_parser.add_argument("model_id", help="Model ID to benchmark")
    benchmark_parser.add_argument("--images", required=True,
                                 help="Test images path or pattern")
    benchmark_parser.add_argument("--iterations", type=int, default=10,
                                 help="Number of iterations (default: 10)")
    benchmark_parser.add_argument("--warmup", type=int, default=3,
                                 help="Warmup iterations (default: 3)")
    benchmark_parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1],
                                 help="Batch sizes to test (default: [1])")
    benchmark_parser.add_argument("--confidence-thresholds", type=float, nargs="+",
                                 default=[0.5], help="Confidence thresholds (default: [0.5])")
    benchmark_parser.add_argument("--iou-thresholds", type=float, nargs="+",
                                 default=[0.5], help="IoU thresholds (default: [0.5])")
    benchmark_parser.add_argument("--profile", action="store_true",
                                 help="Enable profiling")
    benchmark_parser.add_argument("--output-dir", default="./benchmarks",
                                 help="Output directory (default: ./benchmarks)")
    benchmark_parser.add_argument("--output", help="Output file path")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("model_ids", nargs="+", help="Model IDs to compare")
    compare_parser.add_argument("--images", required=True,
                               help="Test images path or pattern")
    compare_parser.add_argument("--iterations", type=int, default=5,
                               help="Iterations per model (default: 5)")
    compare_parser.add_argument("--metric", choices=["speed", "throughput", "accuracy"],
                               default="speed", help="Primary comparison metric")
    
    # List variants command
    variants_parser = subparsers.add_parser("variants", help="List variants for model type")
    variants_parser.add_argument("type", help="Model type")
    
    # System info command
    info_parser = subparsers.add_parser("system", help="Show system information")
    
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
        if args.command == "list":
            if hasattr(args, 'loaded_only') and args.loaded_only:
                # Show only loaded models
                loaded_models = ModelFactory.get_loaded_models()
                if args.format == "json":
                    print(json.dumps(loaded_models, indent=2))
                else:
                    print("Loaded Models:")
                    for model_id in loaded_models:
                        print(f"  - {model_id}")
            else:
                # Use library function
                lib_list_models(args)
        
        elif args.command == "info":
            lib_show_model_info(args)
        
        elif args.command == "load":
            lib_load_model(args)
        
        elif args.command == "unload":
            lib_unload_model(args)
        
        elif args.command == "predict":
            lib_predict_image(args)
        
        elif args.command == "batch-predict":
            cli.batch_predict(args)
        
        elif args.command == "benchmark":
            cli.benchmark_model(args)
        
        elif args.command == "compare":
            cli.compare_models(args)
        
        elif args.command == "variants":
            lib_list_variants(args)
        
        elif args.command == "system":
            cli.show_system_info(args)
        
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            traceback.print_exc()
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()