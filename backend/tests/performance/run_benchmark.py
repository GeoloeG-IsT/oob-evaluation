#!/usr/bin/env python3
"""
Performance benchmark runner for ML Evaluation Platform.

This script runs comprehensive performance benchmarks and generates
detailed reports for inference speed validation.
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_inference_speed import PerformanceTestSuite


def run_quick_benchmark() -> Dict[str, Any]:
    """Run quick performance benchmark with essential tests."""
    
    async def quick_tests():
        suite = PerformanceTestSuite()
        suite._setup_models()
        suite._setup_inference_engine()
        
        print("🚀 Starting Quick Performance Benchmark...")
        
        # Test core models only
        quick_models = ["yolo11_yolo_nano", "yolo11_yolo_small"]
        
        for model_id in quick_models:
            if suite.model_registry.get_model_config(model_id):
                print(f"  ⚡ Testing {model_id}...")
                
                # Model loading test
                result = await suite._run_model_loading_test(model_id)
                suite.results.append(result)
                
                # Single inference test
                result = await suite._run_single_inference_test(
                    model_id, suite.test_images["standard_simple"]
                )
                suite.results.append(result)
        
        return suite._generate_performance_report()
    
    return asyncio.run(quick_tests())


def run_comprehensive_benchmark() -> Dict[str, Any]:
    """Run comprehensive performance benchmark with all tests."""
    
    async def comprehensive_tests():
        suite = PerformanceTestSuite()
        suite._setup_models()
        suite._setup_inference_engine()
        
        print("🔥 Starting Comprehensive Performance Benchmark...")
        
        # Test all available models
        all_models = [
            config.model_id 
            for config in suite.model_registry.list_models()
        ]
        
        for model_id in all_models[:6]:  # Limit to first 6 to avoid excessive runtime
            if suite.model_registry.get_model_config(model_id):
                print(f"  🎯 Testing {model_id}...")
                
                # Model loading test
                print(f"    📥 Model loading...")
                result = await suite._run_model_loading_test(model_id)
                suite.results.append(result)
                
                # Single inference tests with different image types
                image_tests = ["small_simple", "standard_medium", "hd_simple"]
                for image_key in image_tests:
                    if image_key in suite.test_images:
                        print(f"    🖼️  Single inference on {image_key}...")
                        result = await suite._run_single_inference_test(
                            model_id, suite.test_images[image_key]
                        )
                        suite.results.append(result)
                
                # Batch inference test
                print(f"    📦 Batch inference...")
                batch_specs = [
                    suite.test_images["small_simple"],
                    suite.test_images["standard_simple"]
                ]
                result = await suite._run_batch_inference_test(model_id, batch_specs)
                suite.results.append(result)
                
                # Concurrent inference test  
                print(f"    🔀 Concurrent inference...")
                result = await suite._run_concurrent_inference_test(
                    model_id, suite.test_images["standard_simple"], 3
                )
                suite.results.append(result)
        
        return suite._generate_performance_report()
    
    return asyncio.run(comprehensive_tests())


def run_stress_test() -> Dict[str, Any]:
    """Run stress test with high load scenarios."""
    
    async def stress_tests():
        suite = PerformanceTestSuite()
        suite._setup_models()
        suite._setup_inference_engine(max_workers=8)  # More workers for stress test
        
        print("💪 Starting Stress Test Benchmark...")
        
        model_id = "yolo11_yolo_nano"  # Use fastest model for stress testing
        if not suite.model_registry.get_model_config(model_id):
            print("❌ YOLO nano model not available for stress testing")
            return {"error": "Model not available"}
        
        # High concurrent load test
        print("  🚀 High concurrent load (10 requests)...")
        result = await suite._run_concurrent_inference_test(
            model_id, suite.test_images["standard_simple"], 10
        )
        suite.results.append(result)
        
        # Large batch test
        print("  📦 Large batch processing (20 images)...")
        large_batch = [suite.test_images["standard_simple"]] * 20
        result = await suite._run_batch_inference_test(model_id, large_batch)
        suite.results.append(result)
        
        # Memory stress test - repeated inferences
        print("  🧠 Memory stress test (50 inferences)...")
        for i in range(50):
            if i % 10 == 0:
                print(f"    Progress: {i}/50")
            result = await suite._run_single_inference_test(
                model_id, suite.test_images["standard_simple"], monitor_memory=False
            )
            # Only keep every 10th result to avoid memory bloat
            if i % 10 == 0:
                suite.results.append(result)
        
        return suite._generate_performance_report()
    
    return asyncio.run(stress_tests())


def save_report(report: Dict[str, Any], output_file: str) -> None:
    """Save performance report to file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    report["benchmark_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "python_version": sys.version
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Report saved to: {output_path.absolute()}")


def print_summary(report: Dict[str, Any]) -> None:
    """Print benchmark summary to console."""
    if "error" in report:
        print(f"❌ Benchmark failed: {report['error']}")
        return
    
    summary = report.get("summary", {})
    
    print("\n" + "="*60)
    print("📈 PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"Total Tests: {summary.get('total_tests', 0)}")
    print(f"Successful Tests: {summary.get('successful_tests', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
    print(f"Real-time Compliance: {summary.get('real_time_compliance_rate', 0):.1f}%")
    print(f"Memory Compliance: {summary.get('memory_compliance_rate', 0):.1f}%")
    
    # Performance targets
    targets = report.get("performance_targets", {})
    print(f"\nPerformance Targets:")
    print(f"  Real-time Target: {targets.get('real_time_target_ms', 0)}ms")
    print(f"  Max Memory per Inference: {targets.get('max_memory_per_inference_mb', 0)}MB")
    
    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        print(f"\n🔧 Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    # Memory analysis
    memory_analysis = report.get("memory_analysis", {})
    if memory_analysis:
        print(f"\n🧠 Memory Analysis:")
        print(f"  Current Memory: {memory_analysis.get('current_memory_mb', 0):.1f}MB")
        print(f"  Loaded Models: {memory_analysis.get('loaded_models_count', 0)}")
        print(f"  Active Jobs: {memory_analysis.get('active_jobs_count', 0)}")
    
    print("="*60)


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Performance Benchmark Runner for ML Evaluation Platform"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["quick", "comprehensive", "stress"],
        default="quick",
        help="Benchmark mode (default: quick)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="performance_report.json",
        help="Output file for performance report (default: performance_report.json)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    try:
        # Run appropriate benchmark
        if args.mode == "quick":
            report = run_quick_benchmark()
        elif args.mode == "comprehensive":  
            report = run_comprehensive_benchmark()
        elif args.mode == "stress":
            report = run_stress_test()
        else:
            raise ValueError(f"Unknown benchmark mode: {args.mode}")
        
        # Save report
        save_report(report, args.output)
        
        # Print summary unless quiet mode
        if not args.quiet:
            print_summary(report)
        
        # Exit with appropriate code
        summary = report.get("summary", {})
        success_rate = summary.get("success_rate", 0)
        real_time_compliance = summary.get("real_time_compliance_rate", 0)
        
        if success_rate < 80.0:
            print("❌ Benchmark failed: Success rate below 80%")
            sys.exit(1)
        elif real_time_compliance < 90.0:
            print("⚠️  Benchmark warning: Real-time compliance below 90%")
            sys.exit(2)
        else:
            print("✅ Benchmark passed: All targets met")
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Benchmark failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()