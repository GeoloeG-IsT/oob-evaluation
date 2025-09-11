#!/usr/bin/env python3
"""
CLI interface for ML Models Library.

Usage examples:
    python -m backend.src.lib.ml_models.cli list
    python -m backend.src.lib.ml_models.cli info yolo11_nano
    python -m backend.src.lib.ml_models.cli predict yolo11_nano /path/to/image.jpg
    python -m backend.src.lib.ml_models.cli load yolo11_nano
    python -m backend.src.lib.ml_models.cli unload yolo11_nano
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .factory import ModelFactory
from .models import ModelType, ModelVariant
from .registry import get_model_registry


def list_models(args) -> None:
    """List available models."""
    model_type = None
    if args.type:
        try:
            model_type = ModelType(args.type.lower())
        except ValueError:
            print(f"Invalid model type: {args.type}")
            print(f"Available types: {[t.value for t in ModelType]}")
            sys.exit(1)
    
    models = ModelFactory.get_available_models(model_type)
    
    if args.format == "json":
        print(json.dumps(models, indent=2, default=str))
    else:
        print(f"{'Model ID':<20} {'Type':<10} {'Variant':<12} {'Name':<25} {'mAP':<6}")
        print("-" * 80)
        for model_id, info in models.items():
            mAP = info.get("performance_metrics", {}).get("mAP", "N/A")
            print(f"{model_id:<20} {info['model_type']:<10} {info['variant']:<12} {info['name']:<25} {mAP:<6}")


def show_model_info(args) -> None:
    """Show detailed model information."""
    model_info = ModelFactory.get_model_info(args.model_id)
    if not model_info:
        print(f"Model '{args.model_id}' not found")
        sys.exit(1)
    
    if args.format == "json":
        print(json.dumps(model_info, indent=2, default=str))
    else:
        print(f"Model Information: {args.model_id}")
        print("-" * 50)
        print(f"Name: {model_info['name']}")
        print(f"Type: {model_info['model_type']}")
        print(f"Variant: {model_info['variant']}")
        print(f"Description: {model_info['description']}")
        print(f"Input Size: {model_info['input_size']}")
        print(f"Number of Classes: {model_info['num_classes']}")
        print(f"Loaded: {model_info['is_loaded']}")
        
        if model_info['performance_metrics']:
            print("\nPerformance Metrics:")
            for metric, value in model_info['performance_metrics'].items():
                print(f"  {metric}: {value}")


def load_model(args) -> None:
    """Load a model into memory."""
    try:
        wrapper = ModelFactory.load_model(args.model_id)
        print(f"Model '{args.model_id}' loaded successfully")
        
        if args.info:
            model_info = wrapper.get_model_info()
            print(f"Model Info: {json.dumps(model_info, indent=2, default=str)}")
            
    except Exception as e:
        print(f"Failed to load model '{args.model_id}': {str(e)}")
        sys.exit(1)


def unload_model(args) -> None:
    """Unload a model from memory."""
    try:
        ModelFactory.unload_model(args.model_id)
        print(f"Model '{args.model_id}' unloaded successfully")
    except Exception as e:
        print(f"Failed to unload model '{args.model_id}': {str(e)}")
        sys.exit(1)


def predict_image(args) -> None:
    """Run inference on an image."""
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Image file not found: {args.image_path}")
        sys.exit(1)
    
    try:
        # Prepare prediction parameters
        kwargs = {}
        if args.confidence:
            kwargs["confidence"] = args.confidence
        if args.iou_threshold:
            kwargs["iou_threshold"] = args.iou_threshold
        if args.points:
            # Parse points for SAM2: "x1,y1;x2,y2"
            points = []
            for point_str in args.points.split(";"):
                x, y = map(int, point_str.split(","))
                points.append([x, y])
            kwargs["points"] = points
        
        result = ModelFactory.predict(args.model_id, str(image_path), **kwargs)
        
        if args.format == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Prediction Results for {args.image_path}")
            print("-" * 50)
            print(f"Model: {result['model_id']}")
            print(f"Inference Time: {result['inference_time']:.3f}s")
            print(f"Number of Predictions: {len(result['predictions'])}")
            
            for i, pred in enumerate(result['predictions']):
                print(f"\nPrediction {i+1}:")
                if 'class_name' in pred:
                    print(f"  Class: {pred['class_name']} (ID: {pred['class_id']})")
                    print(f"  Confidence: {pred['confidence']:.3f}")
                    if 'bbox' in pred:
                        print(f"  Bounding Box: {pred['bbox']}")
                if 'mask' in pred:
                    print(f"  Mask Area: {pred.get('area', 'N/A')}")
                    print(f"  Confidence: {pred['confidence']:.3f}")
    
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        sys.exit(1)


def list_variants(args) -> None:
    """List variants for a model type."""
    try:
        model_type = ModelType(args.type.lower())
        variants = ModelFactory.get_model_variants(model_type)
        
        print(f"Available variants for {model_type.value.upper()}:")
        for variant in variants:
            print(f"  - {variant.value}")
            
    except ValueError:
        print(f"Invalid model type: {args.type}")
        print(f"Available types: {[t.value for t in ModelType]}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="ML Models Library CLI")
    parser.add_argument("--format", choices=["json", "table"], default="table",
                       help="Output format")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--type", help="Filter by model type")
    
    # Model info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model_id", help="Model ID")
    
    # Load model command
    load_parser = subparsers.add_parser("load", help="Load model into memory")
    load_parser.add_argument("model_id", help="Model ID")
    load_parser.add_argument("--info", action="store_true", help="Show model info after loading")
    
    # Unload model command
    unload_parser = subparsers.add_parser("unload", help="Unload model from memory")
    unload_parser.add_argument("model_id", help="Model ID")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference on image")
    predict_parser.add_argument("model_id", help="Model ID")
    predict_parser.add_argument("image_path", help="Path to image file")
    predict_parser.add_argument("--confidence", type=float, default=0.5,
                               help="Confidence threshold (default: 0.5)")
    predict_parser.add_argument("--iou-threshold", type=float, default=0.5,
                               help="IoU threshold for NMS (default: 0.5)")
    predict_parser.add_argument("--points", help="Points for SAM2 (format: x1,y1;x2,y2)")
    
    # List variants command
    variants_parser = subparsers.add_parser("variants", help="List variants for model type")
    variants_parser.add_argument("type", help="Model type")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    commands = {
        "list": list_models,
        "info": show_model_info,
        "load": load_model,
        "unload": unload_model,
        "predict": predict_image,
        "variants": list_variants
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()