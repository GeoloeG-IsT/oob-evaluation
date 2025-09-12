#!/usr/bin/env python3
"""
CLI interface for Annotation Tools Library.

Usage examples:
    python -m backend.src.lib.annotation_tools.cli assist --model sam2_base --image /path/to/image.jpg --mode segmentation
    python -m backend.src.lib.annotation_tools.cli convert --input annotations.coco --output annotations.yolo --from coco --to yolo
    python -m backend.src.lib.annotation_tools.cli batch --images /path/to/images/*.jpg --assistant sam2 --output results.json
    python -m backend.src.lib.annotation_tools.cli validate --annotations annotations.json --format coco
"""
import argparse
import asyncio
import glob
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from .assistants import (
    DetectionAssistant,
    SAM2Assistant,
    HybridAssistant,
    AssistantConfig,
    AssistanceRequest,
    AssistanceMode,
    InteractivePrompt,
    PromptType
)
from .tools import (
    BoundingBoxTool,
    PolygonTool,
    SegmentationTool,
    PointTool,
    AnnotationValidator,
    ValidationLevel,
    AnnotationShape,
    AnnotationType
)
from .converters import FormatConverter, ConversionConfig
from .processors import BatchAnnotationProcessor, ProcessingConfig, QualityChecker
from .utils import AnnotationUtils, StatisticsCalculator, VisualizationUtils
from ..ml_models import ModelFactory


async def assist_annotation(args):
    """Provide annotation assistance for a single image."""
    print(f"Providing annotation assistance for {args.image}")
    
    try:
        # Create assistant configuration
        config = AssistantConfig(
            model_id=args.model,
            confidence_threshold=args.confidence,
            max_detections=args.max_detections
        )
        
        # Create appropriate assistant
        if args.mode == "detection":
            assistant = DetectionAssistant(config)
        elif args.mode == "segmentation":
            assistant = SAM2Assistant(config)
        elif args.mode == "hybrid":
            assistant = HybridAssistant(config)
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        # Create assistance request
        request = AssistanceRequest(
            image_path=args.image,
            image_width=args.width,
            image_height=args.height,
            mode=AssistanceMode(args.mode)
        )
        
        # Add interactive prompts if provided
        if args.points or args.boxes:
            prompt = InteractivePrompt(prompt_type=PromptType.POINTS)
            
            if args.points:
                # Parse points: "x1,y1;x2,y2"
                points = []
                for point_str in args.points.split(";"):
                    x, y = map(float, point_str.split(","))
                    points.append((x, y))
                prompt.points = points
                prompt.point_labels = [1] * len(points)  # All positive
            
            if args.boxes:
                # Parse boxes: "x1,y1,x2,y2"
                boxes = []
                for box_str in args.boxes.split(";"):
                    coords = list(map(float, box_str.split(",")))
                    if len(coords) == 4:
                        boxes.append(coords)
                prompt.boxes = boxes
            
            request.prompts = prompt
        
        # Load model and perform assistance
        await assistant.load_model()
        result = await assistant.assist(request)
        
        # Display results
        if args.format == "json":
            result_dict = {
                "status": result.status,
                "total_annotations": result.total_annotations,
                "processing_time_ms": result.processing_time_ms,
                "quality_score": result.quality_score,
                "annotations": result.annotations,
                "confidence_stats": result.confidence_stats,
                "warnings": result.warnings,
                "error_message": result.error_message
            }
            print(json.dumps(result_dict, indent=2, default=str))
        else:
            print(f"Status: {result.status}")
            print(f"Total annotations: {result.total_annotations}")
            print(f"Processing time: {result.processing_time_ms:.1f}ms")
            
            if result.quality_score is not None:
                print(f"Quality score: {result.quality_score:.3f}")
            
            if result.confidence_stats:
                stats = result.confidence_stats
                print(f"Confidence: avg={stats.get('mean', 0):.3f}, "
                      f"min={stats.get('min', 0):.3f}, "
                      f"max={stats.get('max', 0):.3f}")
            
            if result.warnings:
                print(f"Warnings: {len(result.warnings)}")
                for warning in result.warnings:
                    print(f"  - {warning}")
            
            if result.error_message:
                print(f"Error: {result.error_message}")
        
        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                if args.output.endswith('.json'):
                    json.dump(result.annotations, f, indent=2)
                else:
                    # Save in specified format
                    config = ConversionConfig(
                        source_format="coco",
                        target_format=args.output_format,
                        image_width=args.width,
                        image_height=args.height
                    )
                    
                    # Convert annotations to AnnotationShape objects
                    annotation_shapes = []
                    for ann_data in result.annotations:
                        ann_shape = _dict_to_annotation_shape(ann_data)
                        annotation_shapes.append(ann_shape)
                    
                    converted = FormatConverter.convert(
                        annotation_shapes,
                        "coco",
                        args.output_format,
                        config
                    )
                    
                    if args.output_format == "coco":
                        json.dump(converted, f, indent=2)
                    elif args.output_format == "yolo":
                        for line in converted["annotations"]:
                            f.write(line + '\n')
                    elif args.output_format in ["pascal_voc", "voc"]:
                        f.write(converted)
            
            print(f"Results saved to: {args.output}")
            
    except Exception as e:
        print(f"Assistance failed: {str(e)}")
        sys.exit(1)


def convert_annotations(args):
    """Convert annotations between different formats."""
    print(f"Converting annotations from {args.from_format} to {args.to_format}")
    
    try:
        # Create conversion configuration
        config = ConversionConfig(
            source_format=args.from_format,
            target_format=args.to_format,
            image_width=args.width,
            image_height=args.height,
            include_confidence=args.include_confidence,
            min_area_threshold=args.min_area
        )
        
        # Set format-specific options
        if args.class_names:
            config.yolo_class_names = args.class_names.split(',')
        
        # Perform conversion
        FormatConverter.convert_file_to_file(
            args.input,
            args.output,
            args.from_format,
            args.to_format,
            config
        )
        
        print(f"Conversion completed: {args.input} -> {args.output}")
        
        # Show statistics if requested
        if args.show_stats:
            # Parse annotations to show statistics
            annotations = FormatConverter.parse(args.input, args.from_format, config)
            stats = StatisticsCalculator.calculate_annotation_stats(annotations)
            
            print(f"\nAnnotation Statistics:")
            print(f"Total annotations: {stats['total_annotations']}")
            
            if stats['category_distribution']:
                print("Category distribution:")
                for category, count in stats['category_distribution'].items():
                    print(f"  {category}: {count}")
            
            if stats['area_stats']:
                area_stats = stats['area_stats']
                print(f"Area stats: min={area_stats['min']:.1f}, "
                      f"max={area_stats['max']:.1f}, "
                      f"mean={area_stats['mean']:.1f}")
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        sys.exit(1)


async def batch_process(args):
    """Process multiple images with batch annotation."""
    print(f"Starting batch annotation processing")
    
    try:
        # Expand image patterns
        image_paths = []
        for pattern in args.images:
            if "*" in pattern or "?" in pattern:
                expanded = glob.glob(pattern, recursive=True)
                image_paths.extend(expanded)
            else:
                image_paths.append(pattern)
        
        # Filter existing files
        existing_paths = [p for p in image_paths if Path(p).exists()]
        
        if not existing_paths:
            print("No valid image files found")
            sys.exit(1)
        
        print(f"Found {len(existing_paths)} images to process")
        
        # Create processing configuration
        processing_config = ProcessingConfig(
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
            timeout_seconds=args.timeout,
            enable_validation=args.validate,
            validation_level=ValidationLevel(args.validation_level),
            output_format=args.output_format,
            continue_on_error=args.continue_on_error,
            auto_fix_issues=args.auto_fix
        )
        
        # Create assistant
        assistant_config = AssistantConfig(
            model_id=args.assistant,
            confidence_threshold=args.confidence,
            max_detections=args.max_detections
        )
        
        if "sam2" in args.assistant.lower():
            assistant = SAM2Assistant(assistant_config)
        elif "yolo" in args.assistant.lower() or "detr" in args.assistant.lower():
            assistant = DetectionAssistant(assistant_config)
        else:
            assistant = HybridAssistant(assistant_config)
        
        # Create batch processor
        processor = BatchAnnotationProcessor(processing_config)
        
        # Submit job
        job_id = processor.submit_job(existing_paths, assistant, args.output)
        print(f"Batch job started: {job_id}")
        
        if not args.async_mode:
            # Wait for completion
            print("Processing images...")
            
            while True:
                job = processor.get_job_status(job_id)
                if not job:
                    print("Job not found")
                    break
                
                print(f"\rProgress: {job.progress_percentage:.1f}% "
                      f"({job.processed_images}/{job.total_images}) "
                      f"Success: {job.successful_images}, "
                      f"Failed: {job.failed_images}", end="")
                
                if job.is_complete:
                    print(f"\nBatch processing {job.status.lower()}")
                    
                    if job.errors:
                        print(f"Errors: {len(job.errors)}")
                        if args.show_errors:
                            for error in job.errors[-5:]:  # Show last 5 errors
                                print(f"  {error}")
                    
                    if job.warnings:
                        print(f"Warnings: {len(job.warnings)}")
                        if args.show_warnings:
                            for warning in job.warnings[-5:]:  # Show last 5 warnings
                                print(f"  {warning}")
                    
                    break
                
                await asyncio.sleep(2)
        else:
            print(f"Job {job_id} started in async mode")
            print(f"Use 'status {job_id}' to check progress")
        
        # Show final statistics
        final_job = processor.get_job_status(job_id)
        if final_job and final_job.results:
            successful_results = [r for r in final_job.results if r["status"] == "completed"]
            total_annotations = sum(r.get("total_annotations", 0) for r in successful_results)
            avg_processing_time = sum(r.get("processing_time_ms", 0) for r in successful_results) / len(successful_results) if successful_results else 0
            
            print(f"\nFinal Statistics:")
            print(f"Total annotations generated: {total_annotations}")
            print(f"Average processing time: {avg_processing_time:.1f}ms per image")
            
            if args.output:
                print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Batch processing failed: {str(e)}")
        sys.exit(1)


def validate_annotations(args):
    """Validate annotation quality and correctness."""
    print(f"Validating annotations: {args.annotations}")
    
    try:
        # Load annotations
        with open(args.annotations, 'r') as f:
            if args.format == "json" or args.annotations.endswith('.json'):
                annotation_data = json.load(f)
            else:
                print(f"Unsupported annotation file format")
                sys.exit(1)
        
        # Parse annotations based on format
        config = ConversionConfig(
            source_format=args.format,
            target_format="coco",
            image_width=args.width,
            image_height=args.height
        )
        
        if args.format == "coco":
            annotations = FormatConverter.parse(annotation_data, "coco", config)
        else:
            print(f"Format {args.format} not yet supported for validation")
            sys.exit(1)
        
        # Perform validation
        validator = AnnotationValidator(ValidationLevel(args.validation_level))
        validation_result = validator.validate_annotations(
            annotations, args.width, args.height
        )
        
        # Display results
        if args.format == "json":
            print(json.dumps(validation_result, indent=2, default=str))
        else:
            print("Validation Results")
            print("=" * 50)
            print(f"Overall valid: {'Yes' if validation_result['overall_valid'] else 'No'}")
            print(f"Total annotations: {validation_result['total_annotations']}")
            print(f"Valid annotations: {validation_result['valid_annotations']}")
            print(f"Invalid annotations: {validation_result['invalid_annotations']}")
            print(f"Total errors: {validation_result['total_errors']}")
            print(f"Total warnings: {validation_result['total_warnings']}")
            
            if args.show_details:
                print("\nDetailed Results:")
                for i, ann_result in enumerate(validation_result['annotation_results']):
                    result = ann_result['validation_result']
                    if not result['is_valid'] or result['warnings']:
                        print(f"\nAnnotation {i+1} ({ann_result['annotation_id']}):")
                        
                        if result['errors']:
                            print("  Errors:")
                            for error in result['errors']:
                                print(f"    - {error}")
                        
                        if result['warnings']:
                            print("  Warnings:")
                            for warning in result['warnings']:
                                print(f"    - {warning}")
        
        # Quality checking if requested
        if args.quality_check:
            print("\nQuality Analysis:")
            quality_checker = QualityChecker(ValidationLevel(args.validation_level))
            quality_report = quality_checker.check_annotations(
                annotations, args.width, args.height
            )
            
            print(f"Quality score: {quality_report['quality_score']:.3f}")
            print(f"Validation rate: {quality_report['validation_rate']:.1f}%")
            
            if quality_report['confidence_stats']:
                stats = quality_report['confidence_stats']
                print(f"Confidence: avg={stats['average']:.3f}, "
                      f"min={stats['minimum']:.3f}, "
                      f"max={stats['maximum']:.3f}")
        
        # Calculate statistics
        if args.show_stats:
            print("\nStatistics:")
            stats = StatisticsCalculator.calculate_annotation_stats(annotations)
            
            print(f"Category distribution:")
            for category, count in stats['category_distribution'].items():
                print(f"  {category}: {count}")
            
            if stats['area_stats']:
                area_stats = stats['area_stats']
                print(f"Area: min={area_stats['min']:.1f}, "
                      f"max={area_stats['max']:.1f}, "
                      f"mean={area_stats['mean']:.1f}")
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        sys.exit(1)


def show_statistics(args):
    """Show detailed statistics for annotations."""
    print(f"Analyzing annotations: {args.annotations}")
    
    try:
        # Load annotations
        config = ConversionConfig(
            source_format=args.format,
            target_format="coco",
            image_width=args.width,
            image_height=args.height
        )
        
        annotations = FormatConverter.parse(args.annotations, args.format, config)
        
        # Calculate comprehensive statistics
        stats = StatisticsCalculator.calculate_annotation_stats(annotations)
        class_balance = StatisticsCalculator.calculate_class_balance(annotations)
        
        if args.format == "json":
            output = {
                "basic_stats": stats,
                "class_balance": class_balance
            }
            
            if args.find_duplicates:
                duplicates = StatisticsCalculator.find_duplicate_annotations(
                    annotations, iou_threshold=args.duplicate_threshold
                )
                output["duplicate_groups"] = duplicates
            
            print(json.dumps(output, indent=2, default=str))
        else:
            print("Annotation Statistics")
            print("=" * 50)
            
            print(f"Total annotations: {stats['total_annotations']}")
            
            if stats['area_stats']:
                area_stats = stats['area_stats']
                print(f"\nArea Statistics:")
                print(f"  Min: {area_stats['min']:.1f}")
                print(f"  Max: {area_stats['max']:.1f}")
                print(f"  Mean: {area_stats['mean']:.1f}")
                print(f"  Median: {area_stats['median']:.1f}")
                print(f"  Total: {area_stats['total']:.1f}")
            
            if stats['confidence_stats']:
                conf_stats = stats['confidence_stats']
                print(f"\nConfidence Statistics:")
                print(f"  Min: {conf_stats['min']:.3f}")
                print(f"  Max: {conf_stats['max']:.3f}")
                print(f"  Mean: {conf_stats['mean']:.3f}")
                print(f"  Median: {conf_stats['median']:.3f}")
            
            print(f"\nCategory Distribution:")
            for category, count in stats['category_distribution'].items():
                percentage = (count / stats['total_annotations']) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
            
            print(f"\nType Distribution:")
            for ann_type, count in stats['type_distribution'].items():
                percentage = (count / stats['total_annotations']) * 100
                print(f"  {ann_type}: {count} ({percentage:.1f}%)")
            
            # Class balance analysis
            print(f"\nClass Balance Analysis:")
            print(f"  Balance score: {class_balance['balance_score']:.3f}")
            print(f"  Imbalance ratio: {class_balance['imbalance_ratio']:.1f}")
            print(f"  Most frequent: {class_balance['most_frequent_class'][0]} ({class_balance['most_frequent_class'][1]})")
            print(f"  Least frequent: {class_balance['least_frequent_class'][0]} ({class_balance['least_frequent_class'][1]})")
            
            # Find duplicates if requested
            if args.find_duplicates:
                duplicates = StatisticsCalculator.find_duplicate_annotations(
                    annotations, iou_threshold=args.duplicate_threshold
                )
                
                if duplicates:
                    print(f"\nDuplicate Groups Found: {len(duplicates)}")
                    for i, group in enumerate(duplicates):
                        print(f"  Group {i+1}: annotations {group}")
                else:
                    print(f"\nNo duplicate annotations found (threshold: {args.duplicate_threshold})")
        
    except Exception as e:
        print(f"Statistics calculation failed: {str(e)}")
        sys.exit(1)


def _dict_to_annotation_shape(ann_data: Dict[str, Any]) -> AnnotationShape:
    """Convert annotation dictionary to AnnotationShape."""
    from .tools import AnnotationShape, AnnotationPoint
    
    annotation = AnnotationShape()
    annotation.category_id = ann_data.get("category_id", 0)
    annotation.category_name = ann_data.get("class_name", "")
    annotation.confidence = ann_data.get("confidence", 1.0)
    annotation.bbox = ann_data.get("bbox", [])
    annotation.area = ann_data.get("area", 0.0)
    annotation.is_crowd = bool(ann_data.get("iscrowd", 0))
    
    # Create points from bbox
    if annotation.bbox and len(annotation.bbox) >= 4:
        x, y, w, h = annotation.bbox
        annotation.points = [
            AnnotationPoint(x=x, y=y),
            AnnotationPoint(x=x+w, y=y+h)
        ]
    
    return annotation


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Annotation Tools CLI")
    parser.add_argument("--format", choices=["json", "table"], default="table",
                       help="Output format")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Assist command
    assist_parser = subparsers.add_parser("assist", help="Provide annotation assistance")
    assist_parser.add_argument("--model", required=True,
                              help="Model ID for assistance (e.g., sam2_base, yolo11_nano)")
    assist_parser.add_argument("--image", required=True,
                              help="Path to image file")
    assist_parser.add_argument("--mode", default="detection",
                              choices=["detection", "segmentation", "interactive", "hybrid"],
                              help="Assistance mode")
    assist_parser.add_argument("--width", type=int, default=640,
                              help="Image width")
    assist_parser.add_argument("--height", type=int, default=640,
                              help="Image height")
    assist_parser.add_argument("--confidence", type=float, default=0.5,
                              help="Confidence threshold")
    assist_parser.add_argument("--max-detections", type=int, default=100,
                              help="Maximum detections")
    assist_parser.add_argument("--points", help="Interactive points (x1,y1;x2,y2)")
    assist_parser.add_argument("--boxes", help="Interactive boxes (x1,y1,x2,y2)")
    assist_parser.add_argument("--output", help="Output file path")
    assist_parser.add_argument("--output-format", default="coco",
                              choices=["coco", "yolo", "pascal_voc"],
                              help="Output format")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert annotation formats")
    convert_parser.add_argument("--input", required=True,
                               help="Input annotation file")
    convert_parser.add_argument("--output", required=True,
                               help="Output annotation file")
    convert_parser.add_argument("--from", dest="from_format", required=True,
                               choices=["coco", "yolo", "pascal_voc"],
                               help="Source format")
    convert_parser.add_argument("--to", dest="to_format", required=True,
                               choices=["coco", "yolo", "pascal_voc"],
                               help="Target format")
    convert_parser.add_argument("--width", type=int, default=640,
                               help="Image width")
    convert_parser.add_argument("--height", type=int, default=640,
                               help="Image height")
    convert_parser.add_argument("--class-names", help="Comma-separated class names")
    convert_parser.add_argument("--include-confidence", action="store_true",
                               help="Include confidence scores")
    convert_parser.add_argument("--min-area", type=float, default=0.0,
                               help="Minimum area threshold")
    convert_parser.add_argument("--show-stats", action="store_true",
                               help="Show conversion statistics")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch annotation processing")
    batch_parser.add_argument("--images", nargs="+", required=True,
                             help="Image file paths or patterns")
    batch_parser.add_argument("--assistant", required=True,
                             help="Assistant model ID")
    batch_parser.add_argument("--output", help="Output file path")
    batch_parser.add_argument("--output-format", default="coco",
                             choices=["coco", "yolo", "pascal_voc"],
                             help="Output format")
    batch_parser.add_argument("--max-concurrent", type=int, default=4,
                             help="Maximum concurrent processes")
    batch_parser.add_argument("--batch-size", type=int, default=10,
                             help="Batch size")
    batch_parser.add_argument("--timeout", type=int, default=300,
                             help="Timeout per batch (seconds)")
    batch_parser.add_argument("--confidence", type=float, default=0.5,
                             help="Confidence threshold")
    batch_parser.add_argument("--max-detections", type=int, default=100,
                             help="Maximum detections per image")
    batch_parser.add_argument("--validate", action="store_true",
                             help="Enable validation")
    batch_parser.add_argument("--validation-level", default="standard",
                             choices=["basic", "standard", "strict"],
                             help="Validation level")
    batch_parser.add_argument("--auto-fix", action="store_true",
                             help="Auto-fix annotation issues")
    batch_parser.add_argument("--continue-on-error", action="store_true",
                             help="Continue processing on errors")
    batch_parser.add_argument("--async", dest="async_mode", action="store_true",
                             help="Run in async mode")
    batch_parser.add_argument("--show-errors", action="store_true",
                             help="Show error details")
    batch_parser.add_argument("--show-warnings", action="store_true",
                             help="Show warning details")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate annotations")
    validate_parser.add_argument("annotations", help="Annotation file path")
    validate_parser.add_argument("--format", default="coco",
                                choices=["coco", "yolo", "pascal_voc"],
                                help="Annotation format")
    validate_parser.add_argument("--width", type=int, default=640,
                                help="Image width")
    validate_parser.add_argument("--height", type=int, default=640,
                                help="Image height")
    validate_parser.add_argument("--validation-level", default="standard",
                                choices=["basic", "standard", "strict"],
                                help="Validation level")
    validate_parser.add_argument("--show-details", action="store_true",
                                help="Show detailed validation results")
    validate_parser.add_argument("--quality-check", action="store_true",
                                help="Perform quality analysis")
    validate_parser.add_argument("--show-stats", action="store_true",
                                help="Show annotation statistics")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show annotation statistics")
    stats_parser.add_argument("annotations", help="Annotation file path")
    stats_parser.add_argument("--format", default="coco",
                             choices=["coco", "yolo", "pascal_voc"],
                             help="Annotation format")
    stats_parser.add_argument("--width", type=int, default=640,
                             help="Image width")
    stats_parser.add_argument("--height", type=int, default=640,
                             help="Image height")
    stats_parser.add_argument("--find-duplicates", action="store_true",
                             help="Find duplicate annotations")
    stats_parser.add_argument("--duplicate-threshold", type=float, default=0.9,
                             help="IoU threshold for duplicate detection")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    if args.command == "assist":
        asyncio.run(assist_annotation(args))
    elif args.command == "convert":
        convert_annotations(args)
    elif args.command == "batch":
        asyncio.run(batch_process(args))
    elif args.command == "validate":
        validate_annotations(args)
    elif args.command == "stats":
        show_statistics(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()