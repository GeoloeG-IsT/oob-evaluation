#!/usr/bin/env python3
"""
Annotation Tools CLI - Main entry point for annotation operations.

This CLI provides a unified command-line interface for managing annotations
in the ML Evaluation Platform. It imports and extends the library CLI functionality
with additional features for production annotation workflows.

Usage:
    python -m backend.src.cli.annotation_cli --help
    python annotation_cli.py assist --model sam2_base --image image.jpg --mode segmentation
    python annotation_cli.py batch "images/*.jpg" --assistant sam2 --output results/
    python annotation_cli.py convert --input data.coco --output data.yolo --from coco --to yolo
    python annotation_cli.py validate annotations.json --format coco --quality-check
    python annotation_cli.py workflow --config workflow.json --input images/ --output annotations/
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
    from backend.src.lib.annotation_tools.cli import (
        assist_annotation as lib_assist_annotation,
        convert_annotations as lib_convert_annotations,
        batch_process as lib_batch_process,
        validate_annotations as lib_validate_annotations,
        show_statistics as lib_show_statistics
    )
    
    # Import additional library components
    from backend.src.lib.annotation_tools.assistants import (
        DetectionAssistant,
        SAM2Assistant,
        HybridAssistant,
        AssistantConfig,
        AssistanceRequest,
        AssistanceMode
    )
    from backend.src.lib.annotation_tools.tools import (
        AnnotationValidator,
        ValidationLevel,
        AnnotationType
    )
    from backend.src.lib.annotation_tools.converters import FormatConverter, ConversionConfig
    from backend.src.lib.annotation_tools.processors import (
        BatchAnnotationProcessor,
        ProcessingConfig,
        QualityChecker,
        AnnotationWorkflow
    )
    from backend.src.lib.annotation_tools.utils import (
        AnnotationUtils,
        StatisticsCalculator,
        VisualizationUtils
    )
    
    logger.info("Successfully imported Annotation Tools library components")
    
except ImportError as e:
    logger.error(f"Failed to import Annotation Tools library: {e}")
    print(f"Error: Failed to import Annotation Tools library: {e}")
    print("Please ensure the backend.src.lib.annotation_tools package is properly installed.")
    sys.exit(1)


class AnnotationCLI:
    """Enhanced CLI for Annotation Tools with production features."""
    
    def __init__(self):
        self.verbose = False
        self.config_file = None
        self.shutdown_event = threading.Event()
        
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
    
    async def run_workflow(self, args) -> None:
        """Run annotation workflow from configuration."""
        print(f"Starting annotation workflow from: {args.config}")
        
        try:
            # Load workflow configuration
            if not Path(args.config).exists():
                print(f"Workflow configuration not found: {args.config}")
                sys.exit(1)
            
            with open(args.config, 'r') as f:
                workflow_config = json.load(f)
            
            # Create workflow
            workflow = AnnotationWorkflow(workflow_config)
            
            # Override input/output if specified
            if args.input:
                workflow_config['input_dir'] = args.input
            if args.output:
                workflow_config['output_dir'] = args.output
            
            # Run workflow
            print("Executing workflow steps...")
            result = await workflow.execute()
            
            # Display results
            if result.success:
                print(f"Workflow completed successfully!")
                print(f"Processed {result.total_images} images")
                print(f"Generated {result.total_annotations} annotations")
                print(f"Success rate: {result.success_rate:.1f}%")
                
                if result.output_files:
                    print("Output files:")
                    for file_path in result.output_files:
                        print(f"  - {file_path}")
            else:
                print(f"Workflow failed: {result.error_message}")
                if result.errors:
                    print("Errors:")
                    for error in result.errors:
                        print(f"  - {error}")
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Workflow failed: {str(e)}")
            sys.exit(1)
    
    def create_workflow_config(self, args) -> None:
        """Create a new workflow configuration file."""
        print(f"Creating workflow configuration: {args.output}")
        
        try:
            # Basic workflow template
            workflow_config = {
                "workflow_name": args.name or "annotation_workflow",
                "description": args.description or f"Workflow created on {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "input_dir": args.input or "./images",
                "output_dir": args.output_dir or "./annotations",
                "steps": []
            }
            
            # Add steps based on arguments
            if args.include_assistance:
                workflow_config["steps"].append({
                    "type": "assistance",
                    "assistant_model": args.assistant_model or "sam2_base",
                    "confidence_threshold": args.confidence or 0.5,
                    "max_detections": args.max_detections or 100,
                    "mode": args.assistance_mode or "hybrid"
                })
            
            if args.include_validation:
                workflow_config["steps"].append({
                    "type": "validation",
                    "validation_level": args.validation_level or "standard",
                    "auto_fix": args.auto_fix or False,
                    "quality_threshold": args.quality_threshold or 0.7
                })
            
            if args.include_conversion:
                workflow_config["steps"].append({
                    "type": "conversion",
                    "target_format": args.target_format or "coco",
                    "include_confidence": args.include_confidence or False
                })
            
            # Processing options
            workflow_config["processing"] = {
                "max_concurrent": args.max_concurrent or 4,
                "batch_size": args.batch_size or 10,
                "timeout_seconds": args.timeout or 300,
                "continue_on_error": args.continue_on_error or True
            }
            
            # Output formats and options
            workflow_config["output"] = {
                "formats": [args.output_format or "coco"],
                "save_intermediate": args.save_intermediate or False,
                "create_visualizations": args.create_visualizations or False,
                "export_statistics": args.export_statistics or True
            }
            
            # Save configuration
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(workflow_config, f, indent=2)
            
            print(f"Workflow configuration saved to: {args.output}")
            print(f"Steps included: {len(workflow_config['steps'])}")
            
            # Show preview
            if not args.quiet:
                print("\nWorkflow Preview:")
                print(json.dumps(workflow_config, indent=2))
        
        except Exception as e:
            logger.error(f"Failed to create workflow config: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Failed to create workflow config: {str(e)}")
            sys.exit(1)
    
    def merge_annotations(self, args) -> None:
        """Merge multiple annotation files."""
        print(f"Merging {len(args.input_files)} annotation files")
        
        try:
            merged_annotations = []
            image_id_offset = 0
            annotation_id_offset = 0
            
            # Process each input file
            for i, input_file in enumerate(args.input_files):
                print(f"Processing file {i+1}/{len(args.input_files)}: {Path(input_file).name}")
                
                if not Path(input_file).exists():
                    print(f"Warning: File not found: {input_file}")
                    continue
                
                # Load annotations
                config = ConversionConfig(
                    source_format=args.input_format,
                    target_format="coco"  # Use COCO as intermediate format
                )
                
                annotations = FormatConverter.parse(input_file, args.input_format, config)
                
                # Adjust IDs to prevent conflicts
                for ann in annotations:
                    if hasattr(ann, 'annotation_id'):
                        ann.annotation_id += annotation_id_offset
                    if hasattr(ann, 'image_id'):
                        ann.image_id += image_id_offset
                
                merged_annotations.extend(annotations)
                
                # Update offsets
                max_image_id = max((ann.image_id for ann in annotations if hasattr(ann, 'image_id')), default=0)
                max_ann_id = max((ann.annotation_id for ann in annotations if hasattr(ann, 'annotation_id')), default=0)
                
                image_id_offset = max_image_id + 1
                annotation_id_offset = max_ann_id + 1
            
            print(f"Merged {len(merged_annotations)} annotations")
            
            # Convert to output format
            output_config = ConversionConfig(
                source_format="coco",
                target_format=args.output_format,
                include_confidence=args.include_confidence
            )
            
            # Save merged annotations
            FormatConverter.export(
                merged_annotations,
                args.output,
                args.output_format,
                output_config
            )
            
            print(f"Merged annotations saved to: {args.output}")
            
            # Show statistics if requested
            if args.show_stats:
                stats = StatisticsCalculator.calculate_annotation_stats(merged_annotations)
                print(f"\nMerged Statistics:")
                print(f"Total annotations: {stats['total_annotations']}")
                print(f"Category distribution:")
                for category, count in stats['category_distribution'].items():
                    print(f"  {category}: {count}")
        
        except Exception as e:
            logger.error(f"Merge annotations failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Merge annotations failed: {str(e)}")
            sys.exit(1)
    
    def split_annotations(self, args) -> None:
        """Split annotation file into train/val/test sets."""
        print(f"Splitting annotations: {args.input}")
        
        try:
            # Load annotations
            config = ConversionConfig(
                source_format=args.input_format,
                target_format="coco"
            )
            
            annotations = FormatConverter.parse(args.input, args.input_format, config)
            
            # Group by image
            image_groups = {}
            for ann in annotations:
                image_id = getattr(ann, 'image_id', 0)
                if image_id not in image_groups:
                    image_groups[image_id] = []
                image_groups[image_id].append(ann)
            
            total_images = len(image_groups)
            print(f"Found {total_images} images with annotations")
            
            # Calculate split sizes
            train_size = int(total_images * args.train_ratio)
            val_size = int(total_images * args.val_ratio)
            test_size = total_images - train_size - val_size
            
            print(f"Split: Train={train_size}, Val={val_size}, Test={test_size}")
            
            # Split images
            import random
            if args.seed:
                random.seed(args.seed)
            
            image_ids = list(image_groups.keys())
            random.shuffle(image_ids)
            
            train_ids = set(image_ids[:train_size])
            val_ids = set(image_ids[train_size:train_size + val_size])
            test_ids = set(image_ids[train_size + val_size:])
            
            # Create splits
            splits = {
                'train': [ann for image_id, anns in image_groups.items() if image_id in train_ids for ann in anns],
                'val': [ann for image_id, anns in image_groups.items() if image_id in val_ids for ann in anns],
                'test': [ann for image_id, anns in image_groups.items() if image_id in test_ids for ann in anns]
            }
            
            # Save splits
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_config = ConversionConfig(
                source_format="coco",
                target_format=args.output_format,
                include_confidence=args.include_confidence
            )
            
            for split_name, split_annotations in splits.items():
                if not split_annotations:
                    continue
                    
                output_file = output_dir / f"{split_name}_annotations.{args.output_format.lower()}"
                
                FormatConverter.export(
                    split_annotations,
                    str(output_file),
                    args.output_format,
                    output_config
                )
                
                print(f"{split_name.capitalize()}: {len(split_annotations)} annotations -> {output_file}")
        
        except Exception as e:
            logger.error(f"Split annotations failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Split annotations failed: {str(e)}")
            sys.exit(1)
    
    def create_visualizations(self, args) -> None:
        """Create annotation visualizations."""
        print(f"Creating visualizations for: {args.annotations}")
        
        try:
            # Load annotations
            config = ConversionConfig(
                source_format=args.input_format,
                target_format="coco"
            )
            
            annotations = FormatConverter.parse(args.annotations, args.input_format, config)
            
            # Create output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create visualizations
            viz_utils = VisualizationUtils()
            
            if args.create_overlay:
                print("Creating overlay visualizations...")
                overlay_dir = output_dir / "overlays"
                overlay_dir.mkdir(exist_ok=True)
                
                viz_utils.create_annotation_overlays(
                    annotations,
                    args.images_dir,
                    str(overlay_dir),
                    show_confidence=args.show_confidence,
                    show_labels=args.show_labels
                )
            
            if args.create_statistics:
                print("Creating statistics visualizations...")
                stats_dir = output_dir / "statistics"
                stats_dir.mkdir(exist_ok=True)
                
                viz_utils.create_statistics_plots(
                    annotations,
                    str(stats_dir)
                )
            
            if args.create_heatmap:
                print("Creating annotation heatmaps...")
                heatmap_dir = output_dir / "heatmaps"
                heatmap_dir.mkdir(exist_ok=True)
                
                viz_utils.create_annotation_heatmaps(
                    annotations,
                    args.images_dir,
                    str(heatmap_dir)
                )
            
            print(f"Visualizations saved to: {output_dir}")
        
        except Exception as e:
            logger.error(f"Create visualizations failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Create visualizations failed: {str(e)}")
            sys.exit(1)
    
    def audit_annotations(self, args) -> None:
        """Perform comprehensive annotation audit."""
        print(f"Auditing annotations: {args.annotations}")
        
        try:
            # Load annotations
            config = ConversionConfig(
                source_format=args.input_format,
                target_format="coco"
            )
            
            annotations = FormatConverter.parse(args.annotations, args.input_format, config)
            
            # Perform comprehensive audit
            audit_results = {
                "file_path": args.annotations,
                "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_annotations": len(annotations)
            }
            
            # Basic statistics
            stats = StatisticsCalculator.calculate_annotation_stats(annotations)
            audit_results["statistics"] = stats
            
            # Validation
            validator = AnnotationValidator(ValidationLevel.STRICT)
            validation_result = validator.validate_annotations(annotations, args.width, args.height)
            audit_results["validation"] = validation_result
            
            # Quality check
            quality_checker = QualityChecker(ValidationLevel.STRICT)
            quality_report = quality_checker.check_annotations(annotations, args.width, args.height)
            audit_results["quality"] = quality_report
            
            # Duplicate detection
            if args.check_duplicates:
                duplicates = StatisticsCalculator.find_duplicate_annotations(
                    annotations, iou_threshold=args.duplicate_threshold
                )
                audit_results["duplicates"] = {
                    "total_groups": len(duplicates),
                    "duplicate_groups": duplicates
                }
            
            # Class balance analysis
            class_balance = StatisticsCalculator.calculate_class_balance(annotations)
            audit_results["class_balance"] = class_balance
            
            # Save audit report
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(audit_results, f, indent=2, default=str)
                
                print(f"Audit report saved to: {args.output}")
            
            # Display summary
            print("\nAudit Summary:")
            print(f"  Total Annotations: {audit_results['total_annotations']}")
            print(f"  Valid Annotations: {validation_result.get('valid_annotations', 0)}")
            print(f"  Quality Score: {quality_report.get('quality_score', 0):.3f}")
            print(f"  Class Balance Score: {class_balance.get('balance_score', 0):.3f}")
            
            if args.check_duplicates and audit_results["duplicates"]["total_groups"] > 0:
                print(f"  Duplicate Groups: {audit_results['duplicates']['total_groups']}")
            
            if validation_result.get('total_errors', 0) > 0:
                print(f"  Validation Errors: {validation_result['total_errors']}")
        
        except Exception as e:
            logger.error(f"Audit annotations failed: {e}")
            if self.verbose:
                traceback.print_exc()
            print(f"Audit annotations failed: {str(e)}")
            sys.exit(1)


def main():
    """Main CLI entry point with enhanced functionality."""
    # Create CLI instance
    cli = AnnotationCLI()
    
    # Main parser
    parser = argparse.ArgumentParser(
        description="Annotation Tools CLI - Unified command-line interface for annotation operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s assist --model sam2_base --image image.jpg --mode segmentation --output result.json
  %(prog)s batch "images/*.jpg" --assistant sam2 --output results/ --max-concurrent 8
  %(prog)s convert --input data.coco --output data.yolo --from coco --to yolo --show-stats
  %(prog)s validate annotations.json --format coco --quality-check --show-details
  %(prog)s workflow --config workflow.json --input images/ --output annotations/
  %(prog)s merge file1.json file2.json file3.json --output merged.json --show-stats
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
    
    # Assist command (uses library function)
    assist_parser = subparsers.add_parser("assist", help="Provide annotation assistance")
    assist_parser.add_argument("--model", required=True,
                              help="Model ID for assistance")
    assist_parser.add_argument("--image", required=True,
                              help="Path to image file")
    assist_parser.add_argument("--mode", default="detection",
                              choices=["detection", "segmentation", "interactive", "hybrid"],
                              help="Assistance mode")
    assist_parser.add_argument("--width", type=int, default=640, help="Image width")
    assist_parser.add_argument("--height", type=int, default=640, help="Image height")
    assist_parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    assist_parser.add_argument("--max-detections", type=int, default=100, help="Maximum detections")
    assist_parser.add_argument("--points", help="Interactive points (x1,y1;x2,y2)")
    assist_parser.add_argument("--boxes", help="Interactive boxes (x1,y1,x2,y2)")
    assist_parser.add_argument("--output", help="Output file path")
    assist_parser.add_argument("--output-format", default="coco",
                              choices=["coco", "yolo", "pascal_voc"],
                              help="Output format")
    
    # Convert command (uses library function)
    convert_parser = subparsers.add_parser("convert", help="Convert annotation formats")
    convert_parser.add_argument("--input", required=True, help="Input annotation file")
    convert_parser.add_argument("--output", required=True, help="Output annotation file")
    convert_parser.add_argument("--from", dest="from_format", required=True,
                               choices=["coco", "yolo", "pascal_voc"],
                               help="Source format")
    convert_parser.add_argument("--to", dest="to_format", required=True,
                               choices=["coco", "yolo", "pascal_voc"],
                               help="Target format")
    convert_parser.add_argument("--width", type=int, default=640, help="Image width")
    convert_parser.add_argument("--height", type=int, default=640, help="Image height")
    convert_parser.add_argument("--class-names", help="Comma-separated class names")
    convert_parser.add_argument("--include-confidence", action="store_true",
                               help="Include confidence scores")
    convert_parser.add_argument("--min-area", type=float, default=0.0, help="Minimum area threshold")
    convert_parser.add_argument("--show-stats", action="store_true", help="Show conversion statistics")
    
    # Batch command (uses library function)
    batch_parser = subparsers.add_parser("batch", help="Batch annotation processing")
    batch_parser.add_argument("--images", nargs="+", required=True, help="Image file paths or patterns")
    batch_parser.add_argument("--assistant", required=True, help="Assistant model ID")
    batch_parser.add_argument("--output", help="Output file path")
    batch_parser.add_argument("--output-format", default="coco",
                             choices=["coco", "yolo", "pascal_voc"],
                             help="Output format")
    batch_parser.add_argument("--max-concurrent", type=int, default=4, help="Maximum concurrent processes")
    batch_parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    batch_parser.add_argument("--timeout", type=int, default=300, help="Timeout per batch (seconds)")
    batch_parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    batch_parser.add_argument("--max-detections", type=int, default=100, help="Maximum detections per image")
    batch_parser.add_argument("--validate", action="store_true", help="Enable validation")
    batch_parser.add_argument("--validation-level", default="standard",
                             choices=["basic", "standard", "strict"],
                             help="Validation level")
    batch_parser.add_argument("--auto-fix", action="store_true", help="Auto-fix annotation issues")
    batch_parser.add_argument("--continue-on-error", action="store_true", help="Continue processing on errors")
    batch_parser.add_argument("--async", dest="async_mode", action="store_true", help="Run in async mode")
    batch_parser.add_argument("--show-errors", action="store_true", help="Show error details")
    batch_parser.add_argument("--show-warnings", action="store_true", help="Show warning details")
    
    # Validate command (uses library function)
    validate_parser = subparsers.add_parser("validate", help="Validate annotations")
    validate_parser.add_argument("annotations", help="Annotation file path")
    validate_parser.add_argument("--format", default="coco",
                                choices=["coco", "yolo", "pascal_voc"],
                                help="Annotation format")
    validate_parser.add_argument("--width", type=int, default=640, help="Image width")
    validate_parser.add_argument("--height", type=int, default=640, help="Image height")
    validate_parser.add_argument("--validation-level", default="standard",
                                choices=["basic", "standard", "strict"],
                                help="Validation level")
    validate_parser.add_argument("--show-details", action="store_true",
                                help="Show detailed validation results")
    validate_parser.add_argument("--quality-check", action="store_true", help="Perform quality analysis")
    validate_parser.add_argument("--show-stats", action="store_true", help="Show annotation statistics")
    
    # Stats command (uses library function)
    stats_parser = subparsers.add_parser("stats", help="Show annotation statistics")
    stats_parser.add_argument("annotations", help="Annotation file path")
    stats_parser.add_argument("--format", default="coco",
                             choices=["coco", "yolo", "pascal_voc"],
                             help="Annotation format")
    stats_parser.add_argument("--width", type=int, default=640, help="Image width")
    stats_parser.add_argument("--height", type=int, default=640, help="Image height")
    stats_parser.add_argument("--find-duplicates", action="store_true", help="Find duplicate annotations")
    stats_parser.add_argument("--duplicate-threshold", type=float, default=0.9,
                             help="IoU threshold for duplicate detection")
    
    # Workflow command (new)
    workflow_parser = subparsers.add_parser("workflow", help="Run annotation workflow")
    workflow_parser.add_argument("--config", required=True, help="Workflow configuration file")
    workflow_parser.add_argument("--input", help="Override input directory")
    workflow_parser.add_argument("--output", help="Override output directory")
    
    # Create workflow command (new)
    create_workflow_parser = subparsers.add_parser("create-workflow", help="Create workflow configuration")
    create_workflow_parser.add_argument("--output", required=True, help="Output configuration file")
    create_workflow_parser.add_argument("--name", help="Workflow name")
    create_workflow_parser.add_argument("--description", help="Workflow description")
    create_workflow_parser.add_argument("--input", help="Input directory")
    create_workflow_parser.add_argument("--output-dir", help="Output directory")
    create_workflow_parser.add_argument("--include-assistance", action="store_true",
                                       help="Include assistance step")
    create_workflow_parser.add_argument("--assistant-model", default="sam2_base",
                                       help="Assistant model ID")
    create_workflow_parser.add_argument("--assistance-mode", default="hybrid",
                                       choices=["detection", "segmentation", "hybrid"],
                                       help="Assistance mode")
    create_workflow_parser.add_argument("--include-validation", action="store_true",
                                       help="Include validation step")
    create_workflow_parser.add_argument("--validation-level", default="standard",
                                       choices=["basic", "standard", "strict"],
                                       help="Validation level")
    create_workflow_parser.add_argument("--include-conversion", action="store_true",
                                       help="Include conversion step")
    create_workflow_parser.add_argument("--target-format", default="coco",
                                       choices=["coco", "yolo", "pascal_voc"],
                                       help="Target format for conversion")
    create_workflow_parser.add_argument("--confidence", type=float, default=0.5,
                                       help="Confidence threshold")
    create_workflow_parser.add_argument("--max-detections", type=int, default=100,
                                       help="Maximum detections")
    create_workflow_parser.add_argument("--auto-fix", action="store_true",
                                       help="Enable auto-fix in validation")
    create_workflow_parser.add_argument("--quality-threshold", type=float, default=0.7,
                                       help="Quality threshold")
    create_workflow_parser.add_argument("--include-confidence", action="store_true",
                                       help="Include confidence in output")
    create_workflow_parser.add_argument("--output-format", default="coco",
                                       help="Output format")
    create_workflow_parser.add_argument("--max-concurrent", type=int, default=4,
                                       help="Max concurrent processes")
    create_workflow_parser.add_argument("--batch-size", type=int, default=10,
                                       help="Batch size")
    create_workflow_parser.add_argument("--timeout", type=int, default=300,
                                       help="Timeout seconds")
    create_workflow_parser.add_argument("--continue-on-error", action="store_true",
                                       help="Continue on error")
    create_workflow_parser.add_argument("--save-intermediate", action="store_true",
                                       help="Save intermediate results")
    create_workflow_parser.add_argument("--create-visualizations", action="store_true",
                                       help="Create visualizations")
    create_workflow_parser.add_argument("--export-statistics", action="store_true",
                                       help="Export statistics")
    create_workflow_parser.add_argument("--quiet", action="store_true",
                                       help="Don't show preview")
    
    # Merge command (new)
    merge_parser = subparsers.add_parser("merge", help="Merge multiple annotation files")
    merge_parser.add_argument("input_files", nargs="+", help="Input annotation files")
    merge_parser.add_argument("--output", required=True, help="Output merged file")
    merge_parser.add_argument("--input-format", default="coco",
                             choices=["coco", "yolo", "pascal_voc"],
                             help="Input format")
    merge_parser.add_argument("--output-format", default="coco",
                             choices=["coco", "yolo", "pascal_voc"],
                             help="Output format")
    merge_parser.add_argument("--include-confidence", action="store_true",
                             help="Include confidence scores")
    merge_parser.add_argument("--show-stats", action="store_true",
                             help="Show merge statistics")
    
    # Split command (new)
    split_parser = subparsers.add_parser("split", help="Split annotations into train/val/test")
    split_parser.add_argument("input", help="Input annotation file")
    split_parser.add_argument("--output-dir", required=True, help="Output directory")
    split_parser.add_argument("--train-ratio", type=float, default=0.7,
                             help="Train split ratio (default: 0.7)")
    split_parser.add_argument("--val-ratio", type=float, default=0.2,
                             help="Validation split ratio (default: 0.2)")
    split_parser.add_argument("--seed", type=int, help="Random seed for reproducible splits")
    split_parser.add_argument("--input-format", default="coco",
                             choices=["coco", "yolo", "pascal_voc"],
                             help="Input format")
    split_parser.add_argument("--output-format", default="coco",
                             choices=["coco", "yolo", "pascal_voc"],
                             help="Output format")
    split_parser.add_argument("--include-confidence", action="store_true",
                             help="Include confidence scores")
    
    # Visualize command (new)
    visualize_parser = subparsers.add_parser("visualize", help="Create annotation visualizations")
    visualize_parser.add_argument("annotations", help="Annotation file path")
    visualize_parser.add_argument("--images-dir", required=True, help="Images directory")
    visualize_parser.add_argument("--output-dir", required=True, help="Output directory")
    visualize_parser.add_argument("--input-format", default="coco",
                                 choices=["coco", "yolo", "pascal_voc"],
                                 help="Input format")
    visualize_parser.add_argument("--create-overlay", action="store_true",
                                 help="Create annotation overlays")
    visualize_parser.add_argument("--create-statistics", action="store_true",
                                 help="Create statistics plots")
    visualize_parser.add_argument("--create-heatmap", action="store_true",
                                 help="Create annotation heatmaps")
    visualize_parser.add_argument("--show-confidence", action="store_true",
                                 help="Show confidence scores in overlays")
    visualize_parser.add_argument("--show-labels", action="store_true",
                                 help="Show class labels in overlays")
    
    # Audit command (new)
    audit_parser = subparsers.add_parser("audit", help="Comprehensive annotation audit")
    audit_parser.add_argument("annotations", help="Annotation file path")
    audit_parser.add_argument("--input-format", default="coco",
                             choices=["coco", "yolo", "pascal_voc"],
                             help="Input format")
    audit_parser.add_argument("--output", help="Audit report output file")
    audit_parser.add_argument("--width", type=int, default=640, help="Image width")
    audit_parser.add_argument("--height", type=int, default=640, help="Image height")
    audit_parser.add_argument("--check-duplicates", action="store_true",
                             help="Check for duplicate annotations")
    audit_parser.add_argument("--duplicate-threshold", type=float, default=0.9,
                             help="IoU threshold for duplicate detection")
    
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
        if args.command == "assist":
            asyncio.run(lib_assist_annotation(args))
        
        elif args.command == "convert":
            lib_convert_annotations(args)
        
        elif args.command == "batch":
            asyncio.run(lib_batch_process(args))
        
        elif args.command == "validate":
            lib_validate_annotations(args)
        
        elif args.command == "stats":
            lib_show_statistics(args)
        
        elif args.command == "workflow":
            asyncio.run(cli.run_workflow(args))
        
        elif args.command == "create-workflow":
            cli.create_workflow_config(args)
        
        elif args.command == "merge":
            cli.merge_annotations(args)
        
        elif args.command == "split":
            cli.split_annotations(args)
        
        elif args.command == "visualize":
            cli.create_visualizations(args)
        
        elif args.command == "audit":
            cli.audit_annotations(args)
        
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