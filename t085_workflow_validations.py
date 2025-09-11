#!/usr/bin/env python3
"""
T085 Workflow Validation Methods

This module contains all the specific workflow validation methods for the 
ML Evaluation Platform quickstart validation system.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import aiohttp
import requests
import numpy as np
from PIL import Image

from t085_comprehensive_validator import ValidationResult

logger = logging.getLogger(__name__)

class WorkflowValidations:
    """Contains all workflow validation methods for T085"""
    
    def __init__(self, session: aiohttp.ClientSession, backend_url: str, test_data: Dict[str, Any]):
        self.session = session
        self.backend_url = backend_url
        self.test_data = test_data
        self.uploaded_images = []
        self.created_annotations = []
        self.available_models = []
        self.training_jobs = []
        self.deployed_models = []
    
    async def validate_step_1_upload_images(self) -> ValidationResult:
        """Validate Step 1: Upload and organize images"""
        logger.info("Validating Step 1: Upload and organize images...")
        
        errors = []
        api_calls = []
        uploaded_images = []
        performance_metrics = {}
        
        try:
            # Test image uploads across all dataset splits
            dataset_splits = ['train', 'validation', 'test']
            images_per_split = 3
            
            total_upload_time = 0
            successful_uploads = 0
            
            for split_idx, dataset_split in enumerate(dataset_splits):
                split_images = [img for img in self.test_data['images'] 
                              if img['config']['split'] == dataset_split][:images_per_split]
                
                for img_data in split_images:
                    image_path = img_data['path']
                    
                    try:
                        start_time = time.time()
                        
                        # Upload via API
                        with open(image_path, 'rb') as f:
                            files = {'files': f}
                            data = {'dataset_split': dataset_split}
                            
                            response = requests.post(
                                f"{self.backend_url}/api/v1/images",
                                files=files,
                                data=data,
                                timeout=30
                            )
                            
                            upload_time = time.time() - start_time
                            total_upload_time += upload_time
                            
                            api_call = {
                                'endpoint': 'POST /api/v1/images',
                                'file': image_path.name,
                                'dataset_split': dataset_split,
                                'status_code': response.status_code,
                                'response_time': upload_time,
                                'success': response.status_code in [200, 201]
                            }
                            api_calls.append(api_call)
                            
                            if response.status_code in [200, 201]:
                                result = response.json()
                                uploaded_image = {
                                    'id': result.get('id'),
                                    'filename': image_path.name,
                                    'dataset_split': dataset_split,
                                    'size': img_data['config']['size'],
                                    'upload_time': upload_time
                                }
                                uploaded_images.append(uploaded_image)
                                self.uploaded_images.append(uploaded_image)
                                successful_uploads += 1
                                
                                logger.info(f"✓ Uploaded {image_path.name} to {dataset_split} ({upload_time:.2f}s)")
                            else:
                                error_msg = f"Upload failed for {image_path.name}: {response.status_code}"
                                errors.append(error_msg)
                                logger.warning(error_msg)
                    
                    except Exception as e:
                        error_msg = f"Upload error for {image_path.name}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
            
            # Verify uploads by listing images
            try:
                start_time = time.time()
                response = requests.get(f"{self.backend_url}/api/v1/images?limit=50", timeout=15)
                list_time = time.time() - start_time
                
                api_calls.append({
                    'endpoint': 'GET /api/v1/images',
                    'status_code': response.status_code,
                    'response_time': list_time,
                    'success': response.status_code == 200
                })
                
                if response.status_code == 200:
                    images_list = response.json()
                    listed_count = len(images_list.get('items', []))
                    
                    # Verify dataset split distribution
                    split_counts = {}
                    for img in images_list.get('items', []):
                        split = img.get('dataset_split', 'unknown')
                        split_counts[split] = split_counts.get(split, 0) + 1
                    
                    logger.info(f"Retrieved {listed_count} images with splits: {split_counts}")
                    
                    # Check if our uploaded images are in the list
                    listed_filenames = {img.get('filename') for img in images_list.get('items', [])}
                    uploaded_filenames = {img['filename'] for img in uploaded_images}
                    missing_files = uploaded_filenames - listed_filenames
                    
                    if missing_files:
                        errors.append(f"Images not found in listing: {missing_files}")
                    
                else:
                    errors.append(f"Failed to list images: {response.status_code}")
            
            except Exception as e:
                errors.append(f"Failed to retrieve images: {str(e)}")
            
            # Calculate performance metrics
            if successful_uploads > 0:
                performance_metrics = {
                    'total_uploads': successful_uploads,
                    'total_upload_time': total_upload_time,
                    'average_upload_time': total_upload_time / successful_uploads,
                    'upload_rate': successful_uploads / total_upload_time if total_upload_time > 0 else 0
                }
        
        except Exception as e:
            errors.append(f"Step 1 validation error: {str(e)}")
        
        success = len(uploaded_images) >= 6 and len(errors) == 0  # Expect at least 6 uploads (2 per split)
        message = f"Uploaded {len(uploaded_images)} images successfully" if success else f"Upload issues: {len(errors)} errors"
        
        return ValidationResult(
            step="Step 1: Upload and Organize Images",
            workflow="Image Management",
            success=success,
            message=message,
            duration=0,  # Set by caller
            data={
                'uploaded_images': uploaded_images,
                'dataset_splits': {split: len([img for img in uploaded_images if img['dataset_split'] == split]) 
                                 for split in dataset_splits}
            },
            errors=errors,
            api_calls=api_calls,
            performance_metrics=performance_metrics
        )
    
    async def validate_step_2_manual_annotation(self) -> ValidationResult:
        """Validate Step 2: Manual annotation"""
        logger.info("Validating Step 2: Manual annotation...")
        
        errors = []
        api_calls = []
        created_annotations = []
        performance_metrics = {}
        
        try:
            if not self.uploaded_images:
                errors.append("No uploaded images available for annotation")
                return ValidationResult(
                    step="Step 2: Manual Annotation",
                    workflow="Annotation",
                    success=False,
                    message="No images available",
                    duration=0,
                    errors=errors
                )
            
            # Create annotations for multiple images with different scenarios
            annotation_scenarios = [
                {
                    'name': 'single_object',
                    'bounding_boxes': [{
                        "x": 100, "y": 100, "width": 200, "height": 150,
                        "class_id": 0, "confidence": 1.0
                    }],
                    'class_labels': ["test_object"]
                },
                {
                    'name': 'multiple_objects',
                    'bounding_boxes': [
                        {"x": 50, "y": 50, "width": 100, "height": 75, "class_id": 0, "confidence": 1.0},
                        {"x": 200, "y": 150, "width": 150, "height": 100, "class_id": 1, "confidence": 1.0}
                    ],
                    'class_labels': ["object_1", "object_2"]
                },
                {
                    'name': 'large_object',
                    'bounding_boxes': [{
                        "x": 20, "y": 20, "width": 400, "height": 300,
                        "class_id": 0, "confidence": 1.0
                    }],
                    'class_labels': ["large_object"]
                }
            ]
            
            total_annotation_time = 0
            
            for i, scenario in enumerate(annotation_scenarios):
                if i >= len(self.uploaded_images):
                    break
                    
                image_id = self.uploaded_images[i]['id']
                
                annotation_data = {
                    "image_id": image_id,
                    "bounding_boxes": scenario['bounding_boxes'],
                    "class_labels": scenario['class_labels'],
                    "user_tag": f"validation_user_{scenario['name']}"
                }
                
                try:
                    start_time = time.time()
                    
                    response = requests.post(
                        f"{self.backend_url}/api/v1/annotations",
                        json=annotation_data,
                        timeout=30
                    )
                    
                    annotation_time = time.time() - start_time
                    total_annotation_time += annotation_time
                    
                    api_call = {
                        'endpoint': 'POST /api/v1/annotations',
                        'scenario': scenario['name'],
                        'image_id': image_id,
                        'objects_count': len(scenario['bounding_boxes']),
                        'status_code': response.status_code,
                        'response_time': annotation_time,
                        'success': response.status_code in [200, 201]
                    }
                    api_calls.append(api_call)
                    
                    if response.status_code in [200, 201]:
                        annotation_result = response.json()
                        created_annotation = {
                            'id': annotation_result.get('id'),
                            'image_id': image_id,
                            'scenario': scenario['name'],
                            'objects_count': len(scenario['bounding_boxes']),
                            'creation_time': annotation_time
                        }
                        created_annotations.append(created_annotation)
                        self.created_annotations.append(created_annotation)
                        
                        logger.info(f"✓ Created annotation for {scenario['name']} scenario ({annotation_time:.2f}s)")
                    else:
                        error_msg = f"Failed to create annotation for {scenario['name']}: {response.status_code}"
                        errors.append(error_msg)
                        
                        # Try to get error details
                        try:
                            error_detail = response.json()
                            logger.warning(f"{error_msg} - {error_detail}")
                        except:
                            logger.warning(error_msg)
                
                except Exception as e:
                    error_msg = f"Annotation error for {scenario['name']}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Verify annotations by retrieving them
            if created_annotations:
                try:
                    # Get annotations for the first annotated image
                    first_image_id = created_annotations[0]['image_id']
                    start_time = time.time()
                    
                    response = requests.get(
                        f"{self.backend_url}/api/v1/annotations?image_id={first_image_id}",
                        timeout=15
                    )
                    retrieve_time = time.time() - start_time
                    
                    api_calls.append({
                        'endpoint': 'GET /api/v1/annotations',
                        'image_id': first_image_id,
                        'status_code': response.status_code,
                        'response_time': retrieve_time,
                        'success': response.status_code == 200
                    })
                    
                    if response.status_code == 200:
                        annotations_list = response.json()
                        retrieved_count = len(annotations_list.get('items', []))
                        logger.info(f"Retrieved {retrieved_count} annotations for verification")
                    else:
                        errors.append(f"Failed to retrieve annotations: {response.status_code}")
                
                except Exception as e:
                    errors.append(f"Failed to verify annotations: {str(e)}")
            
            # Calculate performance metrics
            if created_annotations:
                performance_metrics = {
                    'total_annotations': len(created_annotations),
                    'total_annotation_time': total_annotation_time,
                    'average_annotation_time': total_annotation_time / len(created_annotations),
                    'total_objects_annotated': sum(ann['objects_count'] for ann in created_annotations)
                }
        
        except Exception as e:
            errors.append(f"Step 2 validation error: {str(e)}")
        
        success = len(created_annotations) >= 2 and len(errors) == 0
        message = f"Created {len(created_annotations)} manual annotations" if success else f"Annotation issues: {len(errors)} errors"
        
        return ValidationResult(
            step="Step 2: Manual Annotation",
            workflow="Annotation",
            success=success,
            message=message,
            duration=0,
            data={'annotations': created_annotations},
            errors=errors,
            api_calls=api_calls,
            performance_metrics=performance_metrics
        )
    
    async def validate_step_3_assisted_annotation(self) -> ValidationResult:
        """Validate Step 3: Model selection and assisted annotation"""
        logger.info("Validating Step 3: Model selection and assisted annotation...")
        
        errors = []
        api_calls = []
        assisted_annotations = []
        performance_metrics = {}
        
        try:
            # First, get available models
            try:
                start_time = time.time()
                response = requests.get(f"{self.backend_url}/api/v1/models", timeout=15)
                models_time = time.time() - start_time
                
                api_calls.append({
                    'endpoint': 'GET /api/v1/models',
                    'status_code': response.status_code,
                    'response_time': models_time,
                    'success': response.status_code == 200
                })
                
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = models_data.get('items', [])\n                    self.available_models = available_models
                    
                    model_types = {}\n                    for model in available_models:\n                        model_type = model.get('type', 'unknown')\n                        model_types[model_type] = model_types.get(model_type, 0) + 1
                    \n                    logger.info(f\"Found {len(available_models)} models: {model_types}\")
                    \n                    if not available_models:\n                        errors.append(\"No models available for assisted annotation\")\n                        return ValidationResult(\n                            step=\"Step 3: Model Selection and Assisted Annotation\",\n                            workflow=\"Assisted Annotation\",\n                            success=False,\n                            message=\"No models available\",\n                            duration=0,\n                            errors=errors,\n                            api_calls=api_calls\n                        )
                else:
                    errors.append(f\"Failed to retrieve models: {response.status_code}\")
                    return ValidationResult(
                        step=\"Step 3: Model Selection and Assisted Annotation\",
                        workflow=\"Assisted Annotation\",
                        success=False,
                        message=\"Cannot retrieve models\",
                        duration=0,
                        errors=errors,
                        api_calls=api_calls
                    )
                    
            except Exception as e:
                errors.append(f\"Failed to get models: {str(e)}\")
                return ValidationResult(
                    step=\"Step 3: Model Selection and Assisted Annotation\",
                    workflow=\"Assisted Annotation\",
                    success=False,
                    message=\"Model retrieval failed\",
                    duration=0,
                    errors=errors
                )
            
            # Find suitable models for assisted annotation (prefer segmentation models like SAM2)
            segmentation_models = [m for m in available_models if m.get('type') == 'segmentation']
            detection_models = [m for m in available_models if m.get('type') == 'detection']
            
            # Try assisted annotation with different model types
            test_models = []
            if segmentation_models:
                test_models.append(('segmentation', segmentation_models[0]))
            if detection_models:
                test_models.append(('detection', detection_models[0]))
            
            if not test_models:
                # Use first available model regardless of type
                if available_models:
                    test_models.append(('unknown', available_models[0]))
            
            if not self.uploaded_images:
                errors.append(\"No uploaded images available for assisted annotation\")
                return ValidationResult(
                    step=\"Step 3: Model Selection and Assisted Annotation\",
                    workflow=\"Assisted Annotation\",
                    success=False,
                    message=\"No images available\",
                    duration=0,
                    errors=errors,
                    api_calls=api_calls
                )
            
            total_assisted_time = 0
            
            for model_type, model in test_models:
                # Test on first few images
                for i, uploaded_image in enumerate(self.uploaded_images[:2]):
                    image_id = uploaded_image['id']
                    model_id = model.get('id')
                    
                    annotation_request = {
                        \"image_id\": image_id,
                        \"model_id\": model_id,
                        \"confidence_threshold\": 0.5
                    }
                    
                    try:
                        start_time = time.time()
                        
                        response = requests.post(
                            f\"{self.backend_url}/api/v1/annotations/assisted\",
                            json=annotation_request,
                            timeout=60  # Longer timeout for model inference
                        )
                        
                        assisted_time = time.time() - start_time
                        total_assisted_time += assisted_time
                        
                        api_call = {
                            'endpoint': 'POST /api/v1/annotations/assisted',
                            'model_type': model_type,
                            'model_id': model_id,
                            'model_name': model.get('name', 'unknown'),
                            'image_id': image_id,
                            'status_code': response.status_code,
                            'response_time': assisted_time,
                            'success': response.status_code in [200, 201]
                        }
                        api_calls.append(api_call)
                        
                        if response.status_code in [200, 201]:
                            result = response.json()
                            assisted_annotation = {
                                'id': result.get('id'),
                                'image_id': image_id,
                                'model_id': model_id,
                                'model_name': model.get('name'),
                                'model_type': model_type,
                                'predictions_count': len(result.get('predictions', [])),
                                'confidence_threshold': 0.5,
                                'processing_time': assisted_time
                            }
                            assisted_annotations.append(assisted_annotation)
                            
                            logger.info(f\"✓ Generated assisted annotation with {model.get('name')} ({assisted_time:.2f}s, {assisted_annotation['predictions_count']} predictions)\")
                        else:
                            error_msg = f\"Assisted annotation failed for {model.get('name')}: {response.status_code}\"
                            errors.append(error_msg)
                            
                            # Log response details for debugging
                            try:
                                error_detail = response.json()
                                logger.warning(f\"{error_msg} - {error_detail}\")
                            except:
                                logger.warning(error_msg)
                    
                    except Exception as e:
                        error_msg = f\"Assisted annotation error with {model.get('name')}: {str(e)}\"
                        errors.append(error_msg)
                        logger.error(error_msg)
                    
                    # Don't test all images with all models to save time
                    break
            
            # Calculate performance metrics
            if assisted_annotations:
                performance_metrics = {
                    'total_assisted_annotations': len(assisted_annotations),
                    'total_processing_time': total_assisted_time,
                    'average_processing_time': total_assisted_time / len(assisted_annotations),
                    'total_predictions': sum(ann['predictions_count'] for ann in assisted_annotations),
                    'models_tested': len(set(ann['model_id'] for ann in assisted_annotations))
                }
        
        except Exception as e:
            errors.append(f\"Step 3 validation error: {str(e)}\")
        
        success = len(assisted_annotations) > 0 and len(errors) == 0
        message = f\"Generated {len(assisted_annotations)} assisted annotations\" if success else f\"Assisted annotation issues: {len(errors)} errors\"
        
        return ValidationResult(
            step=\"Step 3: Model Selection and Assisted Annotation\",
            workflow=\"Assisted Annotation\",
            success=success,
            message=message,
            duration=0,
            data={
                'assisted_annotations': assisted_annotations,
                'available_models': len(available_models) if 'available_models' in locals() else 0
            },
            errors=errors,
            api_calls=api_calls,
            performance_metrics=performance_metrics
        )
    
    async def validate_step_4_model_inference(self) -> ValidationResult:
        \"\"\"Validate Step 4: Model inference (single and batch)\"\"\"
        logger.info(\"Validating Step 4: Model inference (single and batch)...\")
        
        errors = []
        api_calls = []
        inference_results = []
        performance_metrics = {}
        
        try:
            if not self.available_models:
                errors.append(\"No models available for inference\")
                return ValidationResult(
                    step=\"Step 4: Model Inference\",
                    workflow=\"Inference\",
                    success=False,
                    message=\"No models available\",
                    duration=0,
                    errors=errors
                )
            
            if not self.uploaded_images:
                errors.append(\"No uploaded images available for inference\")
                return ValidationResult(
                    step=\"Step 4: Model Inference\",
                    workflow=\"Inference\",
                    success=False,
                    message=\"No images available\",
                    duration=0,
                    errors=errors
                )
            
            # Get test images (prefer test split)
            test_images = [img for img in self.uploaded_images if img['dataset_split'] == 'test']
            if not test_images:
                test_images = self.uploaded_images[:3]  # Use first 3 if no test images
            
            # Test single inference
            single_inference_time = 0
            single_inference_count = 0
            
            for i, model in enumerate(self.available_models[:2]):  # Test with first 2 models
                for j, image in enumerate(test_images[:2]):  # Test with first 2 images
                    
                    inference_request = {
                        \"image_id\": image['id'],
                        \"model_id\": model.get('id'),
                        \"confidence_threshold\": 0.5
                    }
                    
                    try:
                        start_time = time.time()
                        
                        response = requests.post(
                            f\"{self.backend_url}/api/v1/inference/single\",
                            json=inference_request,
                            timeout=60
                        )
                        
                        inference_time = time.time() - start_time
                        single_inference_time += inference_time
                        single_inference_count += 1
                        
                        api_call = {
                            'endpoint': 'POST /api/v1/inference/single',
                            'model_name': model.get('name', 'unknown'),
                            'model_id': model.get('id'),
                            'image_id': image['id'],
                            'status_code': response.status_code,
                            'response_time': inference_time,
                            'success': response.status_code in [200, 201]
                        }
                        api_calls.append(api_call)
                        
                        if response.status_code in [200, 201]:
                            result = response.json()
                            inference_result = {
                                'type': 'single',
                                'model_name': model.get('name'),
                                'model_id': model.get('id'),
                                'image_id': image['id'],
                                'predictions_count': len(result.get('predictions', [])),
                                'inference_time': inference_time,
                                'confidence_threshold': 0.5
                            }
                            inference_results.append(inference_result)
                            
                            # Check performance requirement (< 2 seconds)
                            performance_status = \"✓\" if inference_time < 2.0 else \"⚠\"
                            logger.info(f\"{performance_status} Single inference with {model.get('name')}: {inference_time:.2f}s, {inference_result['predictions_count']} predictions\")
                            
                            if inference_time >= 2.0:
                                errors.append(f\"Single inference too slow: {inference_time:.2f}s (requirement: <2s)\")\n                        
                        else:
                            error_msg = f\"Single inference failed for {model.get('name')}: {response.status_code}\"
                            errors.append(error_msg)
                    
                    except Exception as e:
                        error_msg = f\"Single inference error with {model.get('name')}: {str(e)}\"
                        errors.append(error_msg)
                        logger.error(error_msg)
            
            # Test batch inference
            if len(test_images) >= 2:
                batch_image_ids = [img['id'] for img in test_images[:3]]  # Use up to 3 images
                model_for_batch = self.available_models[0]  # Use first model
                
                batch_request = {
                    \"image_ids\": batch_image_ids,
                    \"model_id\": model_for_batch.get('id'),
                    \"confidence_threshold\": 0.5
                }
                
                try:
                    start_time = time.time()
                    
                    response = requests.post(
                        f\"{self.backend_url}/api/v1/inference/batch\",
                        json=batch_request,
                        timeout=120  # Longer timeout for batch processing
                    )
                    
                    batch_time = time.time() - start_time
                    
                    api_call = {
                        'endpoint': 'POST /api/v1/inference/batch',
                        'model_name': model_for_batch.get('name', 'unknown'),
                        'model_id': model_for_batch.get('id'),
                        'image_count': len(batch_image_ids),
                        'status_code': response.status_code,
                        'response_time': batch_time,
                        'success': response.status_code in [200, 201, 202]  # 202 for async processing
                    }
                    api_calls.append(api_call)
                    
                    if response.status_code in [200, 201, 202]:
                        result = response.json()
                        
                        if response.status_code == 202:
                            # Async batch processing
                            job_id = result.get('job_id')
                            logger.info(f\"Batch inference job started: {job_id}\")
                            
                            # Poll for completion (simplified for validation)
                            max_polls = 10
                            poll_interval = 5
                            job_completed = False
                            
                            for poll in range(max_polls):
                                await asyncio.sleep(poll_interval)
                                
                                try:
                                    status_response = requests.get(
                                        f\"{self.backend_url}/api/v1/inference/batch/{job_id}\",
                                        timeout=15
                                    )
                                    
                                    if status_response.status_code == 200:
                                        job_status = status_response.json()
                                        status = job_status.get('status')
                                        
                                        api_calls.append({
                                            'endpoint': f'GET /api/v1/inference/batch/{job_id}',
                                            'poll_attempt': poll + 1,
                                            'job_status': status,
                                            'status_code': status_response.status_code,
                                            'success': True
                                        })
                                        
                                        if status in ['completed', 'failed']:
                                            job_completed = True
                                            break
                                        
                                        logger.info(f\"Batch job {job_id} status: {status} (poll {poll + 1}/{max_polls})\")\n                                    
                                except Exception as e:
                                    logger.warning(f\"Error polling batch job status: {e}\")\n                            
                            batch_result = {
                                'type': 'batch',
                                'model_name': model_for_batch.get('name'),
                                'model_id': model_for_batch.get('id'),
                                'image_count': len(batch_image_ids),
                                'job_id': job_id,
                                'processing_time': batch_time,
                                'completed': job_completed,
                                'async': True
                            }
                        else:
                            # Synchronous batch processing
                            batch_result = {
                                'type': 'batch',
                                'model_name': model_for_batch.get('name'),
                                'model_id': model_for_batch.get('id'),
                                'image_count': len(batch_image_ids),
                                'results_count': len(result.get('results', [])),
                                'processing_time': batch_time,
                                'completed': True,
                                'async': False
                            }
                        
                        inference_results.append(batch_result)
                        logger.info(f\"✓ Batch inference completed: {batch_time:.2f}s for {len(batch_image_ids)} images\")
                        
                    else:
                        error_msg = f\"Batch inference failed: {response.status_code}\"
                        errors.append(error_msg)
                
                except Exception as e:
                    error_msg = f\"Batch inference error: {str(e)}\"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Calculate performance metrics
            performance_metrics = {
                'single_inference_count': single_inference_count,
                'single_inference_total_time': single_inference_time,
                'single_inference_avg_time': single_inference_time / single_inference_count if single_inference_count > 0 else 0,
                'batch_inference_count': len([r for r in inference_results if r['type'] == 'batch']),
                'total_images_processed': sum(1 if r['type'] == 'single' else r.get('image_count', 0) for r in inference_results),
                'performance_requirement_met': all(r.get('inference_time', 0) < 2.0 for r in inference_results if r['type'] == 'single')
            }
        
        except Exception as e:
            errors.append(f\"Step 4 validation error: {str(e)}\")
        
        single_results = [r for r in inference_results if r['type'] == 'single']
        batch_results = [r for r in inference_results if r['type'] == 'batch']
        
        success = len(single_results) > 0 and len(errors) == 0
        message = f\"Completed {len(single_results)} single + {len(batch_results)} batch inferences\" if success else f\"Inference issues: {len(errors)} errors\"
        
        return ValidationResult(
            step=\"Step 4: Model Inference (Single and Batch)\",
            workflow=\"Inference\",
            success=success,
            message=message,
            duration=0,
            data={
                'inference_results': inference_results,
                'single_inferences': len(single_results),
                'batch_inferences': len(batch_results)
            },
            errors=errors,
            api_calls=api_calls,
            performance_metrics=performance_metrics
        )

    # Continue with remaining validation methods...
    async def validate_step_5_performance_evaluation(self) -> ValidationResult:
        \"\"\"Validate Step 5: Performance evaluation\"\"\"
        logger.info(\"Validating Step 5: Performance evaluation...\")
        
        # Implementation placeholder - requires model predictions and ground truth
        # This would test the evaluation metrics calculation endpoints
        
        return ValidationResult(
            step=\"Step 5: Performance Evaluation\",
            workflow=\"Evaluation\",
            success=False,
            message=\"Performance evaluation validation pending - requires ML implementation\",
            duration=0,
            errors=[\"ML evaluation metrics not yet implemented\"]
        )
    
    async def validate_step_6_model_training(self) -> ValidationResult:
        \"\"\"Validate Step 6: Model training/fine-tuning\"\"\"
        logger.info(\"Validating Step 6: Model training/fine-tuning...\")
        
        # Implementation placeholder - requires training infrastructure
        # This would test the training job creation and monitoring endpoints
        
        return ValidationResult(
            step=\"Step 6: Model Training/Fine-tuning\",
            workflow=\"Training\",
            success=False,
            message=\"Model training validation pending - requires ML implementation\",
            duration=0,
            errors=[\"ML training infrastructure not yet implemented\"]
        )
    
    async def validate_step_7_model_deployment(self) -> ValidationResult:
        \"\"\"Validate Step 7: Model deployment\"\"\"
        logger.info(\"Validating Step 7: Model deployment...\")
        
        # Implementation placeholder - requires deployment infrastructure
        # This would test the model deployment and serving endpoints
        
        return ValidationResult(
            step=\"Step 7: Model Deployment\",
            workflow=\"Deployment\",
            success=False,
            message=\"Model deployment validation pending - requires deployment infrastructure\",
            duration=0,
            errors=[\"Deployment infrastructure not yet implemented\"]
        )
    
    async def validate_step_8_data_export(self) -> ValidationResult:
        \"\"\"Validate Step 8: Data export\"\"\"
        logger.info(\"Validating Step 8: Data export...\")
        
        errors = []
        api_calls = []
        export_results = []
        
        try:
            if not self.uploaded_images or not self.created_annotations:
                errors.append(\"No annotated data available for export\")
                return ValidationResult(
                    step=\"Step 8: Data Export\",
                    workflow=\"Export\",
                    success=False,
                    message=\"No data available for export\",
                    duration=0,
                    errors=errors
                )
            
            # Test different export formats
            export_formats = ['COCO', 'YOLO', 'Pascal_VOC']
            annotated_image_ids = [ann['image_id'] for ann in self.created_annotations]
            
            for export_format in export_formats:
                export_request = {
                    \"image_ids\": annotated_image_ids[:3],  # Limit to first 3 for testing
                    \"format\": export_format,
                    \"include_predictions\": False  # Only manual annotations for now
                }
                
                try:
                    start_time = time.time()
                    
                    response = requests.post(
                        f\"{self.backend_url}/api/v1/export/annotations\",
                        json=export_request,
                        timeout=60
                    )
                    
                    export_time = time.time() - start_time
                    
                    api_call = {
                        'endpoint': 'POST /api/v1/export/annotations',
                        'format': export_format,
                        'image_count': len(export_request['image_ids']),
                        'status_code': response.status_code,
                        'response_time': export_time,
                        'success': response.status_code == 200
                    }
                    api_calls.append(api_call)
                    
                    if response.status_code == 200:
                        # Check if response contains file data
                        content_type = response.headers.get('content-type', '')
                        content_length = len(response.content)
                        
                        export_result = {
                            'format': export_format,
                            'image_count': len(export_request['image_ids']),
                            'export_time': export_time,
                            'content_type': content_type,
                            'content_size': content_length
                        }
                        export_results.append(export_result)
                        
                        logger.info(f\"✓ Exported {export_format} format: {content_length} bytes ({export_time:.2f}s)\")
                    else:
                        error_msg = f\"Export failed for {export_format}: {response.status_code}\"
                        errors.append(error_msg)
                
                except Exception as e:
                    error_msg = f\"Export error for {export_format}: {str(e)}\"
                    errors.append(error_msg)
                    logger.error(error_msg)
        
        except Exception as e:
            errors.append(f\"Step 8 validation error: {str(e)}\")
        
        success = len(export_results) > 0 and len(errors) == 0
        message = f\"Exported {len(export_results)} formats successfully\" if success else f\"Export issues: {len(errors)} errors\"
        
        return ValidationResult(
            step=\"Step 8: Data Export\",
            workflow=\"Export\",
            success=success,
            message=message,
            duration=0,
            data={'export_results': export_results},
            errors=errors,
            api_calls=api_calls
        )