"""
ML Evaluation Platform API - Python Examples

This module provides example Python code for interacting with all major API endpoints.
"""

import requests
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path


class MLEvaluationClient:
    """Python client for the ML Evaluation Platform API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client with base URL."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MLEvaluation-Python-Client/1.0.0'
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        
        if not response.ok:
            try:
                error_data = response.json()
                error_msg = error_data.get('message', 'Unknown error')
            except:
                error_msg = response.text
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
        
        return response
    
    # Image Management
    def upload_images(self, file_paths: List[str], dataset_split: str = "train") -> Dict[str, Any]:
        """Upload one or more images."""
        files = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            files.append(('files', open(path, 'rb')))
        
        try:
            data = {'dataset_split': dataset_split}
            response = self._make_request('POST', '/api/v1/images', files=files, data=data)
            return response.json()
        finally:
            # Close all file handles
            for _, file_handle in files:
                file_handle.close()
    
    def list_images(self, dataset_split: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List uploaded images with optional filtering."""
        params = {'limit': limit, 'offset': offset}
        if dataset_split:
            params['dataset_split'] = dataset_split
        
        response = self._make_request('GET', '/api/v1/images', params=params)
        return response.json()
    
    def get_image(self, image_id: str) -> Dict[str, Any]:
        """Get details for a specific image."""
        response = self._make_request('GET', f'/api/v1/images/{image_id}')
        return response.json()
    
    # Model Management
    def list_models(self, model_type: Optional[str] = None, framework: Optional[str] = None) -> Dict[str, Any]:
        """List available models."""
        params = {}
        if model_type:
            params['type'] = model_type
        if framework:
            params['framework'] = framework
        
        response = self._make_request('GET', '/api/v1/models', params=params)
        return response.json()
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get details for a specific model."""
        response = self._make_request('GET', f'/api/v1/models/{model_id}')
        return response.json()
    
    # Annotations
    def create_annotation(self, image_id: str, class_labels: List[str], 
                         bounding_boxes: Optional[List[Dict]] = None,
                         segments: Optional[List[Dict]] = None,
                         user_tag: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create manual annotation."""
        data = {
            'image_id': image_id,
            'class_labels': class_labels
        }
        
        if bounding_boxes:
            data['bounding_boxes'] = bounding_boxes
        if segments:
            data['segments'] = segments
        if user_tag:
            data['user_tag'] = user_tag
        if metadata:
            data['metadata'] = metadata
        
        response = self._make_request('POST', '/api/v1/annotations', json=data)
        return response.json()
    
    def generate_assisted_annotation(self, image_id: str, model_id: str, 
                                   confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Generate assisted annotation using a pre-trained model."""
        data = {
            'image_id': image_id,
            'model_id': model_id,
            'confidence_threshold': confidence_threshold
        }
        
        response = self._make_request('POST', '/api/v1/annotations/assisted', json=data)
        return response.json()
    
    def list_annotations(self, image_id: Optional[str] = None, 
                        model_id: Optional[str] = None,
                        creation_method: Optional[str] = None) -> Dict[str, Any]:
        """List annotations with optional filtering."""
        params = {}
        if image_id:
            params['image_id'] = image_id
        if model_id:
            params['model_id'] = model_id
        if creation_method:
            params['creation_method'] = creation_method
        
        response = self._make_request('GET', '/api/v1/annotations', params=params)
        return response.json()
    
    # Inference
    def run_single_inference(self, image_id: str, model_id: str, 
                           confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Run inference on a single image."""
        data = {
            'image_id': image_id,
            'model_id': model_id,
            'confidence_threshold': confidence_threshold
        }
        
        response = self._make_request('POST', '/api/v1/inference/single', json=data)
        return response.json()
    
    def run_batch_inference(self, image_ids: List[str], model_id: str,
                          confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Start batch inference job."""
        data = {
            'image_ids': image_ids,
            'model_id': model_id,
            'confidence_threshold': confidence_threshold
        }
        
        response = self._make_request('POST', '/api/v1/inference/batch', json=data)
        return response.json()
    
    def get_inference_job(self, job_id: str) -> Dict[str, Any]:
        """Get status of inference job."""
        response = self._make_request('GET', f'/api/v1/inference/jobs/{job_id}')
        return response.json()
    
    def wait_for_inference_job(self, job_id: str, timeout: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        """Wait for inference job to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job_status = self.get_inference_job(job_id)
            status = job_status.get('status', 'unknown')
            
            if status in ['completed', 'failed']:
                return job_status
            
            print(f"Job {job_id} status: {status} ({job_status.get('progress_percentage', 0):.1f}%)")
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Inference job {job_id} did not complete within {timeout} seconds")
    
    # Training
    def start_training(self, base_model_id: str, dataset_id: str, 
                      hyperparameters: Dict[str, Any],
                      metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Start model training job."""
        data = {
            'base_model_id': base_model_id,
            'dataset_id': dataset_id,
            'hyperparameters': hyperparameters
        }
        
        if metadata:
            data['metadata'] = metadata
        
        response = self._make_request('POST', '/api/v1/training/jobs', json=data)
        return response.json()
    
    def get_training_job(self, job_id: str) -> Dict[str, Any]:
        """Get training job status."""
        response = self._make_request('GET', f'/api/v1/training/jobs/{job_id}')
        return response.json()
    
    def wait_for_training_job(self, job_id: str, timeout: int = 3600, poll_interval: int = 30) -> Dict[str, Any]:
        """Wait for training job to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job_status = self.get_training_job(job_id)
            status = job_status.get('status', 'unknown')
            
            if status in ['completed', 'failed']:
                return job_status
            
            progress = job_status.get('progress_percentage', 0)
            print(f"Training job {job_id}: {status} ({progress:.1f}%)")
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Training job {job_id} did not complete within {timeout} seconds")
    
    # Evaluation
    def calculate_metrics(self, model_id: str, dataset_id: str, 
                         metric_types: List[str],
                         iou_threshold: float = 0.5) -> Dict[str, Any]:
        """Calculate performance metrics."""
        data = {
            'model_id': model_id,
            'dataset_id': dataset_id,
            'metric_types': metric_types,
            'iou_threshold': iou_threshold
        }
        
        response = self._make_request('POST', '/api/v1/evaluation/metrics', json=data)
        return response.json()
    
    def compare_models(self, model_ids: List[str], dataset_id: str,
                      metric_types: List[str]) -> Dict[str, Any]:
        """Compare performance of multiple models."""
        data = {
            'model_ids': model_ids,
            'dataset_id': dataset_id,
            'metric_types': metric_types
        }
        
        response = self._make_request('POST', '/api/v1/evaluation/compare', json=data)
        return response.json()
    
    # Deployment
    def deploy_model(self, model_id: str, version: str, 
                    configuration: Dict[str, Any],
                    metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Deploy a model."""
        data = {
            'model_id': model_id,
            'version': version,
            'configuration': configuration
        }
        
        if metadata:
            data['metadata'] = metadata
        
        response = self._make_request('POST', '/api/v1/deployments', json=data)
        return response.json()
    
    def list_deployments(self) -> Dict[str, Any]:
        """List all deployments."""
        response = self._make_request('GET', '/api/v1/deployments')
        return response.json()
    
    def get_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment details."""
        response = self._make_request('GET', f'/api/v1/deployments/{deployment_id}')
        return response.json()
    
    def update_deployment(self, deployment_id: str, 
                         status: Optional[str] = None,
                         configuration: Optional[Dict] = None) -> Dict[str, Any]:
        """Update deployment configuration."""
        data = {}
        if status:
            data['status'] = status
        if configuration:
            data['configuration'] = configuration
        
        response = self._make_request('PATCH', f'/api/v1/deployments/{deployment_id}', json=data)
        return response.json()
    
    # Export
    def export_annotations(self, image_ids: List[str], format: str,
                          include_predictions: bool = False,
                          model_id: Optional[str] = None) -> bytes:
        """Export annotations in specified format."""
        data = {
            'image_ids': image_ids,
            'format': format,
            'include_predictions': include_predictions
        }
        
        if model_id:
            data['model_id'] = model_id
        
        response = self._make_request('POST', '/api/v1/export/annotations', json=data)
        return response.content


# Example usage and workflows
def main():
    """Demonstrate common workflows using the ML Evaluation Platform API."""
    
    # Initialize client
    client = MLEvaluationClient()
    
    print("=== ML Evaluation Platform Python Examples ===\n")
    
    try:
        # 1. Upload images
        print("1. Uploading sample images...")
        image_files = ["sample1.jpg", "sample2.jpg"]  # Replace with actual file paths
        # upload_result = client.upload_images(image_files, dataset_split="train")
        # print(f"   Uploaded {upload_result['success_count']} images")
        
        # 2. List available models
        print("\n2. Listing available models...")
        models_result = client.list_models()
        print(f"   Found {models_result['total_count']} models")
        
        # 3. Create manual annotation (using placeholder IDs)
        print("\n3. Creating manual annotation...")
        # annotation_result = client.create_annotation(
        #     image_id="placeholder-image-id",
        #     class_labels=["person", "car"],
        #     bounding_boxes=[
        #         {"x": 100, "y": 100, "width": 200, "height": 150, "class_id": 0, "confidence": 1.0}
        #     ]
        # )
        # print(f"   Created annotation: {annotation_result['id']}")
        
        # 4. Run single inference (using placeholder IDs)
        print("\n4. Running single image inference...")
        # inference_result = client.run_single_inference(
        #     image_id="placeholder-image-id",
        #     model_id="placeholder-model-id",
        #     confidence_threshold=0.5
        # )
        # print(f"   Found {len(inference_result['predictions'])} objects")
        
        # 5. Start batch inference
        print("\n5. Starting batch inference...")
        # batch_job = client.run_batch_inference(
        #     image_ids=["id1", "id2", "id3"],
        #     model_id="placeholder-model-id"
        # )
        # print(f"   Started batch job: {batch_job['id']}")
        
        # 6. Wait for job completion
        # print("\n6. Waiting for batch job completion...")
        # final_job = client.wait_for_inference_job(batch_job['id'])
        # print(f"   Job completed with status: {final_job['status']}")
        
        # 7. Calculate performance metrics
        print("\n7. Calculating performance metrics...")
        # metrics_result = client.calculate_metrics(
        #     model_id="placeholder-model-id",
        #     dataset_id="placeholder-dataset-id",
        #     metric_types=["mAP", "precision", "recall"]
        # )
        # print(f"   Calculated {len(metrics_result['metrics'])} metrics")
        
        # 8. Deploy model
        print("\n8. Deploying model...")
        # deployment_result = client.deploy_model(
        #     model_id="placeholder-model-id",
        #     version="v1.0.0",
        #     configuration={
        #         "replicas": 1,
        #         "cpu_limit": "1000m",
        #         "memory_limit": "2Gi",
        #         "gpu_required": False
        #     }
        # )
        # print(f"   Deployed model: {deployment_result['id']}")
        
        print("\n=== Examples completed successfully! ===")
        print("Note: Uncomment sections and replace placeholder IDs with actual values to run.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
