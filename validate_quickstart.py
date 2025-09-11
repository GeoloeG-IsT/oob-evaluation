#!/usr/bin/env python3
"""
ML Evaluation Platform - Complete Quickstart Validation Script (T085)

This script validates all 8 quickstart workflows specified in quickstart.md:
1. Upload and organize images 
2. Manual annotation
3. Model selection and assisted annotation
4. Model inference 
5. Performance evaluation
6. Model training/fine-tuning
7. Model deployment
8. Data export

Additionally validates error handling scenarios and generates a comprehensive report.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin

import aiohttp
import asyncpg
import redis
import requests
from PIL import Image
import numpy as np

# Configuration
BASE_DIR = Path(__file__).parent
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ml_eval_platform")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Represents the result of a validation step"""
    step: str
    success: bool
    message: str
    duration: float
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

@dataclass  
class ValidationReport:
    """Complete validation report"""
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[ValidationResult] = field(default_factory=list)
    service_health: Dict[str, bool] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

class QuickstartValidator:
    """Main validation class for ML Evaluation Platform quickstart workflows"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.report = ValidationReport(start_time=datetime.now())
        self.test_images = []
        
    async def setup(self):
        """Initialize connections and prepare test environment"""
        logger.info("Setting up validation environment...")
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Setup test images directory
        test_images_dir = BASE_DIR / "test_images"
        test_images_dir.mkdir(exist_ok=True)
        
        # Generate test images if they don't exist
        await self.generate_test_images(test_images_dir)
        
        logger.info("Validation environment setup complete")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            self.redis_client.close()
    
    async def generate_test_images(self, output_dir: Path):
        """Generate synthetic test images for validation"""
        logger.info("Generating test images...")
        
        # Generate different types of test images
        image_configs = [
            {"name": "test_image_1.jpg", "size": (640, 480), "color": (255, 0, 0)},
            {"name": "test_image_2.png", "size": (800, 600), "color": (0, 255, 0)},
            {"name": "test_image_3.jpg", "size": (1024, 768), "color": (0, 0, 255)},
            {"name": "test_image_4.png", "size": (512, 512), "color": (255, 255, 0)},
            {"name": "test_image_5.jpg", "size": (720, 480), "color": (255, 0, 255)},
        ]
        
        for config in image_configs:
            image_path = output_dir / config["name"]
            if not image_path.exists():
                # Create a simple colored image with some geometric shapes
                img = Image.new('RGB', config["size"], config["color"])
                # Add some simple shapes to make it more realistic for object detection
                pixels = np.array(img)
                h, w = pixels.shape[:2]
                # Add rectangles
                pixels[h//4:3*h//4, w//4:3*w//4] = (128, 128, 128)
                pixels[h//3:2*h//3, w//3:2*w//3] = (64, 64, 64)
                
                img = Image.fromarray(pixels)
                img.save(image_path)
                self.test_images.append(image_path)
                logger.info(f"Generated test image: {image_path}")
    
    async def validate_service_health(self) -> ValidationResult:
        """Validate that all required services are healthy"""
        start_time = time.time()
        errors = []
        service_status = {}
        
        logger.info("Validating service health...")
        
        # Test Backend API
        try:
            async with self.session.get(f"{BACKEND_URL}/health") as response:
                service_status['backend'] = response.status == 200
                if response.status != 200:
                    errors.append(f"Backend API unhealthy: {response.status}")
        except Exception as e:
            service_status['backend'] = False
            errors.append(f"Backend API connection failed: {str(e)}")
        
        # Test Frontend
        try:
            async with self.session.get(FRONTEND_URL) as response:
                service_status['frontend'] = response.status == 200
                if response.status != 200:
                    errors.append(f"Frontend unhealthy: {response.status}")
        except Exception as e:
            service_status['frontend'] = False
            errors.append(f"Frontend connection failed: {str(e)}")
        
        # Test Database
        try:
            self.db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            service_status['database'] = True
        except Exception as e:
            service_status['database'] = False
            errors.append(f"Database connection failed: {str(e)}")
        
        # Test Redis
        try:
            self.redis_client = redis.from_url(REDIS_URL)
            self.redis_client.ping()
            service_status['redis'] = True
        except Exception as e:
            service_status['redis'] = False
            errors.append(f"Redis connection failed: {str(e)}")
        
        self.report.service_health = service_status
        duration = time.time() - start_time
        success = all(service_status.values())
        
        message = "All services healthy" if success else f"Service issues detected: {len(errors)} errors"
        
        return ValidationResult(
            step="Service Health Check",
            success=success,
            message=message,
            duration=duration,
            data=service_status,
            errors=errors
        )
    
    async def validate_step_1_upload_images(self) -> ValidationResult:
        """Validate Step 1: Upload and organize images"""
        start_time = time.time()
        errors = []
        uploaded_images = []
        
        logger.info("Validating Step 1: Upload and organize images...")
        
        try:
            # Test image upload via API
            for i, image_path in enumerate(self.test_images[:3]):
                dataset_split = ['train', 'validation', 'test'][i]
                
                with open(image_path, 'rb') as f:
                    files = {'files': f}
                    data = {'dataset_split': dataset_split}
                    
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/api/v1/images",
                            files=files,
                            data=data,
                            timeout=30
                        )
                        
                        if response.status_code in [200, 201]:
                            result = response.json()
                            uploaded_images.append({
                                'id': result.get('id'),
                                'filename': image_path.name,
                                'dataset_split': dataset_split
                            })
                            logger.info(f"Uploaded {image_path.name} to {dataset_split}")
                        else:
                            errors.append(f"Upload failed for {image_path.name}: {response.status_code}")
                    
                    except Exception as e:
                        errors.append(f"Upload error for {image_path.name}: {str(e)}")
            
            # Verify uploads via GET endpoint
            try:
                response = requests.get(f"{BACKEND_URL}/api/v1/images?limit=10", timeout=15)
                if response.status_code == 200:
                    images_list = response.json()
                    logger.info(f"Retrieved {len(images_list.get('items', []))} uploaded images")
                else:
                    errors.append(f"Failed to list images: {response.status_code}")
            except Exception as e:
                errors.append(f"Failed to retrieve images: {str(e)}")
        
        except Exception as e:
            errors.append(f"Step 1 validation error: {str(e)}")
        
        duration = time.time() - start_time
        success = len(uploaded_images) > 0 and len(errors) == 0
        
        return ValidationResult(
            step="Step 1: Upload and Organize Images",
            success=success,
            message=f"Uploaded {len(uploaded_images)} images" if success else f"{len(errors)} errors occurred",
            duration=duration,
            data={'uploaded_images': uploaded_images},
            errors=errors
        )
    
    async def validate_step_2_manual_annotation(self) -> ValidationResult:
        """Validate Step 2: Manual annotation"""
        start_time = time.time()
        errors = []
        created_annotations = []
        
        logger.info("Validating Step 2: Manual annotation...")
        
        try:
            # Get first uploaded image for annotation
            response = requests.get(f"{BACKEND_URL}/api/v1/images?limit=1", timeout=15)
            if response.status_code != 200:
                errors.append("Could not retrieve images for annotation")
                return ValidationResult(
                    step="Step 2: Manual Annotation",
                    success=False,
                    message="Failed to get images",
                    duration=time.time() - start_time,
                    errors=errors
                )
            
            images = response.json().get('items', [])
            if not images:
                errors.append("No images available for annotation")
                return ValidationResult(
                    step="Step 2: Manual Annotation", 
                    success=False,
                    message="No images available",
                    duration=time.time() - start_time,
                    errors=errors
                )
            
            image_id = images[0]['id']
            
            # Create manual annotation
            annotation_data = {
                "image_id": image_id,
                "bounding_boxes": [{
                    "x": 100,
                    "y": 100, 
                    "width": 200,
                    "height": 150,
                    "class_id": 0,
                    "confidence": 1.0
                }],
                "class_labels": ["test_object"],
                "user_tag": "validation_user"
            }
            
            response = requests.post(
                f"{BACKEND_URL}/api/v1/annotations",
                json=annotation_data,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                annotation_result = response.json()
                created_annotations.append(annotation_result)
                logger.info(f"Created manual annotation: {annotation_result.get('id')}")
            else:
                errors.append(f"Failed to create annotation: {response.status_code}")
        
        except Exception as e:
            errors.append(f"Step 2 validation error: {str(e)}")
        
        duration = time.time() - start_time
        success = len(created_annotations) > 0 and len(errors) == 0
        
        return ValidationResult(
            step="Step 2: Manual Annotation",
            success=success,
            message=f"Created {len(created_annotations)} annotations" if success else f"{len(errors)} errors occurred",
            duration=duration,
            data={'annotations': created_annotations},
            errors=errors
        )
    
    async def validate_remaining_steps(self) -> List[ValidationResult]:
        """Validate remaining steps (3-8) - placeholder implementation"""
        remaining_steps = [
            "Step 3: Model Selection and Assisted Annotation",
            "Step 4: Model Inference", 
            "Step 5: Performance Evaluation",
            "Step 6: Model Training/Fine-tuning",
            "Step 7: Model Deployment",
            "Step 8: Data Export"
        ]
        
        results = []
        
        for step in remaining_steps:
            # For now, mark as pending since full implementation depends on ML models
            result = ValidationResult(
                step=step,
                success=False,
                message="Validation pending - requires ML model implementation", 
                duration=0.1,
                errors=["ML models not yet implemented"]
            )
            results.append(result)
            logger.info(f"Marked {step} as pending")
        
        return results
    
    async def validate_error_handling(self) -> ValidationResult:
        """Validate error handling scenarios"""
        start_time = time.time()
        errors = []
        test_results = []
        
        logger.info("Validating error handling scenarios...")
        
        # Test 1: Unsupported file format
        try:
            # Create a text file and try to upload as image
            text_file = BASE_DIR / "test_images" / "invalid.txt"
            text_file.write_text("This is not an image")
            
            with open(text_file, 'rb') as f:
                files = {'files': f}
                data = {'dataset_split': 'test'}
                
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/images",
                    files=files,
                    data=data,
                    timeout=15
                )
                
                if response.status_code >= 400:
                    test_results.append("✓ Correctly rejected unsupported file format")
                else:
                    errors.append("Failed to reject unsupported file format")
            
            text_file.unlink()  # Clean up
            
        except Exception as e:
            errors.append(f"Error testing file format validation: {str(e)}")
        
        # Test 2: Invalid API request
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/v1/annotations",
                json={"invalid": "data"},
                timeout=15
            )
            
            if response.status_code >= 400:
                test_results.append("✓ Correctly rejected invalid annotation data")
            else:
                errors.append("Failed to reject invalid annotation data")
                
        except Exception as e:
            errors.append(f"Error testing invalid request handling: {str(e)}")
        
        duration = time.time() - start_time
        success = len(test_results) > 0 and len(errors) == 0
        
        return ValidationResult(
            step="Error Handling Validation",
            success=success,
            message=f"Passed {len(test_results)} error handling tests" if success else f"{len(errors)} errors occurred",
            duration=duration,
            data={'test_results': test_results},
            errors=errors
        )
    
    async def generate_report(self) -> str:
        """Generate comprehensive validation report"""
        self.report.end_time = datetime.now()
        total_duration = (self.report.end_time - self.report.start_time).total_seconds()
        
        successful_steps = sum(1 for r in self.report.results if r.success)
        total_steps = len(self.report.results)
        success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
        
        self.report.summary = {
            'total_duration': total_duration,
            'successful_steps': successful_steps,
            'total_steps': total_steps,
            'success_rate': success_rate,
            'service_health': self.report.service_health
        }
        
        # Generate report text
        report_lines = [
            "=" * 80,
            "ML EVALUATION PLATFORM - QUICKSTART VALIDATION REPORT (T085)",
            "=" * 80,
            f"Generated: {self.report.end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Duration: {total_duration:.2f} seconds",
            f"Success Rate: {success_rate:.1f}% ({successful_steps}/{total_steps})",
            "",
            "SERVICE HEALTH:",
            "-" * 40
        ]
        
        for service, healthy in self.report.service_health.items():
            status = "✓ HEALTHY" if healthy else "✗ UNHEALTHY"
            report_lines.append(f"  {service.ljust(15)}: {status}")
        
        report_lines.extend([
            "",
            "VALIDATION RESULTS:",
            "-" * 40
        ])
        
        for result in self.report.results:
            status = "✓ PASS" if result.success else "✗ FAIL"
            report_lines.append(f"  {status} {result.step}")
            report_lines.append(f"      Duration: {result.duration:.2f}s")
            report_lines.append(f"      Message: {result.message}")
            if result.errors:
                report_lines.append(f"      Errors: {', '.join(result.errors)}")
            report_lines.append("")
        
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 40,
            "1. Complete ML model implementation for steps 3-8",
            "2. Deploy services using Docker Compose for full validation",
            "3. Add more comprehensive error handling tests",
            "4. Implement performance benchmarking",
            "5. Add integration tests for real ML workflows",
            "",
            "=" * 80
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report to file
        report_file = BASE_DIR / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_file.write_text(report_text)
        
        logger.info(f"Validation report saved to: {report_file}")
        return report_text
    
    async def run_validation(self) -> str:
        """Execute complete quickstart validation workflow"""
        logger.info("Starting ML Evaluation Platform Quickstart Validation (T085)...")
        
        try:
            await self.setup()
            
            # Execute validation steps
            validation_steps = [
                self.validate_service_health(),
                self.validate_step_1_upload_images(),
                self.validate_step_2_manual_annotation(),
                self.validate_error_handling()
            ]
            
            # Add remaining steps
            for step_result in await self.validate_remaining_steps():
                validation_steps.append(asyncio.coroutine(lambda r=step_result: r)())
            
            # Execute all validations
            for step_coro in validation_steps:
                if asyncio.iscoroutine(step_coro):
                    result = await step_coro
                else:
                    result = step_coro
                    
                self.report.results.append(result)
                
                if result.success:
                    logger.info(f"✓ {result.step} - PASSED ({result.duration:.2f}s)")
                else:
                    logger.warning(f"✗ {result.step} - FAILED ({result.duration:.2f}s): {result.message}")
            
            # Generate final report
            return await self.generate_report()
            
        finally:
            await self.cleanup()

async def main():
    """Main entry point for quickstart validation"""
    validator = QuickstartValidator()
    
    try:
        report = await validator.run_validation()
        print("\n" + "=" * 80)
        print("QUICKSTART VALIDATION COMPLETED")
        print("=" * 80)
        print(report)
        
        # Exit with proper code
        successful_steps = sum(1 for r in validator.report.results if r.success)
        total_steps = len(validator.report.results)
        
        if successful_steps == total_steps:
            print("All validations passed!")
            sys.exit(0)
        else:
            print(f"Some validations failed ({successful_steps}/{total_steps} passed)")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        print(f"VALIDATION ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())