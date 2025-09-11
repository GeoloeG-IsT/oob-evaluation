#!/usr/bin/env python3
"""
API Documentation Generator for ML Evaluation Platform

This script generates comprehensive API documentation from the OpenAPI specification
in multiple formats (HTML, Markdown) with examples and developer guides.
"""

import json
import yaml
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class APIDocumentationGenerator:
    """Generates comprehensive API documentation from OpenAPI specification."""
    
    def __init__(self, spec_path: str):
        """Initialize with OpenAPI specification path."""
        self.spec_path = Path(spec_path)
        self.docs_dir = Path(__file__).parent
        self.spec = self._load_spec()
        
    def _load_spec(self) -> Dict[str, Any]:
        """Load OpenAPI specification from YAML file."""
        try:
            with open(self.spec_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"✗ OpenAPI spec file not found: {self.spec_path}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error loading OpenAPI spec: {e}")
            sys.exit(1)
    
    def generate_all_documentation(self):
        """Generate all documentation formats."""
        print("🚀 Generating comprehensive API documentation...")
        
        # Generate OpenAPI JSON from YAML
        self._generate_openapi_json()
        
        # Generate HTML documentation
        self._generate_html_documentation()
        
        # Generate Markdown documentation
        self._generate_markdown_documentation()
        
        # Generate developer guides
        self._generate_developer_guides()
        
        # Generate request/response examples
        self._generate_examples()
        
        print("✅ All documentation generated successfully!")
    
    def _generate_openapi_json(self):
        """Convert YAML spec to JSON format."""
        json_path = self.docs_dir / "api" / "openapi.json"
        with open(json_path, 'w') as f:
            json.dump(self.spec, f, indent=2)
        print(f"✓ OpenAPI JSON saved to: {json_path}")
    
    def _generate_html_documentation(self):
        """Generate HTML documentation."""
        html_content = self._create_html_documentation()
        html_path = self.docs_dir / "api" / "index.html"
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML documentation saved to: {html_path}")
    
    def _generate_markdown_documentation(self):
        """Generate Markdown documentation."""
        md_content = self._create_markdown_documentation()
        md_path = self.docs_dir / "api" / "README.md"
        
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        print(f"✓ Markdown documentation saved to: {md_path}")
    
    def _generate_developer_guides(self):
        """Generate developer quick-start guides."""
        guides = {
            "quick-start.md": self._create_quickstart_guide(),
            "authentication.md": self._create_authentication_guide(),
            "error-handling.md": self._create_error_handling_guide(),
            "examples.md": self._create_examples_guide(),
        }
        
        guides_dir = self.docs_dir / "guides"
        for filename, content in guides.items():
            guide_path = guides_dir / filename
            with open(guide_path, 'w') as f:
                f.write(content)
            print(f"✓ Guide saved to: {guide_path}")
    
    def _generate_examples(self):
        """Generate request/response examples for all endpoints."""
        examples_dir = self.docs_dir / "examples"
        
        # Generate curl examples
        curl_examples = self._create_curl_examples()
        curl_path = examples_dir / "curl_examples.sh"
        with open(curl_path, 'w') as f:
            f.write(curl_examples)
        print(f"✓ cURL examples saved to: {curl_path}")
        
        # Generate Python examples
        python_examples = self._create_python_examples()
        python_path = examples_dir / "python_examples.py"
        with open(python_path, 'w') as f:
            f.write(python_examples)
        print(f"✓ Python examples saved to: {python_path}")
        
        # Generate request/response examples JSON
        examples_json = self._create_examples_json()
        json_path = examples_dir / "request_response_examples.json"
        with open(json_path, 'w') as f:
            json.dump(examples_json, f, indent=2)
        print(f"✓ Request/response examples saved to: {json_path}")
    
    def _create_html_documentation(self) -> str:
        """Create comprehensive HTML documentation."""
        info = self.spec.get("info", {})
        paths = self.spec.get("paths", {})
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{info.get('title', 'API Documentation')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem; }}
        .endpoint {{ background: #f8f9fa; border-left: 4px solid #007bff; padding: 1rem; margin: 1rem 0; border-radius: 5px; }}
        .method {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 3px; color: white; font-weight: bold; margin-right: 1rem; }}
        .method.get {{ background: #28a745; }}
        .method.post {{ background: #007bff; }}
        .method.put {{ background: #ffc107; color: #000; }}
        .method.patch {{ background: #fd7e14; }}
        .method.delete {{ background: #dc3545; }}
        .schema {{ background: #e9ecef; padding: 1rem; border-radius: 5px; font-family: monospace; white-space: pre-wrap; }}
        .toc {{ background: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 2rem; }}
        .toc ul {{ margin: 0; padding-left: 1.5rem; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background: #f1f3f4; padding: 0.2rem 0.4rem; border-radius: 3px; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{info.get('title', 'API Documentation')}</h1>
        <p>{info.get('description', 'API Documentation')}</p>
        <p><strong>Version:</strong> {info.get('version', '1.0.0')} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
"""
        
        # Generate table of contents
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
                    operation_id = details.get('operationId', f"{method}_{path.replace('/', '_')}")
                    summary = details.get('summary', 'No summary')
                    html += f"            <li><a href=\"#{operation_id}\">{method.upper()} {path}</a> - {summary}</li>\\n"
        
        html += "        </ul>\\n    </div>\\n\\n"
        
        # Generate endpoint documentation
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
                    operation_id = details.get('operationId', f"{method}_{path.replace('/', '_')}")
                    summary = details.get('summary', 'No summary')
                    description = details.get('description', 'No description')
                    
                    html += f"""
    <div id="{operation_id}" class="endpoint">
        <h3><span class="method {method.lower()}">{method.upper()}</span>{path}</h3>
        <h4>{summary}</h4>
        <p>{description}</p>
        
        <h4>Parameters</h4>
"""
                    
                    # Add parameters
                    parameters = details.get('parameters', [])
                    if parameters:
                        html += "        <ul>\\n"
                        for param in parameters:
                            param_name = param.get('name', 'unknown')
                            param_in = param.get('in', 'unknown')
                            param_required = " (required)" if param.get('required', False) else ""
                            param_schema = param.get('schema', {})
                            param_type = param_schema.get('type', 'unknown')
                            html += f"            <li><code>{param_name}</code> ({param_in}) - {param_type}{param_required}</li>\\n"
                        html += "        </ul>\\n"
                    else:
                        html += "        <p>No parameters</p>\\n"
                    
                    # Add request body
                    request_body = details.get('requestBody')
                    if request_body:
                        html += "        <h4>Request Body</h4>\\n"
                        content = request_body.get('content', {})
                        for content_type, content_details in content.items():
                            schema = content_details.get('schema', {})
                            html += f"        <p><strong>Content-Type:</strong> {content_type}</p>\\n"
                            html += f"        <div class=\"schema\">{json.dumps(schema, indent=2)}</div>\\n"
                    
                    # Add responses
                    responses = details.get('responses', {})
                    if responses:
                        html += "        <h4>Responses</h4>\\n"
                        for status_code, response_details in responses.items():
                            description = response_details.get('description', 'No description')
                            html += f"        <h5>Status {status_code}</h5>\\n"
                            html += f"        <p>{description}</p>\\n"
                            
                            content = response_details.get('content', {})
                            for content_type, content_details in content.items():
                                schema = content_details.get('schema', {})
                                html += f"        <p><strong>Content-Type:</strong> {content_type}</p>\\n"
                                html += f"        <div class=\"schema\">{json.dumps(schema, indent=2)}</div>\\n"
                    
                    html += "    </div>\\n"
        
        html += """
</body>
</html>"""
        
        return html
    
    def _create_markdown_documentation(self) -> str:
        """Create comprehensive Markdown documentation."""
        info = self.spec.get("info", {})
        paths = self.spec.get("paths", {})
        
        md = f"""# {info.get('title', 'API Documentation')}

{info.get('description', 'API Documentation')}

**Version:** {info.get('version', '1.0.0')}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Table of Contents

"""
        
        # Generate table of contents
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
                    operation_id = details.get('operationId', f"{method}_{path.replace('/', '_')}")
                    summary = details.get('summary', 'No summary')
                    md += f"- [{method.upper()} {path}](#{operation_id.lower().replace('_', '-')}) - {summary}\\n"
        
        md += "\\n## Endpoints\\n\\n"
        
        # Generate endpoint documentation
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
                    operation_id = details.get('operationId', f"{method}_{path.replace('/', '_')}")
                    summary = details.get('summary', 'No summary')
                    description = details.get('description', 'No description')
                    
                    md += f"""### {method.upper()} {path}

**{summary}**

{description}

**Operation ID:** `{operation_id}`

#### Parameters

"""
                    
                    # Add parameters
                    parameters = details.get('parameters', [])
                    if parameters:
                        md += "| Name | In | Type | Required | Description |\\n"
                        md += "|------|----|----- |----------|-------------|\\n"
                        for param in parameters:
                            param_name = param.get('name', 'unknown')
                            param_in = param.get('in', 'unknown')
                            param_required = "Yes" if param.get('required', False) else "No"
                            param_schema = param.get('schema', {})
                            param_type = param_schema.get('type', 'unknown')
                            param_desc = param.get('description', 'No description')
                            md += f"| `{param_name}` | {param_in} | {param_type} | {param_required} | {param_desc} |\\n"
                    else:
                        md += "No parameters\\n"
                    
                    # Add request body
                    request_body = details.get('requestBody')
                    if request_body:
                        md += "\\n#### Request Body\\n\\n"
                        content = request_body.get('content', {})
                        for content_type, content_details in content.items():
                            schema = content_details.get('schema', {})
                            md += f"**Content-Type:** `{content_type}`\\n\\n"
                            md += f"```json\\n{json.dumps(schema, indent=2)}\\n```\\n\\n"
                    
                    # Add responses
                    responses = details.get('responses', {})
                    if responses:
                        md += "#### Responses\\n\\n"
                        for status_code, response_details in responses.items():
                            description = response_details.get('description', 'No description')
                            md += f"**Status {status_code}**\\n\\n{description}\\n\\n"
                            
                            content = response_details.get('content', {})
                            for content_type, content_details in content.items():
                                schema = content_details.get('schema', {})
                                md += f"**Content-Type:** `{content_type}`\\n\\n"
                                md += f"```json\\n{json.dumps(schema, indent=2)}\\n```\\n\\n"
                    
                    md += "---\\n\\n"
        
        return md
    
    def _create_quickstart_guide(self) -> str:
        """Create a quick-start guide for developers."""
        return """# Quick Start Guide

This guide will help you get started with the ML Evaluation Platform API.

## Base URL

- Development: `http://localhost:8000`
- Production: `https://api.ml-eval.cloud`

## Authentication

Currently, the API does not require authentication for development purposes.

## Common Workflows

### 1. Upload Images

First, upload some images to work with:

```bash
curl -X POST "http://localhost:8000/api/v1/images" \\
  -H "Content-Type: multipart/form-data" \\
  -F "files=@image1.jpg" \\
  -F "files=@image2.jpg" \\
  -F "dataset_split=train"
```

### 2. List Available Models

Check what models are available:

```bash
curl -X GET "http://localhost:8000/api/v1/models"
```

### 3. Run Inference

Run inference on an uploaded image:

```bash
curl -X POST "http://localhost:8000/api/v1/inference/single" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_id": "your-image-id",
    "model_id": "your-model-id",
    "confidence_threshold": 0.5
  }'
```

### 4. Create Manual Annotations

Add manual annotations to images:

```bash
curl -X POST "http://localhost:8000/api/v1/annotations" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_id": "your-image-id",
    "bounding_boxes": [{
      "x": 100,
      "y": 100,
      "width": 200,
      "height": 150,
      "class_id": 1,
      "confidence": 1.0
    }],
    "class_labels": ["person"]
  }'
```

### 5. Start Model Training

Fine-tune a model with your annotations:

```bash
curl -X POST "http://localhost:8000/api/v1/training/jobs" \\
  -H "Content-Type: application/json" \\
  -d '{
    "base_model_id": "your-base-model-id",
    "dataset_id": "your-dataset-id",
    "hyperparameters": {
      "epochs": 100,
      "batch_size": 16,
      "learning_rate": 0.001
    }
  }'
```

## Next Steps

- Check out the [Examples Guide](examples.md) for more detailed examples
- See [Error Handling](error-handling.md) for information about error responses
- Read the full [API Reference](../api/README.md) for complete endpoint documentation

## Support

For questions and support, please refer to the project documentation or open an issue in the repository.
"""
    
    def _create_authentication_guide(self) -> str:
        """Create authentication documentation."""
        return """# Authentication

The ML Evaluation Platform API currently supports development without authentication requirements.

## Authentication Methods

### Development Mode

For development and testing purposes, no authentication is required. All endpoints are accessible without any credentials.

### Production Considerations

In a production environment, the following authentication methods would be implemented:

#### API Key Authentication

API keys would be provided for programmatic access:

```bash
curl -H "X-API-Key: your-api-key" "http://localhost:8000/api/v1/images"
```

#### Bearer Token Authentication

JWT tokens would be supported for user authentication:

```bash
curl -H "Authorization: Bearer your-jwt-token" "http://localhost:8000/api/v1/images"
```

## Security Headers

When authentication is implemented, all requests should include appropriate security headers:

- `X-API-Key`: Your API key
- `Authorization`: Bearer token for JWT authentication
- `Content-Type`: Appropriate content type for the request

## Error Responses

Authentication errors would return:

```json
{
  "error": "Authentication required",
  "message": "Valid API key or bearer token required",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Rate Limiting

Production deployments would include rate limiting:

- 1000 requests per hour per API key
- 10,000 requests per hour for authenticated users
- Burst limits for short-term usage spikes

Rate limit headers would be included in responses:

- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Time when rate limit window resets
"""
    
    def _create_error_handling_guide(self) -> str:
        """Create error handling documentation."""
        return """# Error Handling

This guide explains how to handle errors when working with the ML Evaluation Platform API.

## Error Response Format

All error responses follow a consistent format:

```json
{
  "error": "Error type",
  "message": "Human-readable error message",
  "details": {
    "field": "Additional error details"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## HTTP Status Codes

The API uses standard HTTP status codes:

### 2xx Success

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `202 Accepted`: Request accepted for processing

### 4xx Client Errors

- `400 Bad Request`: Invalid request format or parameters
- `401 Unauthorized`: Authentication required (production)
- `403 Forbidden`: Access denied (production)
- `404 Not Found`: Resource not found
- `413 Payload Too Large`: Request body too large
- `422 Unprocessable Entity`: Validation errors

### 5xx Server Errors

- `500 Internal Server Error`: Unexpected server error
- `502 Bad Gateway`: Upstream service error
- `503 Service Unavailable`: Service temporarily unavailable
- `504 Gateway Timeout`: Upstream service timeout

## Common Error Scenarios

### Invalid Image Upload

```bash
# Request
curl -X POST "http://localhost:8000/api/v1/images" \\
  -F "files=@invalid-file.txt"

# Response (400 Bad Request)
{
  "error": "Invalid file format",
  "message": "Only image files are supported",
  "details": {
    "supported_formats": ["jpg", "jpeg", "png", "tiff", "bmp"]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Resource Not Found

```bash
# Request
curl -X GET "http://localhost:8000/api/v1/images/non-existent-id"

# Response (404 Not Found)
{
  "error": "Resource not found",
  "message": "Image with ID 'non-existent-id' not found",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Validation Errors

```bash
# Request with invalid data
curl -X POST "http://localhost:8000/api/v1/annotations" \\
  -H "Content-Type: application/json" \\
  -d '{"image_id": "invalid-uuid"}'

# Response (422 Unprocessable Entity)
{
  "error": "Validation error",
  "message": "Request validation failed",
  "details": {
    "image_id": "Invalid UUID format",
    "class_labels": "This field is required"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Server Errors

```bash
# Response (500 Internal Server Error)
{
  "error": "Internal server error",
  "message": "An unexpected error occurred",
  "details": {
    "request_id": "req_12345"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Error Handling Best Practices

### 1. Always Check Status Codes

```python
import requests

response = requests.post("http://localhost:8000/api/v1/images", files={"files": open("image.jpg", "rb")})

if response.status_code == 201:
    # Success
    result = response.json()
    print("Images uploaded successfully")
elif response.status_code == 400:
    # Client error
    error_data = response.json()
    print(f"Error: {error_data['message']}")
else:
    # Handle other errors
    print(f"Unexpected error: {response.status_code}")
```

### 2. Implement Retry Logic

```python
import time
import requests
from requests.exceptions import RequestException

def api_request_with_retry(url, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code < 500:  # Don't retry client errors
                return response
        except RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(delay * (2 ** attempt))  # Exponential backoff
    
    raise Exception(f"Request failed after {max_retries} attempts")
```

### 3. Handle Specific Error Cases

```python
def handle_inference_request(image_id, model_id):
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/inference/single",
            json={
                "image_id": image_id,
                "model_id": model_id,
                "confidence_threshold": 0.5
            }
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            error_data = response.json()
            if "image" in error_data["message"].lower():
                raise ValueError("Image not found - please upload the image first")
            elif "model" in error_data["message"].lower():
                raise ValueError("Model not found - please check available models")
        elif response.status_code == 400:
            error_data = response.json()
            raise ValueError(f"Invalid request: {error_data['message']}")
        else:
            response.raise_for_status()
            
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to API: {e}")
```

## Logging and Monitoring

For production applications, implement proper logging:

```python
import logging

logger = logging.getLogger(__name__)

def api_call_with_logging(url, data=None):
    try:
        logger.info(f"Making API call to {url}")
        response = requests.post(url, json=data)
        logger.info(f"API call completed with status {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise
```

## Getting Help

If you encounter persistent errors:

1. Check this error handling guide
2. Verify your request format against the API documentation
3. Check server logs for additional context
4. Open an issue with request/response details
"""
    
    def _create_examples_guide(self) -> str:
        """Create comprehensive examples guide."""
        return """# API Examples

This guide provides comprehensive examples for all major API operations.

## Image Management

### Upload Single Image

```bash
curl -X POST "http://localhost:8000/api/v1/images" \\
  -H "Content-Type: multipart/form-data" \\
  -F "files=@sample_image.jpg" \\
  -F "dataset_split=train"
```

### Upload Multiple Images

```bash
curl -X POST "http://localhost:8000/api/v1/images" \\
  -H "Content-Type: multipart/form-data" \\
  -F "files=@image1.jpg" \\
  -F "files=@image2.jpg" \\
  -F "files=@image3.jpg" \\
  -F "dataset_split=validation"
```

### List Images with Filters

```bash
# Get all training images
curl -X GET "http://localhost:8000/api/v1/images?dataset_split=train&limit=20"

# Get images with pagination
curl -X GET "http://localhost:8000/api/v1/images?limit=50&offset=100"
```

### Get Image Details

```bash
curl -X GET "http://localhost:8000/api/v1/images/{image_id}"
```

## Model Management

### List Available Models

```bash
# Get all models
curl -X GET "http://localhost:8000/api/v1/models"

# Filter by type
curl -X GET "http://localhost:8000/api/v1/models?type=detection"

# Filter by framework
curl -X GET "http://localhost:8000/api/v1/models?framework=YOLO11"
```

### Get Model Details

```bash
curl -X GET "http://localhost:8000/api/v1/models/{model_id}"
```

## Annotations

### Create Manual Annotation

```bash
curl -X POST "http://localhost:8000/api/v1/annotations" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_id": "123e4567-e89b-12d3-a456-426614174000",
    "bounding_boxes": [
      {
        "x": 100,
        "y": 50,
        "width": 200,
        "height": 150,
        "class_id": 0,
        "confidence": 1.0
      }
    ],
    "segments": [
      {
        "polygon": [[100, 50], [300, 50], [300, 200], [100, 200]],
        "class_id": 0,
        "confidence": 1.0
      }
    ],
    "class_labels": ["person"],
    "user_tag": "manual_annotation",
    "metadata": {
      "annotator": "user123",
      "annotation_time": 45
    }
  }'
```

### Generate Assisted Annotation

```bash
curl -X POST "http://localhost:8000/api/v1/annotations/assisted" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_id": "123e4567-e89b-12d3-a456-426614174000",
    "model_id": "456e7890-e89b-12d3-a456-426614174000",
    "confidence_threshold": 0.7
  }'
```

### List Annotations

```bash
# Get all annotations for an image
curl -X GET "http://localhost:8000/api/v1/annotations?image_id={image_id}"

# Get annotations by creation method
curl -X GET "http://localhost:8000/api/v1/annotations?creation_method=user"
```

## Inference

### Single Image Inference

```bash
curl -X POST "http://localhost:8000/api/v1/inference/single" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_id": "123e4567-e89b-12d3-a456-426614174000",
    "model_id": "456e7890-e89b-12d3-a456-426614174000",
    "confidence_threshold": 0.5
  }'
```

### Batch Inference

```bash
curl -X POST "http://localhost:8000/api/v1/inference/batch" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_ids": [
      "123e4567-e89b-12d3-a456-426614174000",
      "234e5678-e89b-12d3-a456-426614174001",
      "345e6789-e89b-12d3-a456-426614174002"
    ],
    "model_id": "456e7890-e89b-12d3-a456-426614174000",
    "confidence_threshold": 0.6
  }'
```

### Monitor Inference Job

```bash
curl -X GET "http://localhost:8000/api/v1/inference/jobs/{job_id}"
```

## Training

### Start Model Training

```bash
curl -X POST "http://localhost:8000/api/v1/training/jobs" \\
  -H "Content-Type: application/json" \\
  -d '{
    "base_model_id": "456e7890-e89b-12d3-a456-426614174000",
    "dataset_id": "789e0123-e89b-12d3-a456-426614174000",
    "hyperparameters": {
      "epochs": 50,
      "batch_size": 16,
      "learning_rate": 0.001,
      "patience": 10
    },
    "metadata": {
      "experiment_name": "custom_training_v1",
      "description": "Fine-tuning YOLO11 on custom dataset"
    }
  }'
```

### Monitor Training Job

```bash
curl -X GET "http://localhost:8000/api/v1/training/jobs/{job_id}"
```

## Performance Evaluation

### Calculate Metrics

```bash
curl -X POST "http://localhost:8000/api/v1/evaluation/metrics" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "456e7890-e89b-12d3-a456-426614174000",
    "dataset_id": "789e0123-e89b-12d3-a456-426614174000",
    "metric_types": ["mAP", "mAP@50", "precision", "recall", "F1"],
    "iou_threshold": 0.5
  }'
```

### Compare Models

```bash
curl -X POST "http://localhost:8000/api/v1/evaluation/compare" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_ids": [
      "456e7890-e89b-12d3-a456-426614174000",
      "567e8901-e89b-12d3-a456-426614174001",
      "678e9012-e89b-12d3-a456-426614174002"
    ],
    "dataset_id": "789e0123-e89b-12d3-a456-426614174000",
    "metric_types": ["mAP", "execution_time"]
  }'
```

## Model Deployment

### Deploy Model

```bash
curl -X POST "http://localhost:8000/api/v1/deployments" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "456e7890-e89b-12d3-a456-426614174000",
    "version": "v1.0.0",
    "configuration": {
      "replicas": 2,
      "cpu_limit": "2000m",
      "memory_limit": "4Gi",
      "gpu_required": true
    },
    "metadata": {
      "environment": "production",
      "description": "Production deployment of custom trained model"
    }
  }'
```

### List Deployments

```bash
curl -X GET "http://localhost:8000/api/v1/deployments"
```

### Update Deployment

```bash
curl -X PATCH "http://localhost:8000/api/v1/deployments/{deployment_id}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "status": "inactive",
    "configuration": {
      "replicas": 1
    }
  }'
```

## Data Export

### Export Annotations

```bash
# Export in COCO format
curl -X POST "http://localhost:8000/api/v1/export/annotations" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_ids": [
      "123e4567-e89b-12d3-a456-426614174000",
      "234e5678-e89b-12d3-a456-426614174001"
    ],
    "format": "COCO",
    "include_predictions": true,
    "model_id": "456e7890-e89b-12d3-a456-426614174000"
  }' \\
  --output annotations_export.zip

# Export in YOLO format
curl -X POST "http://localhost:8000/api/v1/export/annotations" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_ids": [
      "123e4567-e89b-12d3-a456-426614174000"
    ],
    "format": "YOLO",
    "include_predictions": false
  }' \\
  --output yolo_annotations.zip
```

## Python SDK Examples

### Basic Usage

```python
import requests
import json

BASE_URL = "http://localhost:8000"

class MLEvaluationClient:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def upload_image(self, file_path, dataset_split="train"):
        url = f"{self.base_url}/api/v1/images"
        with open(file_path, 'rb') as f:
            files = {"files": f}
            data = {"dataset_split": dataset_split}
            response = self.session.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    
    def run_inference(self, image_id, model_id, confidence_threshold=0.5):
        url = f"{self.base_url}/api/v1/inference/single"
        data = {
            "image_id": image_id,
            "model_id": model_id,
            "confidence_threshold": confidence_threshold
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

# Usage example
client = MLEvaluationClient()

# Upload image
result = client.upload_image("sample_image.jpg", "test")
image_id = result["uploaded_images"][0]["id"]

# Run inference
inference_result = client.run_inference(image_id, "model-id-here")
print(f"Found {len(inference_result['predictions'])} objects")
```

For more examples and detailed integration patterns, see the [Quick Start Guide](quick-start.md).
"""
    
    def _create_curl_examples(self) -> str:
        """Create comprehensive cURL examples."""
        return """#!/bin/bash

# ML Evaluation Platform API - cURL Examples
# 
# This script contains example API calls using cURL for all major endpoints.
# Replace placeholder values (marked with {}) with actual IDs from your system.

BASE_URL="http://localhost:8000"

echo "=== ML Evaluation Platform API Examples ==="
echo

# 1. UPLOAD IMAGES
echo "1. Uploading sample images..."
curl -X POST "$BASE_URL/api/v1/images" \\
  -H "Content-Type: multipart/form-data" \\
  -F "files=@sample_image1.jpg" \\
  -F "files=@sample_image2.jpg" \\
  -F "dataset_split=train"

echo
echo "---"

# 2. LIST IMAGES
echo "2. Listing uploaded images..."
curl -X GET "$BASE_URL/api/v1/images?limit=10"

echo
echo "---"

# 3. GET IMAGE DETAILS
echo "3. Getting image details..."
curl -X GET "$BASE_URL/api/v1/images/{image_id}"

echo
echo "---"

# 4. LIST MODELS
echo "4. Listing available models..."
curl -X GET "$BASE_URL/api/v1/models"

echo
echo "---"

# 5. CREATE MANUAL ANNOTATION
echo "5. Creating manual annotation..."
curl -X POST "$BASE_URL/api/v1/annotations" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_id": "{image_id}",
    "bounding_boxes": [{
      "x": 100,
      "y": 100,
      "width": 200,
      "height": 150,
      "class_id": 1,
      "confidence": 1.0
    }],
    "class_labels": ["person"],
    "user_tag": "manual_example"
  }'

echo
echo "---"

# 6. GENERATE ASSISTED ANNOTATION
echo "6. Generating assisted annotation..."
curl -X POST "$BASE_URL/api/v1/annotations/assisted" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_id": "{image_id}",
    "model_id": "{model_id}",
    "confidence_threshold": 0.7
  }'

echo
echo "---"

# 7. SINGLE IMAGE INFERENCE
echo "7. Running single image inference..."
curl -X POST "$BASE_URL/api/v1/inference/single" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_id": "{image_id}",
    "model_id": "{model_id}",
    "confidence_threshold": 0.5
  }'

echo
echo "---"

# 8. BATCH INFERENCE
echo "8. Starting batch inference job..."
curl -X POST "$BASE_URL/api/v1/inference/batch" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_ids": ["{image_id_1}", "{image_id_2}"],
    "model_id": "{model_id}",
    "confidence_threshold": 0.6
  }'

echo
echo "---"

# 9. START TRAINING JOB
echo "9. Starting model training job..."
curl -X POST "$BASE_URL/api/v1/training/jobs" \\
  -H "Content-Type: application/json" \\
  -d '{
    "base_model_id": "{base_model_id}",
    "dataset_id": "{dataset_id}",
    "hyperparameters": {
      "epochs": 50,
      "batch_size": 16,
      "learning_rate": 0.001,
      "patience": 10
    }
  }'

echo
echo "---"

# 10. CALCULATE PERFORMANCE METRICS
echo "10. Calculating performance metrics..."
curl -X POST "$BASE_URL/api/v1/evaluation/metrics" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "{model_id}",
    "dataset_id": "{dataset_id}",
    "metric_types": ["mAP", "mAP@50", "precision", "recall", "F1"],
    "iou_threshold": 0.5
  }'

echo
echo "---"

# 11. COMPARE MODELS
echo "11. Comparing model performance..."
curl -X POST "$BASE_URL/api/v1/evaluation/compare" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_ids": ["{model_id_1}", "{model_id_2}"],
    "dataset_id": "{dataset_id}",
    "metric_types": ["mAP", "execution_time"]
  }'

echo
echo "---"

# 12. DEPLOY MODEL
echo "12. Deploying model..."
curl -X POST "$BASE_URL/api/v1/deployments" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "{model_id}",
    "version": "v1.0.0",
    "configuration": {
      "replicas": 1,
      "cpu_limit": "1000m",
      "memory_limit": "2Gi",
      "gpu_required": false
    }
  }'

echo
echo "---"

# 13. LIST DEPLOYMENTS
echo "13. Listing deployments..."
curl -X GET "$BASE_URL/api/v1/deployments"

echo
echo "---"

# 14. EXPORT ANNOTATIONS
echo "14. Exporting annotations..."
curl -X POST "$BASE_URL/api/v1/export/annotations" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_ids": ["{image_id}"],
    "format": "COCO",
    "include_predictions": false
  }' \\
  --output exported_annotations.zip

echo
echo "=== Examples completed ==="
echo "Remember to replace {placeholder} values with actual IDs from your system!"
"""
    
    def _create_python_examples(self) -> str:
        """Create comprehensive Python examples."""
        return '''"""
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
    
    print("=== ML Evaluation Platform Python Examples ===\\n")
    
    try:
        # 1. Upload images
        print("1. Uploading sample images...")
        image_files = ["sample1.jpg", "sample2.jpg"]  # Replace with actual file paths
        # upload_result = client.upload_images(image_files, dataset_split="train")
        # print(f"   Uploaded {upload_result['success_count']} images")
        
        # 2. List available models
        print("\\n2. Listing available models...")
        models_result = client.list_models()
        print(f"   Found {models_result['total_count']} models")
        
        # 3. Create manual annotation (using placeholder IDs)
        print("\\n3. Creating manual annotation...")
        # annotation_result = client.create_annotation(
        #     image_id="placeholder-image-id",
        #     class_labels=["person", "car"],
        #     bounding_boxes=[
        #         {"x": 100, "y": 100, "width": 200, "height": 150, "class_id": 0, "confidence": 1.0}
        #     ]
        # )
        # print(f"   Created annotation: {annotation_result['id']}")
        
        # 4. Run single inference (using placeholder IDs)
        print("\\n4. Running single image inference...")
        # inference_result = client.run_single_inference(
        #     image_id="placeholder-image-id",
        #     model_id="placeholder-model-id",
        #     confidence_threshold=0.5
        # )
        # print(f"   Found {len(inference_result['predictions'])} objects")
        
        # 5. Start batch inference
        print("\\n5. Starting batch inference...")
        # batch_job = client.run_batch_inference(
        #     image_ids=["id1", "id2", "id3"],
        #     model_id="placeholder-model-id"
        # )
        # print(f"   Started batch job: {batch_job['id']}")
        
        # 6. Wait for job completion
        # print("\\n6. Waiting for batch job completion...")
        # final_job = client.wait_for_inference_job(batch_job['id'])
        # print(f"   Job completed with status: {final_job['status']}")
        
        # 7. Calculate performance metrics
        print("\\n7. Calculating performance metrics...")
        # metrics_result = client.calculate_metrics(
        #     model_id="placeholder-model-id",
        #     dataset_id="placeholder-dataset-id",
        #     metric_types=["mAP", "precision", "recall"]
        # )
        # print(f"   Calculated {len(metrics_result['metrics'])} metrics")
        
        # 8. Deploy model
        print("\\n8. Deploying model...")
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
        
        print("\\n=== Examples completed successfully! ===")
        print("Note: Uncomment sections and replace placeholder IDs with actual values to run.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
'''
    
    def _create_examples_json(self) -> Dict[str, Any]:
        """Create comprehensive request/response examples."""
        return {
            "endpoints": {
                "POST /api/v1/images": {
                    "description": "Upload images to the platform",
                    "request": {
                        "content_type": "multipart/form-data",
                        "example": {
                            "files": ["@sample_image1.jpg", "@sample_image2.jpg"],
                            "dataset_split": "train"
                        }
                    },
                    "response": {
                        "status": 201,
                        "content_type": "application/json",
                        "example": {
                            "uploaded_images": [
                                {
                                    "id": "123e4567-e89b-12d3-a456-426614174000",
                                    "filename": "sample_image1.jpg",
                                    "file_path": "/uploads/sample_image1.jpg",
                                    "file_size": 1024000,
                                    "format": "JPEG",
                                    "width": 1920,
                                    "height": 1080,
                                    "dataset_split": "train",
                                    "upload_timestamp": "2024-01-15T10:30:00Z",
                                    "metadata": {}
                                }
                            ],
                            "total_count": 2,
                            "success_count": 2,
                            "failed_count": 0
                        }
                    }
                },
                "GET /api/v1/images": {
                    "description": "List uploaded images",
                    "request": {
                        "parameters": {
                            "dataset_split": "train",
                            "limit": 10,
                            "offset": 0
                        }
                    },
                    "response": {
                        "status": 200,
                        "example": {
                            "images": [
                                {
                                    "id": "123e4567-e89b-12d3-a456-426614174000",
                                    "filename": "sample_image1.jpg",
                                    "file_path": "/uploads/sample_image1.jpg",
                                    "file_size": 1024000,
                                    "format": "JPEG",
                                    "width": 1920,
                                    "height": 1080,
                                    "dataset_split": "train",
                                    "upload_timestamp": "2024-01-15T10:30:00Z",
                                    "metadata": {}
                                }
                            ],
                            "total_count": 1,
                            "limit": 10,
                            "offset": 0
                        }
                    }
                },
                "POST /api/v1/annotations": {
                    "description": "Create manual annotation",
                    "request": {
                        "content_type": "application/json",
                        "example": {
                            "image_id": "123e4567-e89b-12d3-a456-426614174000",
                            "bounding_boxes": [
                                {
                                    "x": 100,
                                    "y": 50,
                                    "width": 200,
                                    "height": 150,
                                    "class_id": 0,
                                    "confidence": 1.0
                                }
                            ],
                            "segments": [
                                {
                                    "polygon": [[100, 50], [300, 50], [300, 200], [100, 200]],
                                    "class_id": 0,
                                    "confidence": 1.0
                                }
                            ],
                            "class_labels": ["person"],
                            "user_tag": "manual_annotation",
                            "metadata": {
                                "annotator": "user123",
                                "annotation_time": 45
                            }
                        }
                    },
                    "response": {
                        "status": 201,
                        "example": {
                            "id": "456e7890-e89b-12d3-a456-426614174001",
                            "image_id": "123e4567-e89b-12d3-a456-426614174000",
                            "bounding_boxes": [
                                {
                                    "x": 100,
                                    "y": 50,
                                    "width": 200,
                                    "height": 150,
                                    "class_id": 0,
                                    "confidence": 1.0
                                }
                            ],
                            "segments": [
                                {
                                    "polygon": [[100, 50], [300, 50], [300, 200], [100, 200]],
                                    "class_id": 0,
                                    "confidence": 1.0
                                }
                            ],
                            "class_labels": ["person"],
                            "confidence_scores": [1.0],
                            "creation_method": "user",
                            "model_id": None,
                            "user_tag": "manual_annotation",
                            "created_at": "2024-01-15T10:35:00Z",
                            "metadata": {
                                "annotator": "user123",
                                "annotation_time": 45
                            }
                        }
                    }
                },
                "POST /api/v1/inference/single": {
                    "description": "Run inference on single image",
                    "request": {
                        "content_type": "application/json",
                        "example": {
                            "image_id": "123e4567-e89b-12d3-a456-426614174000",
                            "model_id": "789e0123-e89b-12d3-a456-426614174002",
                            "confidence_threshold": 0.5
                        }
                    },
                    "response": {
                        "status": 200,
                        "example": {
                            "image_id": "123e4567-e89b-12d3-a456-426614174000",
                            "model_id": "789e0123-e89b-12d3-a456-426614174002",
                            "predictions": [
                                {
                                    "id": "pred-456e7890-e89b-12d3-a456-426614174001",
                                    "image_id": "123e4567-e89b-12d3-a456-426614174000",
                                    "bounding_boxes": [
                                        {
                                            "x": 120,
                                            "y": 60,
                                            "width": 180,
                                            "height": 140,
                                            "class_id": 0,
                                            "confidence": 0.85
                                        }
                                    ],
                                    "segments": [],
                                    "class_labels": ["person"],
                                    "confidence_scores": [0.85],
                                    "creation_method": "model",
                                    "model_id": "789e0123-e89b-12d3-a456-426614174002",
                                    "user_tag": None,
                                    "created_at": "2024-01-15T10:40:00Z",
                                    "metadata": {}
                                }
                            ],
                            "execution_time": 0.125,
                            "timestamp": "2024-01-15T10:40:00Z"
                        }
                    }
                },
                "POST /api/v1/training/jobs": {
                    "description": "Start model training job",
                    "request": {
                        "content_type": "application/json",
                        "example": {
                            "base_model_id": "789e0123-e89b-12d3-a456-426614174002",
                            "dataset_id": "abc1234-e89b-12d3-a456-426614174003",
                            "hyperparameters": {
                                "epochs": 50,
                                "batch_size": 16,
                                "learning_rate": 0.001,
                                "patience": 10
                            },
                            "metadata": {
                                "experiment_name": "custom_training_v1",
                                "description": "Fine-tuning YOLO11 on custom dataset"
                            }
                        }
                    },
                    "response": {
                        "status": 202,
                        "example": {
                            "id": "train-def5678-e89b-12d3-a456-426614174004",
                            "base_model_id": "789e0123-e89b-12d3-a456-426614174002",
                            "dataset_id": "abc1234-e89b-12d3-a456-426614174003",
                            "status": "queued",
                            "progress_percentage": 0.0,
                            "hyperparameters": {
                                "epochs": 50,
                                "batch_size": 16,
                                "learning_rate": 0.001,
                                "patience": 10
                            },
                            "execution_logs": "",
                            "start_time": None,
                            "end_time": None,
                            "result_model_id": None,
                            "created_at": "2024-01-15T10:45:00Z",
                            "metadata": {
                                "experiment_name": "custom_training_v1",
                                "description": "Fine-tuning YOLO11 on custom dataset"
                            }
                        }
                    }
                },
                "POST /api/v1/deployments": {
                    "description": "Deploy a model",
                    "request": {
                        "content_type": "application/json",
                        "example": {
                            "model_id": "789e0123-e89b-12d3-a456-426614174002",
                            "version": "v1.0.0",
                            "configuration": {
                                "replicas": 2,
                                "cpu_limit": "2000m",
                                "memory_limit": "4Gi",
                                "gpu_required": True
                            },
                            "metadata": {
                                "environment": "production",
                                "description": "Production deployment of custom trained model"
                            }
                        }
                    },
                    "response": {
                        "status": 202,
                        "example": {
                            "id": "deploy-ghi9012-e89b-12d3-a456-426614174005",
                            "model_id": "789e0123-e89b-12d3-a456-426614174002",
                            "endpoint_url": "https://api.ml-eval.cloud/models/deploy-ghi9012/predict",
                            "version": "v1.0.0",
                            "status": "deploying",
                            "configuration": {
                                "replicas": 2,
                                "cpu_limit": "2000m",
                                "memory_limit": "4Gi",
                                "gpu_required": True
                            },
                            "performance_monitoring": {},
                            "created_at": "2024-01-15T10:50:00Z",
                            "updated_at": "2024-01-15T10:50:00Z",
                            "metadata": {
                                "environment": "production",
                                "description": "Production deployment of custom trained model"
                            }
                        }
                    }
                }
            },
            "error_examples": {
                "400_bad_request": {
                    "description": "Invalid request format or parameters",
                    "example": {
                        "error": "Invalid file format",
                        "message": "Only image files are supported",
                        "details": {
                            "supported_formats": ["jpg", "jpeg", "png", "tiff", "bmp"]
                        },
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                },
                "404_not_found": {
                    "description": "Resource not found",
                    "example": {
                        "error": "Resource not found",
                        "message": "Image with ID '123e4567-e89b-12d3-a456-426614174000' not found",
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                },
                "422_validation_error": {
                    "description": "Request validation failed",
                    "example": {
                        "error": "Validation error",
                        "message": "Request validation failed",
                        "details": {
                            "image_id": "Invalid UUID format",
                            "class_labels": "This field is required"
                        },
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                },
                "500_server_error": {
                    "description": "Internal server error",
                    "example": {
                        "error": "Internal server error",
                        "message": "An unexpected error occurred",
                        "details": {
                            "request_id": "req_12345"
                        },
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        }


if __name__ == "__main__":
    # Generate documentation using the existing contract specification
    spec_path = "/home/pascal/wks/oob-evaluation-claude/specs/001-oob-evaluation-claude/contracts/api-spec.yaml"
    
    generator = APIDocumentationGenerator(spec_path)
    generator.generate_all_documentation()