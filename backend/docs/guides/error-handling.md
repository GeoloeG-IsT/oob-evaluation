# Error Handling

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
curl -X POST "http://localhost:8000/api/v1/images" \
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
curl -X POST "http://localhost:8000/api/v1/annotations" \
  -H "Content-Type: application/json" \
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
