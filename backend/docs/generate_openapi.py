#!/usr/bin/env python3
"""
OpenAPI Specification Generator for ML Evaluation Platform

This script generates the OpenAPI specification from the FastAPI application
and saves it in multiple formats for documentation purposes.
"""

import json
import sys
import os
from pathlib import Path

# Add the backend src directory to Python path
backend_dir = Path(__file__).parent.parent
src_dir = backend_dir / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(backend_dir))

def generate_openapi_spec():
    """Generate OpenAPI specification from FastAPI app."""
    try:
        # Set environment variables for proper configuration
        os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
        os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
        
        # Import from the src directory
        import main
        app = main.app
        
        # Generate OpenAPI schema
        openapi_schema = app.openapi()
        
        # Save as JSON
        api_dir = Path(__file__).parent / "api"
        json_path = api_dir / "openapi.json"
        
        with open(json_path, 'w') as f:
            json.dump(openapi_schema, f, indent=2)
        
        print(f"✓ OpenAPI specification saved to: {json_path}")
        
        # Also save a pretty-printed version for inspection
        pretty_path = api_dir / "openapi_pretty.json"
        with open(pretty_path, 'w') as f:
            json.dump(openapi_schema, f, indent=2, sort_keys=True)
        
        print(f"✓ Pretty OpenAPI specification saved to: {pretty_path}")
        
        return openapi_schema
        
    except ImportError as e:
        print(f"✗ Error importing FastAPI app: {e}")
        print("Make sure you have installed all dependencies and the app is properly configured.")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error generating OpenAPI spec: {e}")
        sys.exit(1)

def print_spec_summary(spec):
    """Print a summary of the generated specification."""
    info = spec.get("info", {})
    paths = spec.get("paths", {})
    components = spec.get("components", {})
    schemas = components.get("schemas", {})
    
    print(f"\n📋 OpenAPI Specification Summary:")
    print(f"   Title: {info.get('title', 'N/A')}")
    print(f"   Version: {info.get('version', 'N/A')}")
    print(f"   Description: {info.get('description', 'N/A')}")
    print(f"   Paths: {len(paths)} endpoints")
    print(f"   Schemas: {len(schemas)} data models")
    
    # List all endpoints
    print(f"\n🔗 Endpoints:")
    for path, methods in paths.items():
        for method, details in methods.items():
            if method.upper() in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
                summary = details.get('summary', 'No summary')
                print(f"   {method.upper():6} {path:50} - {summary}")

if __name__ == "__main__":
    print("🚀 Generating OpenAPI specification for ML Evaluation Platform...")
    spec = generate_openapi_spec()
    print_spec_summary(spec)
    print("\n✅ OpenAPI specification generation completed!")