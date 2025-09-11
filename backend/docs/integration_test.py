#!/usr/bin/env python3
"""
Integration Test for ML Evaluation Platform Documentation System

This script validates that all documentation components are working correctly
and the generated documentation is complete and accurate.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

class DocumentationValidator:
    """Validates the generated documentation system."""
    
    def __init__(self):
        self.docs_dir = Path(__file__).parent
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 ML Evaluation Platform - Documentation Validation")
        print("=" * 55)
        
        checks = [
            ("File Structure", self.check_file_structure),
            ("OpenAPI JSON", self.check_openapi_json),
            ("HTML Documentation", self.check_html_documentation),
            ("Markdown Documentation", self.check_markdown_documentation),
            ("Developer Guides", self.check_developer_guides),
            ("Code Examples", self.check_code_examples),
            ("Build Scripts", self.check_build_scripts),
        ]
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...")
            try:
                check_func()
                print(f"   ✅ {check_name} validation passed")
            except Exception as e:
                self.errors.append(f"{check_name}: {str(e)}")
                print(f"   ❌ {check_name} validation failed: {e}")
        
        # Print summary
        self.print_summary()
        
        return len(self.errors) == 0
    
    def check_file_structure(self):
        """Validate the documentation directory structure."""
        required_dirs = ['api', 'guides', 'examples', 'templates', 'assets']
        required_files = [
            'README.md',
            'Makefile', 
            'doc_generator.py',
            'generate_docs.sh',
            'index.html',
        ]
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = self.docs_dir / dir_name
            if not dir_path.exists():
                raise Exception(f"Required directory missing: {dir_name}")
            if not dir_path.is_dir():
                raise Exception(f"Path exists but is not a directory: {dir_name}")
        
        # Check files
        for file_name in required_files:
            file_path = self.docs_dir / file_name
            if not file_path.exists():
                raise Exception(f"Required file missing: {file_name}")
            if not file_path.is_file():
                raise Exception(f"Path exists but is not a file: {file_name}")
    
    def check_openapi_json(self):
        """Validate the OpenAPI JSON specification."""
        openapi_path = self.docs_dir / "api" / "openapi.json"
        
        if not openapi_path.exists():
            raise Exception("OpenAPI JSON file not found")
        
        # Load and validate JSON structure
        try:
            with open(openapi_path, 'r') as f:
                spec = json.load(f)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {e}")
        
        # Check required OpenAPI fields
        required_fields = ['openapi', 'info', 'paths']
        for field in required_fields:
            if field not in spec:
                raise Exception(f"Missing required OpenAPI field: {field}")
        
        # Check version
        if not spec.get('openapi', '').startswith('3.0'):
            self.warnings.append("OpenAPI version should be 3.0.x")
        
        # Count endpoints
        paths = spec.get('paths', {})
        endpoint_count = sum(len([m for m in methods.keys() if m.upper() in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']]) 
                           for methods in paths.values())
        
        self.info.append(f"Found {endpoint_count} API endpoints")
        
        if endpoint_count < 20:
            self.warnings.append(f"Expected 20+ endpoints, found {endpoint_count}")
    
    def check_html_documentation(self):
        """Validate the HTML documentation."""
        html_files = [
            'api/index.html',
            'index.html'
        ]
        
        for html_file in html_files:
            html_path = self.docs_dir / html_file
            
            if not html_path.exists():
                raise Exception(f"HTML file not found: {html_file}")
            
            with open(html_path, 'r') as f:
                content = f.read()
            
            # Basic HTML structure validation
            if not content.startswith('<!DOCTYPE html>'):
                self.warnings.append(f"{html_file}: Missing DOCTYPE declaration")
            
            if '<html' not in content or '</html>' not in content:
                raise Exception(f"{html_file}: Invalid HTML structure")
            
            # Check for key content
            required_elements = ['<title>', '<head>', '<body>']
            for element in required_elements:
                if element not in content:
                    self.warnings.append(f"{html_file}: Missing {element}")
    
    def check_markdown_documentation(self):
        """Validate the Markdown documentation."""
        md_files = [
            'api/README.md',
            'README.md'
        ]
        
        for md_file in md_files:
            md_path = self.docs_dir / md_file
            
            if not md_path.exists():
                raise Exception(f"Markdown file not found: {md_file}")
            
            with open(md_path, 'r') as f:
                content = f.read()
            
            if len(content) < 1000:
                self.warnings.append(f"{md_file}: File seems too short ({len(content)} chars)")
            
            # Check for headers
            if not content.startswith('# '):
                self.warnings.append(f"{md_file}: Should start with level 1 header")
    
    def check_developer_guides(self):
        """Validate the developer guides."""
        guide_files = [
            'guides/quick-start.md',
            'guides/authentication.md', 
            'guides/error-handling.md',
            'guides/examples.md'
        ]
        
        for guide_file in guide_files:
            guide_path = self.docs_dir / guide_file
            
            if not guide_path.exists():
                raise Exception(f"Guide file not found: {guide_file}")
            
            with open(guide_path, 'r') as f:
                content = f.read()
            
            if len(content) < 500:
                self.warnings.append(f"{guide_file}: Guide seems too short")
            
            # Check for code examples
            if '```' not in content and guide_file != 'guides/authentication.md':
                self.warnings.append(f"{guide_file}: No code examples found")
    
    def check_code_examples(self):
        """Validate the code examples."""
        example_files = [
            'examples/curl_examples.sh',
            'examples/python_examples.py',
            'examples/request_response_examples.json'
        ]
        
        for example_file in example_files:
            example_path = self.docs_dir / example_file
            
            if not example_path.exists():
                raise Exception(f"Example file not found: {example_file}")
            
            with open(example_path, 'r') as f:
                content = f.read()
            
            if len(content) < 1000:
                self.warnings.append(f"{example_file}: File seems too short")
            
            # Specific validations
            if example_file.endswith('.json'):
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    raise Exception(f"{example_file}: Invalid JSON format")
            
            elif example_file.endswith('.sh'):
                if 'curl' not in content:
                    self.warnings.append(f"{example_file}: No curl commands found")
            
            elif example_file.endswith('.py'):
                if 'class' not in content or 'def' not in content:
                    self.warnings.append(f"{example_file}: No Python classes/functions found")
    
    def check_build_scripts(self):
        """Validate the build and automation scripts."""
        script_files = [
            'generate_docs.sh',
            'doc_generator.py',
            'Makefile'
        ]
        
        for script_file in script_files:
            script_path = self.docs_dir / script_file
            
            if not script_path.exists():
                raise Exception(f"Script file not found: {script_file}")
            
            # Check if shell scripts are executable
            if script_file.endswith('.sh'):
                if not os.access(script_path, os.X_OK):
                    self.warnings.append(f"{script_file}: Script is not executable")
            
            # Basic content validation
            with open(script_path, 'r') as f:
                content = f.read()
            
            if len(content) < 100:
                self.warnings.append(f"{script_file}: Script seems too short")
    
    def print_summary(self):
        """Print validation summary."""
        print(f"\n" + "=" * 55)
        print("📊 VALIDATION SUMMARY")
        print("=" * 55)
        
        if self.info:
            print(f"\n📈 Information:")
            for info in self.info:
                print(f"   ℹ️  {info}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   ⚠️  {warning}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   ❌ {error}")
        else:
            print(f"\n✅ No errors found!")
        
        print(f"\n🎯 Overall Status: {'PASS' if not self.errors else 'FAIL'}")
        
        if not self.errors and not self.warnings:
            print("🌟 Perfect! Documentation system is fully validated.")
        elif not self.errors:
            print("✅ Good! Documentation system is working with minor warnings.")
        else:
            print("❌ Issues found that need to be addressed.")


def main():
    """Run the documentation validation."""
    validator = DocumentationValidator()
    
    success = validator.validate_all()
    
    if success:
        print(f"\n🎉 Documentation validation completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Documentation validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()