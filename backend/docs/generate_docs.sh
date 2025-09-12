#!/bin/bash

# ML Evaluation Platform - Documentation Generation Script
# 
# This script automates the generation of comprehensive API documentation
# from the OpenAPI specification in multiple formats.

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BACKEND_DIR")"

echo "🚀 ML Evaluation Platform - Documentation Generation"
echo "=================================================="
echo

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import yaml, json, pathlib" 2>/dev/null || {
    echo "⚠️  Installing required Python packages..."
    pip install pyyaml
}

echo "✅ Dependencies OK"
echo

# Generate documentation
echo "📚 Generating comprehensive API documentation..."
cd "$BACKEND_DIR"

python docs/doc_generator.py

echo
echo "✅ Documentation generation completed!"
echo

# Display generated files
echo "📋 Generated Documentation Files:"
echo "================================="

find docs -type f -name "*.html" -o -name "*.md" -o -name "*.json" -o -name "*.py" -o -name "*.sh" | sort | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "   $file ($size)"
done

echo
echo "🌐 View the documentation:"
echo "   HTML: file://$BACKEND_DIR/docs/api/index.html"
echo "   Markdown: $BACKEND_DIR/docs/api/README.md"
echo "   OpenAPI JSON: $BACKEND_DIR/docs/api/openapi.json"
echo
echo "📖 Developer guides available in: $BACKEND_DIR/docs/guides/"
echo "💻 Code examples available in: $BACKEND_DIR/docs/examples/"
echo

# Check if a web server is available to serve the HTML documentation
if command -v python &> /dev/null; then
    echo "🌍 To serve the documentation locally, run:"
    echo "   cd $BACKEND_DIR/docs && python -m http.server 8080"
    echo "   Then open: http://localhost:8080/api/"
elif command -v npx &> /dev/null; then
    echo "🌍 To serve the documentation locally, run:"
    echo "   cd $BACKEND_DIR/docs && npx serve -s ."
    echo "   Then open the displayed URL"
fi

echo
echo "✨ Documentation generation completed successfully!"