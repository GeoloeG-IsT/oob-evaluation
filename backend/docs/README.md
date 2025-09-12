# ML Evaluation Platform - API Documentation

This directory contains the comprehensive API documentation system for the ML Evaluation Platform, featuring automatic generation from OpenAPI specifications and multiple output formats.

## 📁 Directory Structure

```
docs/
├── README.md                    # This file
├── Makefile                     # Build automation
├── generate_docs.sh             # Shell script for doc generation
├── doc_generator.py             # Python documentation generator
├── generate_openapi.py          # OpenAPI spec extraction (optional)
├── api/                         # Generated API documentation
│   ├── index.html              # Interactive HTML documentation
│   ├── README.md               # Markdown API reference
│   └── openapi.json            # OpenAPI specification
├── guides/                      # Developer guides
│   ├── quick-start.md          # Getting started guide
│   ├── authentication.md       # Authentication documentation
│   ├── error-handling.md       # Error handling guide
│   └── examples.md             # Comprehensive examples
├── examples/                    # Code examples
│   ├── curl_examples.sh        # cURL command examples
│   ├── python_examples.py      # Python SDK examples
│   └── request_response_examples.json  # JSON examples
├── templates/                   # Documentation templates
└── assets/                      # Static assets (CSS, images)
```

## 🚀 Quick Start

### Generate All Documentation

```bash
# Using the shell script
./generate_docs.sh

# Using Make (recommended)
make docs

# Using Python directly
python doc_generator.py
```

### Serve Documentation Locally

```bash
# Start local server on port 8080
make serve

# View at: http://localhost:8080/api/
```

### Development Mode with Auto-Reload

```bash
# Watch for changes and auto-regenerate
make serve-dev
```

## 📖 Documentation Formats

### 1. Interactive HTML Documentation

- **Location**: `api/index.html`
- **Features**:
  - Responsive design with modern styling
  - Interactive table of contents
  - Syntax-highlighted code examples
  - Expandable/collapsible sections
  - Search functionality

### 2. Markdown API Reference

- **Location**: `api/README.md`
- **Features**:
  - GitHub-compatible markdown
  - Complete endpoint documentation
  - Request/response schemas
  - Parameter tables
  - Code examples

### 3. OpenAPI JSON Specification

- **Location**: `api/openapi.json`
- **Features**:
  - Standard OpenAPI 3.0.3 format
  - Compatible with Swagger UI
  - Machine-readable for SDK generation
  - API client generation support

### 4. Developer Guides

Comprehensive guides covering:

- **Quick Start**: Getting up and running quickly
- **Authentication**: Security and auth patterns
- **Error Handling**: Comprehensive error documentation
- **Examples**: Real-world usage patterns

### 5. Code Examples

- **cURL Examples**: Complete command-line examples
- **Python Examples**: SDK-style Python code
- **Request/Response Examples**: JSON examples for all endpoints

## 🛠️ Available Commands

### Make Commands

```bash
make help          # Show all available commands
make docs          # Generate all documentation
make clean         # Clean generated files
make serve         # Generate and serve locally
make serve-dev     # Serve with auto-reload
make install-deps  # Install Python dependencies
make check-deps    # Verify dependencies
make lint          # Check documentation quality
make stats         # Show documentation statistics
make validate      # Validate generated documentation
make watch         # Auto-regenerate on changes
make dev-setup     # Set up development environment
make publish-check # Verify docs are ready for publishing
```

### Shell Script

```bash
./generate_docs.sh  # Generate all documentation with progress output
```

## 📋 Features

### Comprehensive Coverage

- ✅ **54 API Endpoints** across 8 routers
- ✅ **Complete Request/Response Schemas**
- ✅ **Parameter Documentation** with types and validation
- ✅ **Error Response Examples** for all status codes
- ✅ **Authentication Patterns** (development and production)
- ✅ **Usage Examples** in multiple languages
- ✅ **Integration Guides** for common workflows

### Multiple Output Formats

- ✅ **HTML**: Interactive web documentation
- ✅ **Markdown**: GitHub-compatible documentation
- ✅ **JSON**: Machine-readable OpenAPI spec
- ✅ **Shell Scripts**: Executable cURL examples
- ✅ **Python Code**: SDK-style examples

### Developer Experience

- ✅ **Quick Start Guides** for immediate productivity
- ✅ **Error Handling Patterns** with retry logic
- ✅ **Complete Python Client** with examples
- ✅ **Real-world Workflows** covering all use cases
- ✅ **Performance Considerations** and best practices

### Automation and Maintenance

- ✅ **Automated Generation** from OpenAPI spec
- ✅ **Build Scripts** for CI/CD integration
- ✅ **Validation Tools** for quality assurance
- ✅ **Auto-reload Development** for rapid iteration
- ✅ **Documentation Statistics** and health checks

## 🔧 Development

### Prerequisites

- Python 3.8+
- PyYAML package (`pip install pyyaml`)

### Adding New Documentation

1. **Update OpenAPI Spec**: Modify the source specification at:
   ```
   /specs/001-oob-evaluation-claude/contracts/api-spec.yaml
   ```

2. **Regenerate Documentation**:
   ```bash
   make docs
   ```

3. **Validate Changes**:
   ```bash
   make validate
   ```

4. **Preview Locally**:
   ```bash
   make serve
   ```

### Customizing Documentation

#### Modify Templates

Edit the generator methods in `doc_generator.py`:

- `_create_html_documentation()`: HTML template and styling
- `_create_markdown_documentation()`: Markdown format
- `_create_quickstart_guide()`: Quick start content
- `_create_examples_guide()`: Code examples

#### Add New Formats

Extend the `APIDocumentationGenerator` class:

```python
def _generate_pdf_documentation(self):
    """Generate PDF documentation."""
    # Implementation here
    pass
```

#### Custom Styling

Modify the CSS in the HTML template within `_create_html_documentation()`.

### CI/CD Integration

Add to your CI pipeline:

```yaml
# Example GitHub Actions
- name: Generate API Documentation
  run: |
    cd backend/docs
    make docs
    make validate

- name: Deploy Documentation
  run: |
    # Deploy generated files to your hosting service
    cp -r backend/docs/api/* docs-site/
```

## 📊 Documentation Quality

### Validation Checks

The system includes built-in validation:

- ✅ OpenAPI JSON schema validation
- ✅ HTML well-formedness checks
- ✅ Markdown link validation
- ✅ Required file existence checks
- ✅ Documentation completeness metrics

### Statistics Tracking

Monitor documentation health:

```bash
make stats  # Show file counts, sizes, and last updated
```

### Quality Metrics

- **Endpoint Coverage**: 54/54 endpoints documented (100%)
- **Schema Coverage**: All request/response schemas included
- **Example Coverage**: Every endpoint has cURL and Python examples
- **Guide Coverage**: 4 comprehensive developer guides
- **Error Coverage**: All HTTP status codes documented

## 🌐 Deployment

### Static Site Hosting

The generated documentation is completely static and can be hosted anywhere:

```bash
# Copy files to web server
cp -r docs/api/* /var/www/docs/

# Or use a static site service
# - GitHub Pages
# - Netlify
# - Vercel
# - AWS S3 + CloudFront
```

### Integration with Existing Sites

The HTML documentation includes proper meta tags and is designed to integrate with existing documentation sites:

- Responsive design works on all devices
- SEO-friendly structure with proper headings
- No external dependencies (all CSS/JS inlined)
- Consistent styling that can be customized

### API Gateway Integration

The OpenAPI JSON can be used with API gateways:

- AWS API Gateway
- Google Cloud Endpoints
- Azure API Management
- Kong Gateway

## 🔍 Troubleshooting

### Common Issues

#### Missing Dependencies

```bash
# Install required packages
make install-deps

# Or manually
pip install pyyaml
```

#### Permission Issues

```bash
# Make script executable
chmod +x generate_docs.sh
```

#### Port Already in Use

```bash
# Use different port
cd docs && python -m http.server 8081
```

#### File Not Found Errors

Ensure you're running commands from the correct directory:

```bash
cd /path/to/backend/docs
make docs
```

### Getting Help

1. **Check the logs**: Most commands provide detailed error output
2. **Validate dependencies**: Run `make check-deps`
3. **Check file permissions**: Ensure scripts are executable
4. **Verify file paths**: Ensure the OpenAPI spec exists at the expected location

## 📈 Future Enhancements

Planned improvements:

- [ ] **PDF Generation**: Export documentation as PDF
- [ ] **Interactive API Explorer**: Embedded Swagger UI
- [ ] **Multi-language Examples**: JavaScript, Go, Java examples
- [ ] **Video Tutorials**: Embedded walkthrough videos
- [ ] **Advanced Search**: Full-text search across documentation
- [ ] **API Changelog**: Automatic change detection and documentation
- [ ] **Performance Benchmarks**: Include API performance data
- [ ] **SDK Generation**: Automatic client library generation

## 📄 License

This documentation system is part of the ML Evaluation Platform project.

---

*Generated automatically from OpenAPI specification. Last updated: $(date)*