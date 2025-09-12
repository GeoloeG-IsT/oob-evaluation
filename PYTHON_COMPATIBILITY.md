# Python Version Compatibility Guide

## Overview

The ML Evaluation Platform has been tested with Python 3.11-3.13, but there are some compatibility considerations, especially with Python 3.13.

## Quick Solutions

### Option 1: Automated Setup (Recommended)
```bash
python setup_backend.py
```

This script will:
- Detect your Python version
- Install appropriate dependencies
- Handle system requirements
- Provide fallbacks for incompatible packages

### Option 2: Manual Installation

#### For Python 3.13 (Latest)
```bash
cd backend
pip install -r requirements.txt
```

**Known Issues with Python 3.13:**
- `psycopg2-binary` may fail to install (we use `psycopg` v3 instead)
- Some ML packages (PyTorch, Ultralytics) may not have wheels yet
- Package compilation may require system dependencies

#### For Python 3.12 (Recommended for ML)
```bash
cd backend  
pip install -r requirements-python312.txt
```

This version has full compatibility with all ML frameworks.

## System Dependencies

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libpq-dev python3-dev build-essential
```

### CentOS/RHEL/Fedora
```bash
sudo yum install postgresql-devel python3-devel gcc
# Or with dnf:
sudo dnf install postgresql-devel python3-devel gcc
```

### macOS
```bash
brew install postgresql
```

### Windows
```bash
# Install PostgreSQL from: https://www.postgresql.org/download/windows/
# Or use conda:
conda install postgresql
```

## Package-Specific Solutions

### Database Connection (psycopg2 error)

**Error:** `pg_config executable not found`

**Solution 1 (Python 3.13+):** Use psycopg v3
```bash
pip install "psycopg[binary,pool]>=3.1.8"
```

**Solution 2 (Python ≤3.12):** Install system dependencies first
```bash
# Ubuntu/Debian
sudo apt-get install libpq-dev python3-dev

# Then install psycopg2
pip install psycopg2-binary
```

**Solution 3:** Use conda
```bash
conda install psycopg2
```

### ML Frameworks (PyTorch, etc.)

**For Python 3.13:** These may not be available yet
```bash
# Check if available:
pip install torch torchvision

# If failed, use CPU-only version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or wait for official Python 3.13 support
```

**For Python 3.12 and earlier:** Full support
```bash
pip install torch torchvision ultralytics transformers
```

## Environment Setup Examples

### Development with Python 3.13
```bash
# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate     # Windows

# Install core dependencies only
pip install fastapi uvicorn sqlalchemy asyncpg "psycopg[binary]"

# Skip ML packages for now
export SKIP_ML_PACKAGES=true
```

### Production with Python 3.12
```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r backend/requirements-python312.txt
```

### Using Docker (Recommended)
```bash
# Use Python 3.12 in Docker for consistent environment
docker-compose up -d

# All dependencies are pre-installed in containers
```

## Troubleshooting

### Common Error: "Microsoft Visual C++ 14.0 is required" (Windows)
```bash
# Install Visual Studio Build Tools
# Or use conda:
conda install psycopg2 pytorch torchvision -c pytorch
```

### Common Error: "Failed building wheel for psycopg2"
```bash
# Option 1: Use binary version
pip install psycopg2-binary

# Option 2: Use psycopg v3 (Python 3.13+)
pip install "psycopg[binary]"

# Option 3: Install system dependencies (see above)
```

### Common Error: "No module named 'torch'"
```bash
# For Python 3.13, try:
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Or downgrade to Python 3.12:
pyenv install 3.12.0
pyenv local 3.12.0
```

## Recommended Approach

1. **For Development:** Use Python 3.12 for full ML compatibility
2. **For Production:** Use Docker containers with Python 3.12
3. **For Python 3.13:** Use core functionality only, add ML later

## Testing Your Installation

```bash
python -c "
import fastapi, uvicorn, sqlalchemy, asyncpg
print('✅ Core packages working')

try:
    import psycopg
    print('✅ PostgreSQL (psycopg v3) working')
except:
    import psycopg2
    print('✅ PostgreSQL (psycopg2) working')

try:
    import torch, torchvision
    print('✅ ML frameworks working')
except:
    print('⚠️ ML frameworks not available')
"
```

## Getting Help

If you continue to have issues:

1. Check your Python version: `python --version`
2. Check your pip version: `pip --version`
3. Try the automated setup: `python setup_backend.py`
4. Use Docker for consistent environment: `docker-compose up -d`
5. Consider using Python 3.12 for full compatibility

## Version Support Matrix

| Python Version | Core API | Database | ML Frameworks | Status |
|---------------|----------|----------|---------------|---------|
| 3.11 | ✅ | ✅ | ✅ | Fully Supported |
| 3.12 | ✅ | ✅ | ✅ | Recommended |
| 3.13 | ✅ | ✅ | ⚠️ | Partial (Core Only) |

The ML Evaluation Platform core functionality works with all supported Python versions. ML frameworks may require Python ≤3.12 until official support is released.