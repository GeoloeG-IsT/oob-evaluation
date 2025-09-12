#!/usr/bin/env python3
"""
Backend Setup Script for ML Evaluation Platform
Handles Python version compatibility and dependency installation
"""
import sys
import subprocess
import platform
from pathlib import Path

def get_python_version():
    """Get the current Python version as tuple (major, minor)"""
    return sys.version_info[:2]

def run_command(command, description=""):
    """Run a shell command with error handling"""
    print(f"🔄 {description}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if e.stdout:
            print(f"Standard output: {e.stdout}")
        return False

def install_system_dependencies():
    """Install system-level dependencies for PostgreSQL and other packages"""
    system = platform.system().lower()
    
    if system == "linux":
        # Check if we can install system packages
        distro_commands = [
            (["apt-get", "update"], "Updating package list"),
            (["apt-get", "install", "-y", "libpq-dev", "python3-dev", "build-essential"], 
             "Installing PostgreSQL development libraries")
        ]
        
        for cmd, desc in distro_commands:
            print(f"🔄 {desc}")
            # Try with sudo first, then without
            if not run_command(["sudo"] + cmd, desc):
                print("⚠️ Could not install system dependencies with sudo. Please install manually:")
                print("  Ubuntu/Debian: sudo apt-get install libpq-dev python3-dev build-essential")
                print("  CentOS/RHEL: sudo yum install postgresql-devel python3-devel gcc")
                print("  macOS: brew install postgresql")
                return False
    
    elif system == "darwin":  # macOS
        print("🔄 Checking for Homebrew and PostgreSQL")
        if not run_command(["brew", "list", "postgresql"], "Checking PostgreSQL installation"):
            print("Installing PostgreSQL via Homebrew...")
            if not run_command(["brew", "install", "postgresql"], "Installing PostgreSQL"):
                print("⚠️ Could not install PostgreSQL. Please install manually: brew install postgresql")
                return False
    
    else:
        print(f"⚠️ Unsupported system: {system}")
        print("Please install PostgreSQL development libraries manually")
    
    return True

def main():
    """Main setup function"""
    print("🚀 ML Evaluation Platform Backend Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = get_python_version()
    print(f"📍 Python version: {python_version[0]}.{python_version[1]}")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        sys.exit(1)
    
    print(f"📂 Working directory: {backend_dir}")
    
    # Choose requirements file based on Python version
    if python_version >= (3, 13):
        print("🐍 Using Python 3.13+ compatible requirements")
        requirements_file = "requirements.txt"
        print("⚠️ Note: Some ML packages may not be fully compatible with Python 3.13")
        print("💡 Consider using Python 3.12 for full ML framework support")
    else:
        print("🐍 Using Python 3.12 compatible requirements")
        requirements_file = "requirements-python312.txt"
    
    requirements_path = backend_dir / requirements_file
    if not requirements_path.exists():
        print(f"❌ Requirements file not found: {requirements_path}")
        sys.exit(1)
    
    # Install system dependencies
    print("\n📦 Installing system dependencies...")
    if not install_system_dependencies():
        print("⚠️ System dependencies installation failed, but continuing...")
    
    # Upgrade pip
    print("\n🔧 Upgrading pip...")
    if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      "Upgrading pip"):
        print("⚠️ Pip upgrade failed, but continuing...")
    
    # Install Python dependencies
    print(f"\n📦 Installing Python dependencies from {requirements_file}...")
    
    # Try installing in stages for better error handling
    core_packages = [
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0", 
        "pydantic[email]>=2.5.0",
        "python-dotenv>=1.0.0"
    ]
    
    print("🔄 Installing core packages...")
    for package in core_packages:
        if not run_command([sys.executable, "-m", "pip", "install", package],
                          f"Installing {package}"):
            print(f"❌ Failed to install {package}")
            sys.exit(1)
    
    # Install database packages
    print("🔄 Installing database packages...")
    db_packages = []
    
    if python_version >= (3, 13):
        db_packages = ["psycopg[binary,pool]>=3.1.8", "asyncpg>=0.29.0"]
    else:
        db_packages = ["psycopg2-binary>=2.9.9", "asyncpg>=0.29.0"]
    
    for package in db_packages:
        if not run_command([sys.executable, "-m", "pip", "install", package],
                          f"Installing {package}"):
            print(f"❌ Failed to install {package}")
            # For database packages, this is critical
            if "psycopg" in package:
                print("💡 Try installing PostgreSQL development libraries:")
                print("  Ubuntu/Debian: sudo apt-get install libpq-dev python3-dev")
                print("  CentOS/RHEL: sudo yum install postgresql-devel python3-devel")
                print("  macOS: brew install postgresql")
                sys.exit(1)
    
    # Install remaining packages from requirements file
    print(f"🔄 Installing remaining packages from {requirements_file}...")
    install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    
    if not run_command(install_cmd, f"Installing from {requirements_file}"):
        print("⚠️ Some packages failed to install. This is common with Python 3.13.")
        print("💡 You can:")
        print("  1. Use Python 3.12 for better compatibility")
        print("  2. Install failed packages manually when wheels become available")
        print("  3. Use the core functionality without ML frameworks for now")
    
    # Verify installation
    print("\n✅ Verifying core installation...")
    test_imports = [
        "fastapi",
        "uvicorn", 
        "pydantic",
        "sqlalchemy",
        "asyncpg"
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - not available")
    
    print("\n🎉 Backend setup completed!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env.development and configure your settings")
    print("2. Start the development server: uvicorn main:app --reload")
    print("3. Check the API docs at: http://localhost:8000/docs")
    
    if python_version >= (3, 13):
        print("\n⚠️ Python 3.13 Note:")
        print("Some ML packages may not work yet. Consider using Python 3.12")
        print("for full compatibility with PyTorch, Ultralytics, etc.")

if __name__ == "__main__":
    main()