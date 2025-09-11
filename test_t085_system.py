#!/usr/bin/env python3
"""
T085 System Test - Quick verification of validation system components

This script performs basic tests to ensure the T085 validation system
is properly configured and can execute basic operations.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing module imports...")
    
    try:
        # Test core modules
        import aiohttp
        import asyncpg
        import redis
        import requests
        import yaml
        from PIL import Image
        import numpy as np
        
        logger.info("✓ All core dependencies imported successfully")
        return True
    
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        logger.error("Install missing dependencies with: pip install -r t085_requirements.txt")
        return False

def test_validation_modules():
    """Test that validation system modules can be imported"""
    logger.info("Testing validation system modules...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from t085_comprehensive_validator import T085ComprehensiveValidator, ValidationResult, ValidationReport
        from t085_workflow_validations import WorkflowValidations
        from t085_error_validation import ErrorHandlingValidation, PerformanceValidation
        
        logger.info("✓ All validation modules imported successfully")
        return True
    
    except ImportError as e:
        logger.error(f"✗ Validation module import failed: {e}")
        return False

def test_configuration_files():
    """Test that required configuration files exist"""
    logger.info("Testing configuration files...")
    
    base_dir = Path(__file__).parent
    required_files = [
        'docker-compose.yml',
        'docker-compose.validation.yml',
        't085_requirements.txt',
        't085_comprehensive_validator.py',
        't085_workflow_validations.py',
        't085_error_validation.py',
        'run_t085_validation.sh'
    ]
    
    missing_files = []
    for filename in required_files:
        if not (base_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"✗ Missing configuration files: {missing_files}")
        return False
    
    logger.info("✓ All configuration files present")
    return True

def test_docker_compose_syntax():
    """Test Docker Compose file syntax"""
    logger.info("Testing Docker Compose configuration...")
    
    try:
        import yaml
        
        base_dir = Path(__file__).parent
        
        # Test main compose file
        with open(base_dir / 'docker-compose.yml', 'r') as f:
            main_config = yaml.safe_load(f)
        
        # Test validation compose file
        with open(base_dir / 'docker-compose.validation.yml', 'r') as f:
            validation_config = yaml.safe_load(f)
        
        # Basic validation
        assert 'services' in main_config, "Main compose file missing services section"
        assert 'services' in validation_config, "Validation compose file missing services section"
        
        # Check for required services
        required_services = ['backend', 'frontend', 'db', 'redis']
        main_services = main_config.get('services', {}).keys()
        
        missing_services = [svc for svc in required_services if svc not in main_services]
        if missing_services:
            logger.error(f"✗ Missing required services in compose file: {missing_services}")
            return False
        
        logger.info("✓ Docker Compose configuration valid")
        return True
    
    except Exception as e:
        logger.error(f"✗ Docker Compose configuration error: {e}")
        return False

async def test_validation_system_basic():
    """Test basic validation system initialization"""
    logger.info("Testing validation system initialization...")
    
    try:
        from t085_comprehensive_validator import T085ComprehensiveValidator
        
        # Create validator instance
        validator = T085ComprehensiveValidator()
        
        # Test setup (without network calls)
        await validator.setup()
        
        # Test test data generation
        test_images_dir = Path(__file__).parent / "test_images"
        if test_images_dir.exists():
            import shutil
            shutil.rmtree(test_images_dir)
        
        await validator.generate_test_images()
        
        # Verify test images were created
        if not test_images_dir.exists() or not list(test_images_dir.glob("*.jpg")):
            logger.error("✗ Test image generation failed")
            return False
        
        # Test cleanup
        await validator.cleanup()
        
        logger.info("✓ Validation system initialization successful")
        return True
    
    except Exception as e:
        logger.error(f"✗ Validation system test failed: {e}")
        return False

def test_script_executability():
    """Test that shell scripts are executable"""
    logger.info("Testing shell script executability...")
    
    base_dir = Path(__file__).parent
    script_path = base_dir / 'run_t085_validation.sh'
    
    if not script_path.exists():
        logger.error("✗ Validation runner script not found")
        return False
    
    import stat
    if not script_path.stat().st_mode & stat.S_IEXEC:
        logger.error("✗ Validation runner script not executable")
        logger.info("Fix with: chmod +x run_t085_validation.sh")
        return False
    
    logger.info("✓ Shell scripts are executable")
    return True

async def run_all_tests():
    """Run all system tests"""
    logger.info("=" * 60)
    logger.info("T085 System Test - Validating validation system setup")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Validation Modules", test_validation_modules),
        ("Configuration Files", test_configuration_files),
        ("Docker Compose Syntax", test_docker_compose_syntax),
        ("Script Executability", test_script_executability),
        ("Validation System Basic", test_validation_system_basic),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
            else:
                failed += 1
        
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED - T085 system is ready!")
        logger.info("You can now run: ./run_t085_validation.sh")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED - Fix issues before running validation")
        logger.info("Check the error messages above and resolve the issues")
        return False

def main():
    """Main entry point"""
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()