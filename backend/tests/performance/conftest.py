"""
Performance test configuration and fixtures.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def performance_temp_dir():
    """Create temporary directory for performance test files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ml_eval_perf_"))
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session") 
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def performance_test_marker(request):
    """Automatically mark performance tests."""
    if "performance" in str(request.fspath):
        request.node.add_marker(pytest.mark.performance)


# Performance test markers
def pytest_configure(config):
    """Configure performance test markers."""
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "memory: Memory usage and leak tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: Benchmarking tests"
    )
    config.addinivalue_line(
        "markers", "regression: Performance regression tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for performance tests."""
    for item in items:
        # Add performance marker to all tests in performance directory
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add specific markers based on test name
        if "memory" in item.name:
            item.add_marker(pytest.mark.memory)
        if "benchmark" in item.name:
            item.add_marker(pytest.mark.benchmark) 
        if "regression" in item.name:
            item.add_marker(pytest.mark.regression)