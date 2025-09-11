# Performance Tests for ML Evaluation Platform

This directory contains comprehensive performance tests ensuring real-time inference requirements (T082) are met across all supported model variants.

## Overview

The performance test suite validates:

- **Real-time Inference Performance**: < 2 seconds target for single image inference
- **Batch Processing Performance**: Efficient concurrent operations with memory management
- **Model Loading Performance**: Optimized model loading across all variants
- **Concurrent Request Handling**: Support for multiple simultaneous requests
- **Memory Usage Monitoring**: Prevention of memory leaks and resource cleanup
- **Performance Regression Detection**: Automatic detection of performance degradation

## Test Coverage

### Model Variants Tested

- **YOLO11/12**: nano, small, medium, large, extra-large
- **RT-DETR**: R18, R34, R50, R101, RF-DETR (nano, small, medium)  
- **SAM2**: Tiny, Small, Base+, Large

### Image Test Scenarios

- **Small images**: 320x240 (simple, medium, complex content)
- **Standard resolution**: 640x480 (simple, medium, complex content)
- **HD resolution**: 1280x720 (simple, medium, complex content)
- **UHD resolution**: 1920x1080 (simple, medium content)
- **Edge cases**: tiny (64x64), wide (1600x400), tall (400x1600)

### Performance Benchmarks

| Metric | Target | Description |
|--------|--------|-------------|
| Real-time Inference | < 2000ms | Single image inference time |
| Model Loading | Varies by model | See benchmark configuration |
| Memory per Inference | < 1024MB | Memory usage per single inference |
| Batch Memory | < 4096MB | Memory usage for batch processing |
| Concurrent Response | < 5000ms | Response time under concurrent load |

## Running Performance Tests

### Run All Performance Tests
```bash
# Run all performance tests
pytest tests/performance/ -m performance -v

# Run with performance report output
PRINT_PERFORMANCE_REPORT=true pytest tests/performance/ -m performance -v
```

### Run Specific Test Categories
```bash
# Run only memory tests
pytest tests/performance/ -m memory -v

# Run only benchmark tests
pytest tests/performance/ -m benchmark -v

# Run only regression tests
pytest tests/performance/ -m regression -v
```

### Run Individual Test Methods
```bash
# Test real-time inference performance
pytest tests/performance/test_inference_speed.py::TestInferencePerformance::test_single_inference_real_time_performance -v

# Test memory leak detection
pytest tests/performance/test_inference_speed.py::TestInferencePerformance::test_memory_leak_detection -v

# Test concurrent handling
pytest tests/performance/test_inference_speed.py::TestInferencePerformance::test_concurrent_inference_handling -v
```

### Standalone Benchmark
```bash
# Run comprehensive benchmark outside of pytest
cd backend
python tests/performance/test_inference_speed.py
```

## Test Structure

### Main Test Class: `TestInferencePerformance`

1. **`test_yolo_model_loading_performance`**: Tests model loading speed across YOLO variants
2. **`test_single_inference_real_time_performance`**: Validates < 2s inference requirement
3. **`test_batch_inference_performance`**: Tests batch processing efficiency
4. **`test_concurrent_inference_handling`**: Validates concurrent request support
5. **`test_different_image_sizes_performance`**: Tests across various image dimensions
6. **`test_memory_leak_detection`**: Detects memory leaks in repeated operations
7. **`test_resource_cleanup`**: Validates proper resource cleanup
8. **`test_performance_regression_detection`**: Detects performance regressions
9. **`test_generate_performance_report`**: Generates comprehensive performance report

### Performance Test Suite: `PerformanceTestSuite`

Core testing framework providing:

- Model setup for all supported variants
- Test image generation with different complexities
- Memory monitoring during operations
- Performance benchmarking and comparison
- Comprehensive reporting and analysis

### Benchmark Configuration: `PerformanceBenchmark`

Defines performance targets and thresholds:

- Real-time inference targets
- Model loading time limits by variant
- Memory usage limits  
- Throughput requirements
- Concurrent processing standards

## Performance Monitoring

### Memory Monitoring
The `MemoryMonitor` class tracks:
- Baseline memory usage
- Peak memory during operations
- Memory deltas for leak detection

### Performance Metrics
Each test collects:
- Inference time (preprocessing, inference, postprocessing)
- Total execution time
- Memory usage
- Throughput (images per second)
- Success/failure rates

## Regression Detection

The test suite automatically detects regressions by:
- Comparing against configured performance targets
- Identifying tests exceeding baseline metrics by >20%
- Flagging memory usage increases
- Tracking throughput degradation

## Report Generation

Performance reports include:
- Summary statistics and compliance rates
- Results grouped by test type and model
- Performance regression analysis
- Memory leak analysis
- Optimization recommendations

## Integration with CI/CD

### Performance Gates
Tests enforce performance requirements as hard failures:
- Inference time must be < 2 seconds
- Memory usage within configured limits
- Success rates above 80% threshold
- Real-time compliance above 90%

### Continuous Monitoring
Regular performance testing enables:
- Performance trend tracking
- Early regression detection
- Resource usage optimization
- Model performance comparison

## Troubleshooting

### Common Issues

**High Memory Usage:**
- Check for model cleanup after tests
- Verify garbage collection is working
- Monitor for unclosed resources

**Slow Inference Times:**
- Verify model loading optimization
- Check system resource availability
- Review model variant selection

**Test Failures:**
- Check model dependencies are available
- Verify temporary directory permissions
- Review memory and CPU limits

### Debug Mode
```bash
# Run with verbose debugging
pytest tests/performance/ -v -s --tb=long

# Run with memory profiling
python -m memory_profiler tests/performance/test_inference_speed.py
```

## Contributing

When adding new performance tests:

1. Follow the existing test patterns
2. Use appropriate performance markers
3. Include memory monitoring
4. Document performance expectations
5. Update benchmark configurations
6. Verify CI/CD integration

## Performance Optimization

Based on test results, consider:

- Model quantization for faster inference
- Batch processing optimization
- Memory pool management
- Concurrent processing tuning
- Resource cleanup automation