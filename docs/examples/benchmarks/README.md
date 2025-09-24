# Performance Benchmarking & Load Testing

This directory contains benchmarking scripts and load testing tools for validating the performance optimizations described in the main documentation.

## Running Benchmarks

```bash
# Install benchmarking dependencies
pip install pytest-benchmark locust

# Run algorithm benchmarks
python benchmark_algorithms.py

# Run load tests
locust -f load_test_matching.py --host=http://localhost:8000
```

## Benchmark Results

Target performance metrics are defined in `PERFORMANCE_OPTIMIZATION.md`. Use these scripts to validate optimizations meet the SLA requirements:

- **Vector Search**: <50ms for 100k vectors
- **Matching Pipeline**: <5s end-to-end
- **Embedding Generation**: 1000 chunks/min throughput
- **Database Operations**: 10k rows/sec bulk insert

## Files

- `benchmark_algorithms.py` - Micro-benchmarks for core algorithms
- `load_test_matching.py` - End-to-end matching service load test  
- `benchmark_results.md` - Historical benchmark results
- `performance_regression_test.py` - Automated performance regression detection