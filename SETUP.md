# RAG-CSD Setup Guide

This guide provides step-by-step instructions to set up the RAG-CSD (Retrieval-Augmented Generation with Computational Storage Devices) project environment and run benchmarks with visualization dashboards.

## Quick Start

### 1. Create Conda Environment

```bash
# Create the conda environment from the provided YAML file
conda env create -f environment.yml

# Activate the environment
conda activate rag-csd
```

### 2. Install the Project

```bash
# Install the project in development mode
pip install -e .
```

### 3. Create Vector Database

```bash
# Process sample data and create vector database
python scripts/create_vector_db.py --input data/raw --output data/vectors --config config/default.yaml

# For larger dataset (optional)
python scripts/create_vector_db.py --input data/raw_large --output data/vectors_large --config config/default.yaml
```

## Running Benchmarks and Dashboards

### Standard Benchmark

Run comprehensive performance comparison between standard and CSD-simulated execution:

```bash
# Basic benchmark
python examples/benchmark.py --vector-db data/vectors --queries data/test_queries.json

# Advanced benchmark with custom parameters
python examples/benchmark.py \
    --vector-db data/vectors \
    --queries data/test_queries.json \
    --top-k 10 \
    --runs 5 \
    --output results/benchmark_results.json
```

### Interactive Dashboard Demo

Launch the performance monitoring dashboard with real-time visualization:

```bash
# Full dashboard demo (simulation + live monitoring)
python examples/dashboard_demo.py --vector-db data/vectors --mode both --duration 120

# Simulation only
python examples/dashboard_demo.py --vector-db data/vectors --mode simulate --duration 180 --query-rate 2.0

# Live monitoring only
python examples/dashboard_demo.py --vector-db data/vectors --mode live --duration 60
```

Dashboard outputs will be saved to `dashboard_output/` including:
- Real-time latency plots
- System comparison charts
- Performance reports (JSON)
- Raw metrics data

### Comprehensive Benchmarking

For extensive performance analysis across multiple configurations:

```bash
# Run comprehensive benchmark suite
python examples/comprehensive_benchmark.py \
    --vector-db data/vectors \
    --output-dir results/comprehensive/ \
    --num-queries 100 \
    --batch-sizes 1,5,10,20 \
    --top-k-values 5,10,20
```

## Advanced Usage

### Async Processing Demo

Test asynchronous query processing capabilities:

```bash
python examples/async_demo.py --vector-db data/vectors --concurrent-queries 10
```

### Batch Processing

Process multiple queries efficiently:

```bash
python scripts/batch_query.py \
    --vector-db data/vectors \
    --queries data/test_queries.json \
    --batch-size 10 \
    --output results/batch_results.json
```

### Custom Data Processing

Process your own documents:

```bash
# Process PDF documents
python scripts/process_pdfs.py --input corpus/ --output data/custom_vectors/

# Evaluate retrieval performance
python scripts/evaluate_retrieval.py --vector-db data/vectors --queries data/test_queries.json
```

## Environment Options

### CPU-Only Setup

For CPU-only environments, modify `environment.yml`:
- Remove `pytorch-cuda=11.8`
- Use `faiss-cpu` instead of `faiss-gpu`

### GPU Setup

For GPU acceleration:
- Keep `pytorch-cuda=11.8` (or appropriate CUDA version)
- Replace `faiss-cpu` with `faiss-gpu`
- Ensure CUDA drivers are properly installed

### Alternative Installation

If conda is not available, use pip with requirements files:

```bash
# Create virtual environment
python -m venv rag-csd-env
source rag-csd-env/bin/activate  # On Windows: rag-csd-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Configuration

### Default Configuration

The project uses `config/default.yaml` for default settings. Key parameters:

```yaml
# Vector database settings
vector_db:
  dimension: 384
  index_type: "IVFFlat"
  nlist: 100

# CSD simulation settings
csd:
  enabled: false
  latency_reduction: 0.3
  bandwidth_multiplier: 2.0

# Caching settings
cache:
  enabled: true
  max_size: 1000
```

### Custom Configuration

Create custom configuration files and specify them:

```bash
python examples/benchmark.py --config config/custom.yaml --vector-db data/vectors
```

## Troubleshooting

### Common Issues

1. **CUDA errors**: Ensure CUDA version matches PyTorch installation
2. **Memory issues**: Reduce batch sizes or use CPU-only mode
3. **Import errors**: Verify all dependencies are installed and environment is activated

### Performance Tips

1. Use GPU acceleration for better performance
2. Optimize vector database size based on available memory
3. Adjust batch sizes for optimal throughput
4. Enable caching for repeated queries

## Outputs and Results

- **Benchmark results**: JSON files with detailed performance metrics
- **Dashboard outputs**: PNG plots and interactive HTML reports
- **Logs**: Detailed execution logs for debugging
- **Raw metrics**: CSV/JSON files for further analysis

## Next Steps

1. Explore the generated visualizations in the output directories
2. Experiment with different configurations
3. Add your own datasets using the data processing scripts
4. Integrate with your applications using the provided APIs

For more details, see the documentation in the `docs/` directory.