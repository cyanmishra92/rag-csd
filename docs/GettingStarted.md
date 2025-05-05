# Getting Started with RAG-CSD

This guide provides step-by-step instructions to set up and run the RAG-CSD system.

## Prerequisites

- Python 3.8 or higher
- pip
- git
- make (optional, but recommended)

## Quick Setup

For a quick setup that creates all necessary directories, installs dependencies, generates sample data, and creates a vector database, run:

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-csd.git
cd rag-csd

# Set up the environment and run a test query
make quickstart
```

## Manual Setup (Step by Step)

If you prefer to set up the project step by step or don't have `make` available:

### 1. Clone and navigate to the repository

```bash
git clone https://github.com/yourusername/rag-csd.git
cd rag-csd
```

### 2. Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Or activate on Windows
# venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

### 4. Create project directories

```bash
mkdir -p config data/raw data/processed data/vectors logs
mkdir -p examples rag_csd/embedding rag_csd/retrieval rag_csd/augmentation rag_csd/csd rag_csd/utils
```

### 5. Generate sample data

```bash
python scripts/create_sample_data.py --output-dir data/raw --query-dir data
```

### 6. Create vector database

```bash
python scripts/create_vector_db.py --config config/default.yaml --input data/raw --output data/vectors
```

### 7. Run a test query

```bash
python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "What is computational storage and how does it work?"
```

## Testing CSD Simulation

To run a query with CSD simulation enabled:

```bash
python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "What is computational storage and how does it work?" --use-csd
```

## Common Commands

Here are some useful commands for working with the system:

```bash
# Create sample data
python scripts/create_sample_data.py --output-dir data/raw --query-dir data

# Create vector database
python scripts/create_vector_db.py --config config/default.yaml --input data/raw --output data/vectors

# Run a query (standard mode)
python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "YOUR QUERY HERE"

# Run a query (CSD simulation mode)
python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "YOUR QUERY HERE" --use-csd

# Run a query with larger number of results
python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "YOUR QUERY HERE" --top-k 10
```

## Configuration

You can modify the system behavior by editing the configuration file at `config/default.yaml`. Key parameters include:

- **embedding.model**: The model used for generating embeddings
- **retrieval.top_k**: Number of documents to retrieve
- **retrieval.similarity_metric**: Similarity metric (cosine, dot, euclidean)
- **csd.enabled**: Whether to enable CSD simulation
- **csd.latency**: Simulated CSD latency in milliseconds

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure you've activated the virtual environment and installed all dependencies.

2. **ImportError with FAISS**: You may need to install FAISS separately. For CPU-only use:
   ```bash
   pip install faiss-cpu
   ```
   For GPU support:
   ```bash
   pip install faiss-gpu
   ```

3. **No documents found in vector database**: Ensure you've run `create_sample_data.py` and `create_vector_db.py` before running queries.

### Logging

To increase logging verbosity for debugging:

```bash
python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "YOUR QUERY HERE" --log-level DEBUG
```

## Next Steps

Once you're familiar with the basic operation, you can:

1. Replace the sample data with your own corpus
2. Experiment with different embedding models and parameters
3. Extend the CSD simulation for your specific hardware target
4. Implement additional retrieval or augmentation strategies