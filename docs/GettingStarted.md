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

## Using the Convenience Script

The repository includes a convenience script (`run_rag.sh`) for common operations:

```bash
# Make the script executable
chmod +x run_rag.sh

# Process PDF files and create a vector database
./run_rag.sh pdf2vec --input papers/ --vector-db data/paper_vectors

# Run a single query
./run_rag.sh query --query "What is computational storage?" --top-k 10

# Run batch queries from a file
./run_rag.sh batch --query data/query_template.json --output results/batch1

# Run benchmark
./run_rag.sh benchmark --csd --top-k 5 --output results/benchmark_k5
```

Run `./run_rag.sh --help` for more options.

## Working with PDFs

The system can process PDF files to create vector databases:

```bash
# Process a single PDF
python scripts/process_pdfs.py --input path/to/paper.pdf --output data/vectors --config config/default.yaml

# Process multiple PDFs from a directory
python scripts/process_pdfs.py --input path/to/papers/ --output data/vectors --config config/default.yaml

# Process PDFs listed in a text file
python scripts/process_pdfs.py --input papers_list.txt --output data/vectors --config config/default.yaml
```

## Running Batch Queries

You can run multiple queries at once using the batch query script:

```bash
# Run queries from a JSON file
python scripts/batch_query.py --config config/default.yaml --vector-db data/vectors --queries data/query_template.json --output results/batch_results.json

# Run queries from a CSV file
python scripts/batch_query.py --config config/default.yaml --vector-db data/vectors --queries data/query_template.csv --output results/batch_results.json

# Run with custom top-k and batch size
python scripts/batch_query.py --config config/default.yaml --vector-db data/vectors --queries data/query_template.json --output results/batch_results.json --top-k 10 --batch-size 64
```

## Testing CSD Simulation

To run a query with CSD simulation enabled:

```bash
# Single query with CSD simulation
python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "What is computational storage?" --use-csd

# Batch queries with CSD simulation
python scripts/batch_query.py --config config/default.yaml --vector-db data/vectors --queries data/query_template.json --output results/csd_results.json --use-csd

# Using the convenience script
./run_rag.sh query --query "What is computational storage?" --csd
./run_rag.sh batch --query data/query_template.json --output results/csd_batch --csd
```

## Customizing Parameters

You can customize key parameters in several ways:

### 1. Modifying the Config File

Edit `config/default.yaml` to change default parameters like:
- `embedding.batch_size`: Batch size for embedding generation
- `retrieval.top_k`: Number of documents to retrieve
- `csd.latency`: Simulated CSD latency

### 2. Command Line Arguments

Override config values using command line arguments:

```bash
# Override top-k
python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "YOUR QUERY" --top-k 10

# Override batch size
python scripts/process_pdfs.py --input papers/ --output data/vectors --config config/default.yaml --batch-size 64
```

### 3. Using Presets

The config file includes presets for different retrieval scenarios:
- `minimal`: top-k=3, higher similarity threshold
- `standard`: top-k=5, balanced threshold
- `comprehensive`: top-k=10, lower threshold
- `exhaustive`: top-k=20, lowest threshold

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

3. **PDF Processing Errors**: Make sure you have PyPDF2 installed:
   ```bash
   pip install PyPDF2
   ```

4. **No documents found in vector database**: Ensure you've run the appropriate script to create the vector database.

### Logging

To increase logging verbosity for debugging:

```bash
python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "YOUR QUERY HERE" --log-level DEBUG
```

## Next Steps

Once you're familiar with the basic operation, you can:

1. Replace the sample data with your own corpus or PDFs
2. Create custom query templates for your specific needs
3. Experiment with different embedding models and parameters
4. Extend the CSD simulation for your specific hardware target
5. Implement additional retrieval or augmentation strategies