# RAG-CSD: Retrieval-Augmented Generation with Computational Storage Devices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

RAG-CSD is a framework for implementing Retrieval-Augmented Generation (RAG) systems with a focus on offloading vector operations to Computational Storage Devices (CSDs). This project specifically implements the Embedding (E), Retrieval (R), and Augmentation (A) components of the RAG pipeline, with the goal of moving computationally intensive vector operations closer to where the data resides.

## Architecture

The system follows a three-stage process:

1. **Embedding (E)**: Tokenize queries and encode them into vector embeddings using a small transformer model.
2. **Retrieval (R)**: Perform vector similarity search against a vector database to find the most relevant documents.
3. **Augmentation (A)**: Enhance the original query with the retrieved documents for improved context.

This implementation focuses on optimizing the first two stages (E and R) for execution on Computational Storage Devices, while the actual generation stage (G) is intended to be performed on GPUs.

## Features

- Efficient vector embedding generation on CSDs
- Fast similarity search optimized for computational storage
- Configurable retrieval parameters (top-k, similarity metrics, etc.)
- Python reference implementation with CSD simulation
- Benchmarking tools for performance evaluation
- Support for various embedding models and vector databases

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-csd.git
cd rag-csd
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install development dependencies (optional):
```bash
pip install -r requirements-dev.txt
```

5. Install the package in development mode:
```bash
pip install -e .
```

## Configuration

The system is configured using YAML files located in the `config/` directory. The default configuration is in `config/default.yaml`. You can create custom configuration files or override specific parameters via command-line arguments.

Key configuration parameters include:

- `embedding.model`: The embedding model to use
- `embedding.dimension`: The dimension of the embeddings
- `retrieval.top_k`: Number of documents to retrieve
- `retrieval.similarity_metric`: Metric for computing similarity (cosine, dot, euclidean)
- `csd.enabled`: Whether to use the CSD simulation
- `csd.latency`: Simulated CSD latency in milliseconds

## Usage

### Creating a Vector Database

Use the provided script to create a vector database from your text corpus:

```bash
python scripts/create_vector_db.py --config config/default.yaml --input data/raw/ --output data/vectors/
```

### Running a Simple Query

```python
from rag_csd.embedding import Encoder
from rag_csd.retrieval import VectorStore
from rag_csd.augmentation import Augmentor

# Initialize components
encoder = Encoder()
vector_store = VectorStore("data/vectors/")
augmentor = Augmentor()

# Process a query
query = "What is computational storage?"
query_embedding = encoder.encode(query)
retrieved_docs = vector_store.search(query_embedding, top_k=5)
augmented_query = augmentor.augment(query, retrieved_docs)

print(f"Original query: {query}")
print(f"Retrieved documents: {len(retrieved_docs)}")
print(f"Augmented query: {augmented_query}")
```

### Benchmarking

```bash
python examples/benchmark.py --config config/benchmark.yaml
```

## Development Roadmap

1. **Phase 1**: Python reference implementation with CSD simulation
2. **Phase 2**: Integration with real CSD hardware interfaces
3. **Phase 3**: Performance optimization and tuning
4. **Phase 4**: Integration with generation models (optional)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project builds on the concepts of Retrieval-Augmented Generation (RAG)
- Thanks to the research community for advances in computational storage technologies