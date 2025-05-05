# RAG-CSD Project Summary

We've created a complete framework for implementing Retrieval-Augmented Generation (RAG) with a focus on Computational Storage Devices (CSDs). This implementation focuses on the Embedding (E), Retrieval (R), and Augmentation (A) components, with provisions to move vector operations to computational storage.

## Project Structure Overview

```
rag-csd/
├── config/                 # Configuration files
├── data/                   # Data storage
│   ├── raw/                # Original corpus
│   ├── processed/          # Processed text chunks
│   └── vectors/            # Vector embeddings
├── rag_csd/                # Main package
│   ├── embedding/          # Query embedding modules
│   ├── retrieval/          # Vector retrieval modules
│   ├── augmentation/       # Query augmentation modules
│   ├── csd/                # CSD simulation and interface
│   └── utils/              # Utility functions
├── scripts/                # Helper scripts
├── examples/               # Usage examples
└── tests/                  # Unit tests
```

## Key Components

1. **Embedding Module (`rag_csd/embedding/encoder.py`)**
   - Handles tokenization and vector encoding of queries
   - Supports both standard execution and CSD simulation

2. **Retrieval Module (`rag_csd/retrieval/vector_store.py`)**
   - Manages vector database access and similarity search
   - Implements FAISS for efficient vector search
   - Includes CSD simulation mode for storage-side search

3. **Augmentation Module (`rag_csd/augmentation/augmentor.py`)**
   - Enhances queries with retrieved document content
   - Supports different augmentation strategies

4. **CSD Simulation (`rag_csd/csd/simulator.py`)**
   - Simulates the behavior of computational storage devices
   - Models latency, bandwidth, and parallel operations

5. **Utility Scripts**
   - `scripts/create_sample_data.py`: Generates test corpus
   - `scripts/create_vector_db.py`: Creates vector database
   - `examples/simple_query.py`: Demonstrates RAG pipeline

## Configuration

The system is configured via YAML files in the `config/` directory:

- **General Settings**: Device selection, logging, etc.
- **Data Processing**: Chunking parameters
- **Embedding**: Model selection and parameters
- **Retrieval**: Vector DB type, similarity metrics
- **Augmentation**: Strategy and formatting
- **CSD Simulation**: Latency, bandwidth, and operation parameters

## Getting Started

### Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/rag-csd.git
   cd rag-csd
   ```

2. **Create environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   make install  # Installs required dependencies
   ```

3. **Create sample data and vector database**:
   ```bash
   make sample-data  # Creates test corpus
   make vector-db    # Creates vector database
   ```

### Running a Query

To test the system with a simple query:

```bash
make run-query
```

To run with CSD simulation enabled:

```bash
make run-query-csd
```

### Quick Start

To set up everything and run a query in one command:

```bash
make quickstart
```

## Next Steps

1. **Implement Real CSD Interface**:
   - Replace simulation with actual CSD hardware interface
   - Adapt code to work with specific CSD hardware APIs

2. **Performance Optimization**:
   - Benchmark different vector index techniques
   - Optimize for specific hardware configurations
   - Explore quantization for reduced memory footprint
   - Implement caching mechanisms

3. **Integration with Generation Models**:
   - Add optional integration with LLMs for the generation stage
   - Support for local models (e.g., LLaMA, Mistral) or API-based models
   - Create end-to-end RAG pipeline examples

4. **Additional Features**:
   - Implement reranking of retrieved documents
   - Add support for more vector database backends
   - Create visualization tools for retrieval quality
   - Add evaluation metrics for retrieval accuracy

## Implementation Details

### Computational Storage Simulation

The current implementation simulates CSD behavior with configurable parameters:

- **Latency**: Simulates processing delay in milliseconds
- **Bandwidth**: Models data transfer speeds
- **Parallel Operations**: Limits concurrent operations
- **Memory**: Constrains available computational resources

This allows for testing and optimization before moving to actual CSD hardware.

### Vector Database

The vector database implementation uses FAISS and supports three similarity metrics:

- **Cosine Similarity**: Measures angle between vectors (normalized dot product)
- **Dot Product**: Measures directional similarity (unnormalized)
- **Euclidean Distance**: Measures straight-line distance

### Query Augmentation

The augmentation component offers three strategies:

- **Concatenate**: Simply appends retrieved chunks to the query
- **Template**: Uses a configurable template to format the query and context
- **Weighted**: Includes relevance scores and prioritizes by similarity

## Migration Path to Real CSDs

To adapt this code for real Computational Storage Devices:

1. **Hardware Selection**:
   - Identify target CSD hardware with compute capabilities
   - Document the hardware API and constraints

2. **Interface Development**:
   - Replace `rag_csd/csd/simulator.py` with actual CSD interface
   - Implement native code for the CSD platform (C/C++)
   - Create firmware for vector operations on the device

3. **Data Management**:
   - Implement efficient storage format for vectors on CSDs
   - Develop data transfer protocols between host and device

4. **Performance Tuning**:
   - Balance workload between host and CSD
   - Optimize for specific CSD hardware characteristics

## Conclusion

This RAG-CSD framework provides a solid foundation for exploring Retrieval-Augmented Generation with computational storage. The modular design allows easy adaptation to different hardware platforms and use cases, while the simulation components enable development and testing without actual CSD hardware.

By moving vector operations closer to storage, this approach has the potential to significantly reduce data movement and improve overall system efficiency for RAG applications, especially for large-scale deployments.

## Resources

For more information on the underlying technologies:

- [FAISS Library](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [Sentence Transformers](https://www.sbert.net/) - For generating embeddings
- [SNIA Computational Storage](https://www.snia.org/education/what-is-computational-storage) - Standards organization for computational storage
- [NVMe Computational Storage](https://nvmexpress.org/computational-storage-is-available-in-nvme-2-0/) - NVMe standards for computational storage