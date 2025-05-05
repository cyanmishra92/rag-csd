#!/bin/bash
# Script to run common RAG-CSD operations with customizable parameters

# Default values
CONFIG_FILE="config/default.yaml"
VECTOR_DB_DIR="data/vectors"
INPUT_DIR="data/raw"
OUTPUT_DIR="results"
TOP_K=5
BATCH_SIZE=32
USE_CSD=false
QUERY_FILE="data/query_template.json"
LOG_LEVEL="INFO"

# Function to print usage
print_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  pdf2vec      Process PDF files and create vector database"
    echo "  query        Run a single query"
    echo "  batch        Run batch queries from a file"
    echo "  benchmark    Run benchmark comparing standard vs CSD"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE       Configuration file (default: $CONFIG_FILE)"
    echo "  -i, --input DIR/FILE    Input directory or file (default: $INPUT_DIR)"
    echo "  -v, --vector-db DIR     Vector database directory (default: $VECTOR_DB_DIR)"
    echo "  -o, --output DIR        Output directory for results (default: $OUTPUT_DIR)"
    echo "  -k, --top-k NUM         Number of documents to retrieve (default: $TOP_K)"
    echo "  -b, --batch-size NUM    Batch size for embedding (default: $BATCH_SIZE)"
    echo "  -q, --query TEXT/FILE   Query text or file with queries (default: $QUERY_FILE)"
    echo "  -l, --log-level LEVEL   Logging level (default: $LOG_LEVEL)"
    echo "  --csd                   Use CSD simulation (default: disabled)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 pdf2vec -i papers/ -v data/paper_vectors"
    echo "  $0 pdf2vec -i papers.txt -v data/paper_vectors"
    echo "  $0 query -q \"What is computational storage?\" -k 10"
    echo "  $0 batch -q queries.json -o results/batch1"
    echo "  $0 benchmark --csd -k 5 -o results/benchmark_k5"
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

# Parse command
COMMAND=$1
shift

# Parse options
while [ $# -gt 0 ]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -v|--vector-db)
            VECTOR_DB_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -q|--query)
            QUERY_FILE="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --csd)
            USE_CSD=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run appropriate command
case "$COMMAND" in
    pdf2vec)
        echo "Processing PDFs and creating vector database..."
        echo "Input: $INPUT_DIR"
        echo "Vector DB: $VECTOR_DB_DIR"
        echo "Config: $CONFIG_FILE"
        echo "Batch size: $BATCH_SIZE"
        
        python scripts/process_pdfs.py \
            --input "$INPUT_DIR" \
            --output "$VECTOR_DB_DIR" \
            --config "$CONFIG_FILE" \
            --batch-size "$BATCH_SIZE" \
            --log-level "$LOG_LEVEL"
        
        if [ $? -eq 0 ]; then
            echo "Vector database created successfully."
        else
            echo "Error creating vector database."
            exit 1
        fi
        ;;
        
    query)
        echo "Running query..."
        echo "Query: $QUERY_FILE"
        echo "Vector DB: $VECTOR_DB_DIR"
        echo "Config: $CONFIG_FILE"
        echo "Top-k: $TOP_K"
        
        # Check if query is a file or a direct query
        if [ -f "$QUERY_FILE" ]; then
            # It's a file, use the first query
            QUERY_TEXT=$(head -n 1 "$QUERY_FILE")
        else
            # It's a direct query
            QUERY_TEXT="$QUERY_FILE"
        fi
        
        # Add CSD flag if enabled
        CSD_FLAG=""
        if [ "$USE_CSD" = true ]; then
            CSD_FLAG="--use-csd"
            echo "Using CSD simulation"
        fi
        
        python examples/simple_query.py \
            --config "$CONFIG_FILE" \
            --vector-db "$VECTOR_DB_DIR" \
            --query "$QUERY_TEXT" \
            --top-k "$TOP_K" \
            --log-level "$LOG_LEVEL" \
            $CSD_FLAG
        ;;
        
    batch)
        echo "Running batch queries..."
        echo "Query file: $QUERY_FILE"
        echo "Vector DB: $VECTOR_DB_DIR"
        echo "Config: $CONFIG_FILE"
        echo "Top-k: $TOP_K"
        echo "Output: $OUTPUT_DIR/batch_results.json"
        
        # Add CSD flag if enabled
        CSD_FLAG=""
        if [ "$USE_CSD" = true ]; then
            CSD_FLAG="--use-csd"
            echo "Using CSD simulation"
        fi
        
        python scripts/batch_query.py \
            --config "$CONFIG_FILE" \
            --vector-db "$VECTOR_DB_DIR" \
            --queries "$QUERY_FILE" \
            --output "$OUTPUT_DIR/batch_results.json" \
            --top-k "$TOP_K" \
            --batch-size "$BATCH_SIZE" \
            --log-level "$LOG_LEVEL" \
            $CSD_FLAG
            
        if [ $? -eq 0 ]; then
            echo "Batch query results saved to $OUTPUT_DIR/batch_results.json"
        else
            echo "Error running batch queries."
            exit 1
        fi
        ;;
        
    benchmark)
        echo "Running benchmark..."
        echo "Vector DB: $VECTOR_DB_DIR"
        echo "Config: $CONFIG_FILE"
        echo "Top-k: $TOP_K"
        echo "Output: $OUTPUT_DIR/benchmark_results.json"
        
        python examples/benchmark.py \
            --config "$CONFIG_FILE" \
            --vector-db "$VECTOR_DB_DIR" \
            --queries "data/test_queries.json" \
            --top-k "$TOP_K" \
            --output "$OUTPUT_DIR/benchmark_results.json" \
            --log-level "$LOG_LEVEL"
            
        if [ $? -eq 0 ]; then
            echo "Benchmark results saved to $OUTPUT_DIR/benchmark_results.json"
        else
            echo "Error running benchmark."
            exit 1
        fi
        ;;
        
    *)
        echo "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac

exit 0
