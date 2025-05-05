# Process PDFs
./run_rag.sh pdf2vec --input papers/ --vector-db data/paper_vectors

# Run queries against the PDF vector database
./run_rag.sh batch --query data/query_template.json --vector-db data/paper_vectors --output results/pdf_queries --top-k 10

# Compare performance with and without CSD
./run_rag.sh batch --query data/query_template.json --vector-db data/paper_vectors --output results/standard_mode
./run_rag.sh batch --query data/query_template.json --vector-db data/paper_vectors --output results/csd_mode --csd

# Custom top-k and batch size
./run_rag.sh query --query "What are the benefits of computational storage?" --top-k 15 --batch-size 64
