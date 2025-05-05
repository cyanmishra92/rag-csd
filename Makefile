# RAG-CSD Makefile

.PHONY: all setup install dev clean test lint format sample-data vector-db run-query

# Default target
all: setup install

# Setup the project structure
setup:
	@echo "Creating project structure..."
	mkdir -p config data/raw data/processed data/vectors logs
	mkdir -p examples rag_csd/embedding rag_csd/retrieval rag_csd/augmentation rag_csd/csd rag_csd/utils
	touch data/raw/.gitkeep data/processed/.gitkeep data/vectors/.gitkeep
	@echo "Project structure created."

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed."

# Install development dependencies
dev:
	@echo "Installing development dependencies..."
	pip install -r requirements-dev.txt
	@echo "Development dependencies installed."

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ .coverage htmlcov/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".coverage" -delete
	@echo "Clean complete."

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v
	@echo "Tests complete."

# Run code linting
lint:
	@echo "Running linting..."
	flake8 rag_csd/ tests/ examples/ scripts/
	pylint rag_csd/ tests/ examples/ scripts/
	mypy rag_csd/
	@echo "Linting complete."

# Format code
format:
	@echo "Formatting code..."
	black rag_csd/ tests/ examples/ scripts/
	isort rag_csd/ tests/ examples/ scripts/
	@echo "Formatting complete."

# Create sample data
sample-data:
	@echo "Creating sample data..."
	python scripts/create_sample_data.py --output-dir data/raw --query-dir data
	@echo "Sample data created."

# Create vector database
vector-db:
	@echo "Creating vector database..."
	python scripts/create_vector_db.py --config config/default.yaml --input data/raw --output data/vectors
	@echo "Vector database created."

# Run a sample query
run-query:
	@echo "Running sample query..."
	python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "What is computational storage and how does it work?"
	@echo "Query complete."

# Run a sample query with CSD simulation
run-query-csd:
	@echo "Running sample query with CSD simulation..."
	python examples/simple_query.py --config config/default.yaml --vector-db data/vectors --query "What is computational storage and how does it work?" --use-csd
	@echo "Query with CSD simulation complete."

# Setup full development environment
dev-setup: setup install dev sample-data vector-db

# Quick start: setup everything and run a sample query
quickstart: dev-setup run-query

# Help message
help:
	@echo "RAG-CSD Makefile targets:"
	@echo "  all         : Setup project structure and install dependencies"
	@echo "  setup       : Create project directory structure"
	@echo "  install     : Install dependencies"
	@echo "  dev         : Install development dependencies"
	@echo "  clean       : Remove build artifacts and caches"
	@echo "  test        : Run tests"
	@echo "  lint        : Run code linting"
	@echo "  format      : Format code"
	@echo "  sample-data : Create sample data for testing"
	@echo "  vector-db   : Create vector database from sample data"
	@echo "  run-query   : Run a sample query"
	@echo "  run-query-csd : Run a sample query with CSD simulation"
	@echo "  dev-setup   : Setup full development environment"
	@echo "  quickstart  : Setup everything and run a sample query"