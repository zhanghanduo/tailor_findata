# Financial Data RAG System

A Retrieval-Augmented Generation (RAG) system specifically designed for processing financial documents with table support. The system handles both textual content and tabular data, providing specialized processing for pre-text, post-text, and table content in both markdown and structured formats.

## Features

- Specialized handling of financial documents with tables
- Support for both markdown and structured table formats
- **Hybrid retrieval** combining vector search and BM25 for better results
- Progress tracking for long-running operations
- Checkpointing and resumable operations
- Memory-efficient batch processing
- Hierarchical document relationships

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Configuration

1. Create a `.env` file in the project root with your configuration:
```env
OPENAI_API_KEY=your_api_key_here
DATA_PATH=path/to/your/data
PERSIST_DIR=path/to/persist/indices
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=gpt-4o-mini-turbo
TEMPERATURE=0.1
```

## Usage

### Basic Usage

```python
from rag_system.main import FinancialRAGSystem

# Initialize the system
rag = FinancialRAGSystem()

# Initialize with custom settings
rag = FinancialRAGSystem(
    data_path="path/to/data",
    persist_dir="path/to/indices",
    embedding_model="BAAI/bge-small-en-v1.5",
    llm_model="gpt-4o-mini-turbo",
    temperature=0.1,
    rebuild_indices=False,
    vector_weight=0.6,  # Weight for semantic vector search
    bm25_weight=0.4     # Weight for keyword-based BM25 search
)

# Initialize the system (this will load/process data and build indices)
rag.initialize()

# Query the system
result = rag.query("What was the revenue growth in Q2 2023?")
print(result["response"])
```

### Command Line Usage

```bash
# Run with default settings
python -m rag_system.main

# Run with custom settings
python -m rag_system.main \
    --data_path path/to/data \
    --persist_dir path/to/indices \
    --embedding_model BAAI/bge-small-en-v1.5 \
    --llm_model gpt-4o-mini-turbo \
    --temperature 0.1 \
    --rebuild_indices \
    --vector_weight 0.6 \
    --bm25_weight 0.4 \
    --query "What was the revenue growth in Q2 2023?"
```

## Hybrid Retrieval System

The system uses a hybrid retrieval approach that combines:

1. **Dense Retrieval (Vector Search)**: Captures semantic meaning using embedding models
2. **Sparse Retrieval (BM25)**: Captures keyword matches using BM25 algorithm

This hybrid approach provides several benefits:
- Better coverage of both semantic and exact matching
- Improved retrieval of numerical data in tables
- Enhanced robustness to different query phrasings

You can adjust the balance between these approaches using the `vector_weight` and `bm25_weight` parameters. For:
- More semantic search: Increase `vector_weight` relative to `bm25_weight`
- More keyword matching: Increase `bm25_weight` relative to `vector_weight`

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

This project follows PEP 8 guidelines. To check code style:

```bash
flake8 rag_system/
```

## License

MIT License
