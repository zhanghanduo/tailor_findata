# Financial Data RAG System

A Retrieval-Augmented Generation (RAG) system designed for financial data, capable of handling tables, cross-text references, and numerical calculations.

## Features

- **Table-aware processing**: Detects and processes tables in financial documents
- **Numerical data extraction**: Automatically extracts and analyzes numerical values in financial texts
- **Query rewriting and expansion**: Enhances retrieval by transforming complex financial queries
- **Multi-stage retrieval**: Combines fine-grained retrieval with broader context retrieval
- **SubQuestion Query Engine**: Breaks down complex financial queries into simpler sub-questions
- **Numerical computation**: Performs calculations and comparisons on retrieved financial data

## System Architecture

The system consists of the following components:

1. **Data Processing**
   - Custom table extraction and parsing
   - Hierarchical chunking for financial data
   - Numerical data extraction and annotation

2. **Indexing**
   - Separate indices for text and tables
   - Context-preserving index for broader queries
   - Vector store persistence using Chroma

3. **Query Processing**
   - Query rewriting for financial terminology
   - Query expansion for improved retrieval
   - Sub-question generation for complex queries
   - Numerical query analysis

4. **Retrieval**
   - Multi-stage retrieval from different indices
   - Hybrid retrieval combining vector search with keyword filtering
   - Table-aware retrieval prioritization
   - Context-preserving retrieval for related information

5. **Response Generation**
   - Sub-question answering for complex queries
   - Numerical operations on retrieved data
   - LLM-based calculation and analysis
   - Response synthesis from multiple sources

## Setup and Installation

1. **Install Requirements**

```bash
pip install -r requirements_rag.txt
```

2. **Set API Keys**

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic Usage

```python
from rag_system.main import FinancialRAGSystem

# Initialize the RAG system
rag_system = FinancialRAGSystem(
    data_path="path/to/your/financial_data.json",
    persist_dir="path/to/store/indices"
)

# Initialize the system (loads data, creates indices, etc.)
rag_system.initialize()

# Query the system
result = rag_system.query("What was the revenue growth from 2008 to 2009?")

# Print the response
print(result["response"])
```

### Command Line Usage

```bash
python -m rag_system.main --data_path data/sample_knowledge.json --query "What was the revenue in 2008?"
```

### Run Example Queries

```bash
python -m rag_system.example
```

## Configuration

Key configuration parameters in `config.py`:

- `EMBEDDING_MODEL`: Model for embeddings (default: "BAAI/bge-small-en-v1.5")
- `LLM_MODEL`: LLM model for query processing (default: "gpt-4o-mini-turbo")
- `CHUNK_SIZE`: Size of text chunks (default: 512)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `TABLE_DETECTION_THRESHOLD`: Threshold for detecting tables (default: 2)
- `SIMILARITY_TOP_K`: Number of top similar nodes to retrieve (default: 5)
- `RERANK_TOP_N`: Number of top nodes after reranking (default: 3)

## Customization

### Custom Data Format

The system expects financial data in the following JSON format:

```json
[
  {
    "id": "document_id",
    "content": "Text content with tables...",
    "metadata": {
      "source": "document_source",
      "filename": "original_filename"
    }
  }
]
```

### Extending the System

You can extend the system by:

1. Adding custom retrievers in `models/retriever.py`
2. Implementing additional numerical operations in `utils/numerical_operations.py`
3. Creating new query transformation strategies in `utils/query_transformers.py`

## License

[MIT License](LICENSE) 