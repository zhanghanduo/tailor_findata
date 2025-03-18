# Financial Data RAG System

A specialized Retrieval-Augmented Generation (RAG) system for financial data using llama-index, designed to handle complex financial queries involving tables, cross-document references, and numerical calculations.

## Overview

This RAG system is specifically designed to work with financial data from annual reports and financial statements. It can:

- Process and extract tables from financial documents
- Handle numerical data and perform calculations
- Break down complex financial queries into simpler sub-questions
- Retrieve information across multiple documents and tables
- Generate comprehensive responses to financial questions

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements_rag.txt
```

3. Set up your OpenAI API key:
```bash
# Create a .env file with your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Running the System

1. Run example queries:
```bash
python -m rag_system.example
```

2. Use the command line interface:
```bash
python -m rag_system.main --data_path data/sample_knowledge.json --query "What was the revenue in 2008?"
```

3. Or use it in your Python code:
```python
from rag_system.main import FinancialRAGSystem

# Initialize the system
rag_system = FinancialRAGSystem(
    data_path="data/sample_knowledge.json",
    persist_dir="output/chroma_db"
)
rag_system.initialize()

# Ask a question
result = rag_system.query("What was Jack Henry's net income in 2009?")
print(result["response"])
```

## Features in Detail

### Table Processing

The system automatically detects and extracts tables from financial texts, converting them into structured data that can be used for retrieval and numerical calculations.

### Query Transformation

Complex financial queries are transformed using:
- Query rewriting to better match financial terminology
- Query expansion to capture different phrasings
- Sub-question generation to break down complex queries

### Multi-Stage Retrieval

The system implements a sophisticated retrieval strategy:
- Separate indices for text and tables
- Context-aware retrieval for broader understanding
- Numerical data detection and prioritization
- Hybrid vector and keyword search

### Numerical Operations

For queries involving calculations:
- Automatic detection of calculation requirements
- Extraction of relevant numerical values
- Statistics computation (mean, median, min, max)
- Time-based comparisons

## System Architecture

![RAG System Architecture](rag_system/docs/architecture_diagram.png)

The system is composed of these main components:
1. **Data Processing Pipeline**: Handles document loading, table extraction, and chunking
2. **Indexing System**: Creates separate indices for different types of content
3. **Query Processing**: Transforms queries for optimal retrieval
4. **Retrieval System**: Combines multiple retrieval strategies
5. **Response Generation**: Synthesizes information and performs calculations

## Extending the System

The modular design makes it easy to extend:

- Add new data sources in `data_loader.py`
- Implement custom chunking strategies in `custom_chunker.py`
- Create new retrieval methods in `retriever.py`
- Add specialized numerical operations in `numerical_operations.py`

## Limitations

- Currently works with the provided sample financial data
- Requires an OpenAI API key for LLM operations
- Table detection is based on pipe characters and may not work for all formats

## Future Improvements

- Support for more complex table formats
- Integration with financial databases and APIs
- Implementation of financial-specific calculation tools
- Enhanced cross-report comparison capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

For more detailed information, please see the [full documentation](rag_system/README.md). 