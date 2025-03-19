from setuptools import setup, find_packages

setup(
    name="rag_system",
    version="0.1.0",
    description="A RAG system for financial data processing with table support",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "llama-index",  # Core RAG functionality
        "llama-index-embeddings-huggingface",  # For HuggingFace embeddings
        "llama-index-retrievers-bm25",  # For BM25 retrieval
        "rank_bm25",    # For BM25 ranking
        "pandas",       # Table processing
        "numpy",        # Numerical operations
        "tqdm",        # Progress bars
        "chromadb",    # Vector store
        "python-dotenv",  # Environment variables
        "openai",      # OpenAI API
        "transformers", # For HuggingFace models
        "torch",       # Required for HuggingFace
        "sentence-transformers",  # For embeddings
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 