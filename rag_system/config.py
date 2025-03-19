import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data paths
DATA_PATH = "data/knowledge.json"
PERSIST_DIR = "output/chroma_db"

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
# EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"

# LLM configuration
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1
MAX_TOKENS = 4096

# Retrieval parameters
TOP_K = 10
SIMILARITY_TOP_K = 8
RERANK_TOP_N = 5

# Chunking parameters
CHUNK_SIZE = 384
CHUNK_OVERLAP = 75

# Table parameters
TABLE_DETECTION_THRESHOLD = 1 