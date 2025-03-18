import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data paths
DATA_PATH = "data/sample_knowledge.json"
PERSIST_DIR = "output/chroma_db"

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# LLM configuration
LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.1
MAX_TOKENS = 4096

# Retrieval parameters
TOP_K = 5
SIMILARITY_TOP_K = 5
RERANK_TOP_N = 3

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Table parameters
TABLE_DETECTION_THRESHOLD = 2  # Minimum number of '|' characters to consider a line as part of a table 