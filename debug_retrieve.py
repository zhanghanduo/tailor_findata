import logging
import sys
logging.basicConfig(level=logging.DEBUG)

from rag_system.models.hybrid_retriever import HybridFinancialRetriever
from llama_index.core.schema import QueryBundle
from llama_index.core import VectorStoreIndex

# Create dummy indices
dummy_index = VectorStoreIndex.from_documents([])

# Create a retriever with the dummy indices
retriever = HybridFinancialRetriever(
    pre_text_index=dummy_index,
    post_text_index=dummy_index,
    table_index=dummy_index,
)

# Try calling the retrieve method directly
print("Attempting to call retrieve...")
try:
    results = retriever.retrieve("test query")
    print("Success! Got results:", results)
except Exception as e:
    print("Error:", str(e))
    import traceback
    traceback.print_exc() 