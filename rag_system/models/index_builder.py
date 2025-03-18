from typing import List, Dict, Any, Optional
import os
import logging
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import chromadb

logger = logging.getLogger(__name__)

class FinancialIndexBuilder:
    """
    Builder class for creating and persisting vector indices for financial data.
    """
    
    def __init__(
        self,
        persist_dir: str,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1
    ):
        """
        Initialize the index builder.
        
        Args:
            persist_dir: Directory to persist the index.
            embedding_model_name: Name of the embedding model to use.
            llm_model_name: Name of the LLM model to use.
            temperature: Temperature setting for the LLM.
        """
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        
        # Create directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize embeddings and LLM
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        self.llm = OpenAI(model=llm_model_name, temperature=temperature)
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Vector stores
        self.text_store = None
        self.table_store = None
        self.context_store = None
        
        # Indices
        self.text_index = None
        self.table_index = None
        self.context_index = None
    
    def build_indices(self, nodes: List[TextNode]) -> Dict[str, Any]:
        """
        Build separate indices for different types of content.
        
        Args:
            nodes: List of TextNode objects.
            
        Returns:
            Dictionary containing the built indices.
        """
        # Separate nodes by type
        text_nodes = []
        table_nodes = []
        
        for node in nodes:
            if node.metadata.get("is_table", False):
                table_nodes.append(node)
            else:
                text_nodes.append(node)
        
        logger.info(f"Building indices for {len(text_nodes)} text nodes and {len(table_nodes)} table nodes")
        
        # Build text index
        text_collection = self.chroma_client.create_collection("text_collection")
        self.text_store = ChromaVectorStore(chroma_collection=text_collection)
        text_storage_context = StorageContext.from_defaults(vector_store=self.text_store)
        self.text_index = VectorStoreIndex(text_nodes, storage_context=text_storage_context)
        
        # Build table index
        table_collection = self.chroma_client.create_collection("table_collection")
        self.table_store = ChromaVectorStore(chroma_collection=table_collection)
        table_storage_context = StorageContext.from_defaults(vector_store=self.table_store)
        self.table_index = VectorStoreIndex(table_nodes, storage_context=table_storage_context)
        
        # Build context index with all nodes
        context_collection = self.chroma_client.create_collection("context_collection")
        self.context_store = ChromaVectorStore(chroma_collection=context_collection)
        context_storage_context = StorageContext.from_defaults(vector_store=self.context_store)
        self.context_index = VectorStoreIndex(nodes, storage_context=context_storage_context)
        
        logger.info("Successfully built all indices")
        
        return {
            "text_index": self.text_index,
            "table_index": self.table_index,
            "context_index": self.context_index
        }
    
    def load_indices(self) -> Dict[str, Any]:
        """
        Load existing indices from disk.
        
        Returns:
            Dictionary containing the loaded indices.
        """
        logger.info(f"Loading indices from {self.persist_dir}")
        
        # Load text index
        text_collection = self.chroma_client.get_collection("text_collection")
        self.text_store = ChromaVectorStore(chroma_collection=text_collection)
        self.text_index = VectorStoreIndex.from_vector_store(self.text_store)
        
        # Load table index
        table_collection = self.chroma_client.get_collection("table_collection")
        self.table_store = ChromaVectorStore(chroma_collection=table_collection)
        self.table_index = VectorStoreIndex.from_vector_store(self.table_store)
        
        # Load context index
        context_collection = self.chroma_client.get_collection("context_collection")
        self.context_store = ChromaVectorStore(chroma_collection=context_collection)
        self.context_index = VectorStoreIndex.from_vector_store(self.context_store)
        
        logger.info("Successfully loaded all indices")
        
        return {
            "text_index": self.text_index,
            "table_index": self.table_index,
            "context_index": self.context_index
        }
    
    def get_indices(self) -> Dict[str, Any]:
        """
        Get the current indices.
        
        Returns:
            Dictionary containing the current indices.
        """
        indices = {
            "text_index": self.text_index,
            "table_index": self.table_index,
            "context_index": self.context_index
        }
        
        # Check if indices exist
        if any(index is None for index in indices.values()):
            try:
                # Try to load indices
                indices = self.load_indices()
            except Exception as e:
                logger.warning(f"Failed to load indices: {str(e)}")
                logger.warning("Indices do not exist yet. Build indices first.")
                indices = {k: None for k in indices.keys()}
        
        return indices 