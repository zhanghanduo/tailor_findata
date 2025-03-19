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
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

class FinancialIndexBuilder:
    """
    Builder class for creating and persisting vector indices for financial data.
    """
    
    def __init__(
        self,
        persist_dir: str,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        llm_model_name: str = "gpt-4o-mini-turbo",
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
        
        # Create directory structure - use the persist_dir as is without adding subdirectories
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize embeddings and LLM
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        self.llm = OpenAI(model=llm_model_name, temperature=temperature)
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        # Initialize Chroma client using the specified persist_dir
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Vector stores
        self.pre_text_store = None
        self.post_text_store = None
        self.table_store = None
        
        # Indices
        self.pre_text_index = None
        self.post_text_index = None
        self.table_index = None
    
    def build_indices(self, nodes: List[TextNode]) -> Dict[str, VectorStoreIndex]:
        """
        Build indices for different types of nodes.
        """
        # Ensure directory exists
        os.makedirs(self.persist_dir, exist_ok=True)

        # Group nodes by type
        pre_text_nodes = [node for node in nodes if node.metadata.get("text_type") == "pre_text"]
        post_text_nodes = [node for node in nodes if node.metadata.get("text_type") == "post_text"]
        table_nodes = [node for node in nodes if node.metadata.get("is_table", False)]
        
        logger.info(f"Building indices for: {len(pre_text_nodes)} pre_text nodes, "
                   f"{len(post_text_nodes)} post_text nodes, "
                   f"{len(table_nodes)} table nodes")
        
        # Delete existing collections to start fresh
        try:
            self.chroma_client.delete_collection("pre_text_collection")
            self.chroma_client.delete_collection("post_text_collection")
            self.chroma_client.delete_collection("table_collection")
            logger.info("Deleted existing collections to rebuild from scratch")
        except ValueError:
            # Collections don't exist, which is fine
            pass
        
        # Create vector stores with fresh collections
        pre_text_collection = self.chroma_client.create_collection("pre_text_collection")
        post_text_collection = self.chroma_client.create_collection("post_text_collection")
        table_collection = self.chroma_client.create_collection("table_collection")
        
        self.pre_text_store = ChromaVectorStore(chroma_collection=pre_text_collection)
        self.post_text_store = ChromaVectorStore(chroma_collection=post_text_collection)
        self.table_store = ChromaVectorStore(chroma_collection=table_collection)
        
        # Create storage contexts
        pre_text_storage = StorageContext.from_defaults(vector_store=self.pre_text_store)
        post_text_storage = StorageContext.from_defaults(vector_store=self.post_text_store)
        table_storage = StorageContext.from_defaults(vector_store=self.table_store)
        
        # Create indices with storage contexts to ensure nodes are stored
        logger.info(f"Building pre_text index with {len(pre_text_nodes)} nodes...")
        self.pre_text_index = VectorStoreIndex(
            nodes=pre_text_nodes,
            storage_context=pre_text_storage,
            show_progress=True
        )
        
        # Verify the collection has documents
        pre_text_doc_count = pre_text_collection.count()
        logger.info(f"Pre_text collection has {pre_text_doc_count} documents")
        
        logger.info(f"Building post_text index with {len(post_text_nodes)} nodes...")
        self.post_text_index = VectorStoreIndex(
            nodes=post_text_nodes,
            storage_context=post_text_storage,
            show_progress=True
        )
        
        # Verify the collection has documents
        post_text_doc_count = post_text_collection.count()
        logger.info(f"Post_text collection has {post_text_doc_count} documents")
        
        logger.info(f"Building table index with {len(table_nodes)} nodes...")
        self.table_index = VectorStoreIndex(
            nodes=table_nodes,
            storage_context=table_storage,
            show_progress=True
        )
        
        # Verify the collection has documents
        table_doc_count = table_collection.count()
        logger.info(f"Table collection has {table_doc_count} documents")
        
        logger.info("Successfully built all indices")
        
        return {
            "pre_text": self.pre_text_index,
            "post_text": self.post_text_index,
            "table": self.table_index
        }
    
    def _get_or_create_collection(self, name: str) -> Collection:
        """
        Get an existing collection or create a new one if it doesn't exist.
        
        Args:
            name: Name of the collection.
            
        Returns:
            ChromaDB collection.
        """
        try:
            # Try to get the existing collection first
            collection = self.chroma_client.get_collection(name)
            # Log how many documents are in the collection
            count = collection.count()
            logger.info(f"Using existing collection '{name}' with {count} documents")
            return collection
        except ValueError:
            # Collection doesn't exist, create a new one
            logger.info(f"Creating new collection '{name}'")
            return self.chroma_client.create_collection(name)
    
    def load_indices(self) -> Dict[str, VectorStoreIndex]:
        """
        Load existing indices from disk.
        
        Returns:
            Dictionary containing the loaded indices.
        """
        logger.info(f"Loading indices from {self.persist_dir}")
        
        # Ensure directory exists
        os.makedirs(self.persist_dir, exist_ok=True)

        try:
            # Get collections
            pre_text_collection = self.chroma_client.get_collection("pre_text_collection")
            post_text_collection = self.chroma_client.get_collection("post_text_collection")
            table_collection = self.chroma_client.get_collection("table_collection")
            
            # Log collection sizes
            pre_text_count = pre_text_collection.count()
            post_text_count = post_text_collection.count()
            table_count = table_collection.count()
            
            logger.info(f"Found collections with document counts: pre_text={pre_text_count}, post_text={post_text_count}, table={table_count}")
            
            # Create vector stores
            self.pre_text_store = ChromaVectorStore(chroma_collection=pre_text_collection)
            self.post_text_store = ChromaVectorStore(chroma_collection=post_text_collection)
            self.table_store = ChromaVectorStore(chroma_collection=table_collection)
            
            # Create storage contexts (helps with proper loading)
            pre_text_storage = StorageContext.from_defaults(vector_store=self.pre_text_store)
            post_text_storage = StorageContext.from_defaults(vector_store=self.post_text_store)
            table_storage = StorageContext.from_defaults(vector_store=self.table_store)
            
            # Create indices from vector stores with storage contexts
            self.pre_text_index = VectorStoreIndex.from_vector_store(
                vector_store=self.pre_text_store,
                storage_context=pre_text_storage
            )
            
            self.post_text_index = VectorStoreIndex.from_vector_store(
                vector_store=self.post_text_store,
                storage_context=post_text_storage
            )
            
            self.table_index = VectorStoreIndex.from_vector_store(
                vector_store=self.table_store,
                storage_context=table_storage
            )
            
            # Verify indices were loaded successfully
            if not hasattr(self.pre_text_index, '_vector_store') or not hasattr(self.post_text_index, '_vector_store') or not hasattr(self.table_index, '_vector_store'):
                logger.warning("Failed to properly link vector stores to indices")
                raise ValueError("Indices did not load properly, vector stores not linked")
            
            logger.info("Successfully loaded all indices")
            
            return {
                "pre_text": self.pre_text_index,
                "post_text": self.post_text_index,
                "table": self.table_index
            }
            
        except Exception as e:
            logger.error(f"Failed to load indices: {str(e)}")
            raise
    
    def get_indices(self) -> Dict[str, VectorStoreIndex]:
        """
        Get the current indices.
        
        Returns:
            Dictionary containing the current indices.
        """
        indices = {
            "pre_text": self.pre_text_index,
            "post_text": self.post_text_index,
            "table": self.table_index
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

    def build_indices_in_batches(self, nodes: List[TextNode], batch_size: int = 1000):
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            # Process batch 