import os
import logging
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI

from rag_system.config import (
    OPENAI_API_KEY, DATA_PATH, PERSIST_DIR, EMBEDDING_MODEL, 
    LLM_MODEL, TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP, 
    TABLE_DETECTION_THRESHOLD, TOP_K, SIMILARITY_TOP_K, RERANK_TOP_N
)
from rag_system.utils.data_loader import load_financial_data, preprocess_data
from rag_system.utils.custom_chunker import FinancialTextSplitter, create_hierarchical_nodes
from rag_system.models.index_builder import FinancialIndexBuilder
from rag_system.models.retriever import FinancialRetriever
from rag_system.models.sub_question_engine import FinancialSubQuestionEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialRAGSystem:
    """
    Main class for the Financial RAG system.
    """
    
    def __init__(
        self,
        data_path: str = DATA_PATH,
        persist_dir: str = PERSIST_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        temperature: float = TEMPERATURE,
        rebuild_indices: bool = False
    ):
        """
        Initialize the Financial RAG system.
        
        Args:
            data_path: Path to the financial data.
            persist_dir: Directory to persist the indices.
            embedding_model: Name of the embedding model to use.
            llm_model: Name of the LLM model to use.
            temperature: Temperature setting for the LLM.
            rebuild_indices: Whether to rebuild indices even if they exist.
        """
        # Load environment variables
        load_dotenv()
        
        # Set API key from config or environment
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        
        self.data_path = data_path
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.rebuild_indices = rebuild_indices
        
        # Initialize LLM
        self.llm = OpenAI(model=llm_model, temperature=temperature)
        
        # Initialize index builder
        self.index_builder = FinancialIndexBuilder(
            persist_dir=persist_dir,
            embedding_model_name=embedding_model,
            llm_model_name=llm_model,
            temperature=temperature
        )
        
        # Placeholders for other components
        self.processed_docs = None
        self.nodes = None
        self.indices = None
        self.retriever = None
        self.sub_question_engine = None
    
    def load_and_process_data(self) -> List[Dict[str, Any]]:
        """
        Load and preprocess the financial data.
        
        Returns:
            List of processed documents.
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Load raw data
        raw_data = load_financial_data(self.data_path)
        
        # Preprocess data to handle tables
        self.processed_docs = preprocess_data(
            raw_data, 
            min_pipe_chars=TABLE_DETECTION_THRESHOLD
        )
        
        logger.info(f"Processed {len(raw_data)} documents into {len(self.processed_docs)} chunks")
        
        return self.processed_docs
    
    def create_nodes(self) -> List[Any]:
        """
        Create nodes from the processed documents.
        
        Returns:
            List of TextNode objects.
        """
        logger.info("Creating nodes from processed documents")
        
        # Ensure documents are processed
        if not self.processed_docs:
            self.load_and_process_data()
        
        # Create text splitter
        text_splitter = FinancialTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            table_detection_threshold=TABLE_DETECTION_THRESHOLD
        )
        
        # Split documents into nodes
        self.nodes = text_splitter.split_documents(self.processed_docs)
        
        # Create hierarchical relationships between nodes
        self.nodes = create_hierarchical_nodes(self.nodes)
        
        logger.info(f"Created {len(self.nodes)} nodes")
        
        return self.nodes
    
    def build_or_load_indices(self) -> Dict[str, Any]:
        """
        Build or load indices for the nodes.
        
        Returns:
            Dictionary of indices.
        """
        # Check if indices need to be built
        if self.rebuild_indices:
            logger.info("Rebuilding indices")
            
            # Ensure nodes are created
            if not self.nodes:
                self.create_nodes()
            
            # Build indices
            self.indices = self.index_builder.build_indices(self.nodes)
        else:
            logger.info("Loading existing indices if available")
            
            try:
                # Try to load existing indices
                self.indices = self.index_builder.get_indices()
                
                # Check if indices exist
                if any(index is None for index in self.indices.values()):
                    logger.info("One or more indices do not exist, creating nodes and building indices")
                    
                    # Create nodes if not already done
                    if not self.nodes:
                        self.create_nodes()
                    
                    # Build indices
                    self.indices = self.index_builder.build_indices(self.nodes)
            except Exception as e:
                logger.warning(f"Failed to load indices: {str(e)}")
                logger.info("Creating nodes and building indices")
                
                # Create nodes if not already done
                if not self.nodes:
                    self.create_nodes()
                
                # Build indices
                self.indices = self.index_builder.build_indices(self.nodes)
        
        return self.indices
    
    def initialize_retriever(self) -> FinancialRetriever:
        """
        Initialize the retriever.
        
        Returns:
            FinancialRetriever object.
        """
        logger.info("Initializing retriever")
        
        # Ensure indices are built or loaded
        if not self.indices:
            self.build_or_load_indices()
        
        # Create retriever
        self.retriever = FinancialRetriever(
            text_index=self.indices["text_index"],
            table_index=self.indices["table_index"],
            context_index=self.indices["context_index"],
            llm=self.llm,
            similarity_top_k=SIMILARITY_TOP_K,
            rerank_top_n=RERANK_TOP_N
        )
        
        return self.retriever
    
    def initialize_sub_question_engine(self) -> FinancialSubQuestionEngine:
        """
        Initialize the sub-question engine.
        
        Returns:
            FinancialSubQuestionEngine object.
        """
        logger.info("Initializing sub-question engine")
        
        # Ensure indices are built or loaded
        if not self.indices:
            self.build_or_load_indices()
        
        # Create sub-question engine
        self.sub_question_engine = FinancialSubQuestionEngine(
            text_index=self.indices["text_index"],
            table_index=self.indices["table_index"],
            context_index=self.indices["context_index"],
            llm=self.llm
        )
        
        return self.sub_question_engine
    
    def retrieve(self, query: str, use_query_transformation: bool = True) -> Dict[str, Any]:
        """
        Retrieve relevant nodes for a query.
        
        Args:
            query: User query.
            use_query_transformation: Whether to use query transformation.
            
        Returns:
            Dictionary containing retrieval results.
        """
        logger.info(f"Retrieving for query: {query}")
        
        # Ensure retriever is initialized
        if not self.retriever:
            self.initialize_retriever()
        
        # Perform retrieval
        return self.retriever.retrieve(query, use_query_transformation)
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a query with the sub-question engine.
        
        Args:
            query: User query.
            
        Returns:
            Dictionary containing query results.
        """
        logger.info(f"Processing query: {query}")
        
        # Ensure sub-question engine is initialized
        if not self.sub_question_engine:
            self.initialize_sub_question_engine()
        
        # Determine if query potentially needs numerical operations
        # This is a simple heuristic; the actual analysis is done in the sub-question engine
        has_numerical_terms = any(term in query.lower() for term in [
            "calculate", "compute", "sum", "average", "mean", "total", "difference",
            "percent", "percentage", "ratio", "compare", "growth", "increase", "decrease"
        ])
        
        # Process query
        if has_numerical_terms:
            logger.info("Query potentially requires numerical operations")
            return self.sub_question_engine.query_with_numerical_operations(query)
        else:
            return self.sub_question_engine.query(query)
    
    def initialize(self) -> None:
        """
        Initialize the entire RAG system.
        """
        logger.info("Initializing Financial RAG system")
        
        # Load and process data
        self.load_and_process_data()
        
        # Create nodes
        self.create_nodes()
        
        # Build or load indices
        self.build_or_load_indices()
        
        # Initialize retriever
        self.initialize_retriever()
        
        # Initialize sub-question engine
        self.initialize_sub_question_engine()
        
        logger.info("Financial RAG system initialized successfully")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial RAG System")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the financial data")
    parser.add_argument("--persist_dir", type=str, default=PERSIST_DIR, help="Directory to persist the indices")
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL, help="Name of the embedding model to use")
    parser.add_argument("--llm_model", type=str, default=LLM_MODEL, help="Name of the LLM model to use")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature setting for the LLM")
    parser.add_argument("--rebuild_indices", action="store_true", help="Whether to rebuild indices even if they exist")
    parser.add_argument("--query", type=str, help="Query to process")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create RAG system
    rag_system = FinancialRAGSystem(
        data_path=args.data_path,
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        temperature=args.temperature,
        rebuild_indices=args.rebuild_indices
    )
    
    # Initialize system
    rag_system.initialize()
    
    # Process query if provided
    if args.query:
        result = rag_system.query(args.query)
        print("\nQuery:", args.query)
        print("\nResponse:", result["response"])
        
        # Print sub-questions if available
        if result.get("sub_questions"):
            print("\nSub-questions:")
            for i, sub_q in enumerate(result["sub_questions"]):
                print(f"\n{i+1}. {sub_q['question']}")
                print(f"   Answer: {sub_q['response']}")
        
        # Print numerical response if available
        if result.get("numerical_response"):
            print("\nNumerical analysis:", result["numerical_response"])
    else:
        print("System initialized successfully. Use --query to process a query.")

if __name__ == "__main__":
    main() 