from typing import List, Dict, Any, Optional, Union
import logging
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI

from rag_system.utils.query_transformers import QueryTransformer

logger = logging.getLogger(__name__)

class FinancialRetriever:
    """
    Custom retriever for financial data that combines results from multiple indices
    and applies various retrieval enhancement strategies.
    """
    
    def __init__(
        self,
        text_index: VectorStoreIndex,
        table_index: VectorStoreIndex,
        context_index: VectorStoreIndex,
        llm: Optional[OpenAI] = None,
        similarity_top_k: int = 5,
        rerank_top_n: int = 3
    ):
        """
        Initialize the financial retriever.
        
        Args:
            text_index: VectorStoreIndex for text nodes.
            table_index: VectorStoreIndex for table nodes.
            context_index: VectorStoreIndex for all nodes with broader context.
            llm: OpenAI LLM for query transformation.
            similarity_top_k: Number of top results to retrieve from each index.
            rerank_top_n: Number of top results to keep after reranking.
        """
        self.text_index = text_index
        self.table_index = table_index
        self.context_index = context_index
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.rerank_top_n = rerank_top_n
        
        # Initialize retrievers
        self.text_retriever = VectorIndexRetriever(
            index=text_index,
            similarity_top_k=similarity_top_k
        )
        
        self.table_retriever = VectorIndexRetriever(
            index=table_index,
            similarity_top_k=similarity_top_k
        )
        
        self.context_retriever = VectorIndexRetriever(
            index=context_index,
            similarity_top_k=similarity_top_k
        )
        
        # Initialize query transformer if LLM is provided
        self.query_transformer = QueryTransformer(
            model_name=llm.model if llm else "gpt-3.5-turbo",
            temperature=llm.temperature if llm else 0.1
        ) if llm else None
    
    def retrieve(self, query: str, use_query_transformation: bool = True) -> Dict[str, Any]:
        """
        Retrieve relevant nodes for a given query.
        
        Args:
            query: User query.
            use_query_transformation: Whether to use query transformation strategies.
            
        Returns:
            Dictionary containing retrieved nodes and other metadata.
        """
        logger.info(f"Retrieving for query: {query}")
        
        # Store original query
        original_query = query
        
        # Apply query transformation if enabled and available
        transformed_queries = []
        if use_query_transformation and self.query_transformer:
            # Rewrite query for better retrieval
            query = self.query_transformer.rewrite_query(query)
            logger.info(f"Rewritten query: {query}")
            
            # Expand query into multiple alternatives
            expanded_queries = self.query_transformer.expand_query(original_query)
            transformed_queries = expanded_queries
            logger.info(f"Generated {len(expanded_queries)} expanded queries")
        
        # Retrieve from text index
        text_nodes = self.text_retriever.retrieve(query)
        
        # Retrieve from table index
        table_nodes = self.table_retriever.retrieve(query)
        
        # Retrieve from context index
        context_nodes = self.context_retriever.retrieve(query)
        
        # Perform multi-query retrieval if expanded queries are available
        multi_query_nodes = []
        if transformed_queries:
            for expanded_query in transformed_queries:
                # Retrieve from all indices
                additional_text_nodes = self.text_retriever.retrieve(expanded_query)
                additional_table_nodes = self.table_retriever.retrieve(expanded_query)
                
                multi_query_nodes.extend(additional_text_nodes)
                multi_query_nodes.extend(additional_table_nodes)
        
        # Combine all retrieved nodes
        all_nodes = []
        all_nodes.extend(text_nodes)
        all_nodes.extend(table_nodes)
        all_nodes.extend(multi_query_nodes)
        
        # Deduplicate nodes by ID
        unique_nodes_dict = {}
        for node in all_nodes:
            if node.node.node_id not in unique_nodes_dict:
                unique_nodes_dict[node.node.node_id] = node
            else:
                # If duplicate, keep the one with higher similarity score
                if node.score > unique_nodes_dict[node.node.node_id].score:
                    unique_nodes_dict[node.node.node_id] = node
        
        # Convert back to list
        unique_nodes = list(unique_nodes_dict.values())
        
        # Add the broader context nodes
        context_node_ids = set(node.node.node_id for node in unique_nodes)
        additional_context = [
            node for node in context_nodes 
            if node.node.node_id not in context_node_ids
        ]
        
        # Apply similarity-based reranking
        similarity_processor = SimilarityPostprocessor(similarity_cutoff=0.7)
        reranked_nodes = similarity_processor.postprocess_nodes(unique_nodes)
        
        # Take the top N nodes after reranking
        top_nodes = reranked_nodes[:self.rerank_top_n]
        
        # Add up to 2 table nodes if they're not already in top nodes
        top_node_ids = set(node.node.node_id for node in top_nodes)
        table_nodes_to_add = [
            node for node in table_nodes 
            if node.node.node_id not in top_node_ids
        ][:2]
        
        # Combine everything
        final_nodes = top_nodes + table_nodes_to_add + additional_context[:2]
        
        # Return results
        return {
            "original_query": original_query,
            "transformed_query": query if query != original_query else None,
            "expanded_queries": transformed_queries,
            "retrieved_nodes": final_nodes,
            "text_node_count": len(text_nodes),
            "table_node_count": len(table_nodes),
            "context_node_count": len(context_nodes),
            "multi_query_node_count": len(multi_query_nodes),
            "final_node_count": len(final_nodes)
        }
    
    def hybrid_retrieve(self, query: str) -> List[NodeWithScore]:
        """
        Perform hybrid retrieval combining vector search with keyword filtering.
        
        Args:
            query: User query.
            
        Returns:
            List of retrieved nodes with scores.
        """
        # First retrieve with vector search
        vector_results = self.retrieve(query)
        retrieved_nodes = vector_results["retrieved_nodes"]
        
        # Get node IDs from vector retrieval
        retrieved_ids = set(node.node.node_id for node in retrieved_nodes)
        
        # Extract keywords for filtering
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        
        # Filter context nodes that contain at least one keyword but weren't retrieved by vector search
        keyword_nodes = []
        for node in self.context_retriever.retrieve(query, similarity_top_k=20):
            if node.node.node_id not in retrieved_ids:
                node_text = node.node.text.lower()
                if any(keyword in node_text for keyword in keywords):
                    keyword_nodes.append(node)
        
        # Combine results
        combined_nodes = retrieved_nodes + keyword_nodes[:3]
        
        return combined_nodes
    
    def retrieve_with_numerical_focus(self, query: str) -> List[NodeWithScore]:
        """
        Retrieve with a focus on nodes containing numerical data.
        
        Args:
            query: User query.
            
        Returns:
            List of retrieved nodes with scores.
        """
        # First perform standard retrieval
        retrieval_result = self.retrieve(query)
        retrieved_nodes = retrieval_result["retrieved_nodes"]
        
        # Analyze if the query requires numerical data
        if self.query_transformer:
            numerical_analysis = self.query_transformer.analyze_numerical_query(query)
            needs_numerical = numerical_analysis.get("needs_numerical_computation", False)
            
            if needs_numerical:
                # Filter nodes to prioritize those with numerical data
                numerical_nodes = []
                non_numerical_nodes = []
                
                for node in retrieved_nodes:
                    if node.node.metadata.get("contains_numbers", False) or node.node.metadata.get("is_table", False):
                        numerical_nodes.append(node)
                    else:
                        non_numerical_nodes.append(node)
                
                # Ensure we have enough numerical nodes
                if len(numerical_nodes) < 3:
                    # Retrieve more nodes specifically from tables
                    additional_table_nodes = self.table_retriever.retrieve(query, similarity_top_k=10)
                    
                    # Only add tables not already included
                    existing_ids = set(node.node.node_id for node in numerical_nodes)
                    for node in additional_table_nodes:
                        if node.node.node_id not in existing_ids:
                            numerical_nodes.append(node)
                            existing_ids.add(node.node.node_id)
                            if len(numerical_nodes) >= 5:
                                break
                
                # Combine, prioritizing numerical nodes
                return numerical_nodes + non_numerical_nodes
        
        return retrieved_nodes 