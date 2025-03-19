from typing import List, Dict, Any, Optional, Union
import logging
import re
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
        pre_text_index: VectorStoreIndex,
        post_text_index: VectorStoreIndex,
        table_markdown_index: VectorStoreIndex,
        table_data_index: VectorStoreIndex,
        context_index: VectorStoreIndex,
        llm: Optional[OpenAI] = None,
        similarity_top_k: int = 5,
        rerank_top_n: int = 3
    ):
        """
        Initialize the financial retriever.
        
        Args:
            pre_text_index: VectorStoreIndex for pre_text nodes.
            post_text_index: VectorStoreIndex for post_text nodes.
            table_markdown_index: VectorStoreIndex for table_markdown nodes.
            table_data_index: VectorStoreIndex for table_data nodes.
            context_index: VectorStoreIndex for all nodes with broader context.
            llm: OpenAI LLM for query transformation.
            similarity_top_k: Number of top results to retrieve from each index.
            rerank_top_n: Number of top results to keep after reranking.
        """
        self.pre_text_index = pre_text_index
        self.post_text_index = post_text_index
        self.table_markdown_index = table_markdown_index
        self.table_data_index = table_data_index
        self.context_index = context_index
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.rerank_top_n = rerank_top_n
        
        # Initialize retrievers
        self.pre_text_retriever = VectorIndexRetriever(
            index=pre_text_index,
            similarity_top_k=similarity_top_k
        )
        
        self.post_text_retriever = VectorIndexRetriever(
            index=post_text_index,
            similarity_top_k=similarity_top_k
        )
        
        self.table_markdown_retriever = VectorIndexRetriever(
            index=table_markdown_index,
            similarity_top_k=similarity_top_k
        )
        
        self.table_data_retriever = VectorIndexRetriever(
            index=table_data_index,
            similarity_top_k=similarity_top_k
        )
        
        self.context_retriever = VectorIndexRetriever(
            index=context_index,
            similarity_top_k=similarity_top_k
        )
        
        # Initialize query transformer if LLM is provided
        self.query_transformer = QueryTransformer(
            model_name=llm.model if llm else "gpt-4o-mini",
            temperature=llm.temperature if llm else 0.1
        ) if llm else None
    
    def _is_numerical_query(self, query: str) -> bool:
        """
        Determine if a query likely requires numerical computation.
        
        Args:
            query: The user query.
            
        Returns:
            Boolean indicating if the query likely requires numerical computation.
        """
        numerical_terms = [
            "calculate", "compute", "sum", "average", "mean", "median", "total", 
            "difference", "percent", "percentage", "ratio", "compare", "growth", 
            "increase", "decrease", "how much", "how many", "cost", "revenue", 
            "profit", "loss", "income", "expense", "budget", "forecast", "projection",
            "margin", "value", "price", "quantity", "amount"
        ]
        
        return any(term in query.lower() for term in numerical_terms)
    
    def _is_table_related_query(self, query: str) -> bool:
        """
        Determine if a query likely requires table data.
        
        Args:
            query: The user query.
            
        Returns:
            Boolean indicating if the query likely requires table data.
        """
        table_terms = [
            "table", "row", "column", "cell", "grid", "spreadsheet", "matrix",
            "data", "value", "header", "entry", "record"
        ]
        
        return any(term in query.lower() for term in table_terms)
    
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
        
        # Analyze query to determine which indices to prioritize
        is_numerical = self._is_numerical_query(query)
        is_table_related = self._is_table_related_query(query)
        
        # Retrieve from all indices with appropriate weighting
        pre_text_nodes = self.pre_text_retriever.retrieve(query)
        post_text_nodes = self.post_text_retriever.retrieve(query)
        
        # For numerical/table queries, prioritize table data
        if is_numerical or is_table_related:
            table_markdown_nodes = self.table_markdown_retriever.retrieve(query, similarity_top_k=self.similarity_top_k * 2)
            table_data_nodes = self.table_data_retriever.retrieve(query, similarity_top_k=self.similarity_top_k * 2)
        else:
            table_markdown_nodes = self.table_markdown_retriever.retrieve(query)
            table_data_nodes = self.table_data_retriever.retrieve(query)
        
        # Retrieve from context index
        context_nodes = self.context_retriever.retrieve(query)
        
        # Perform multi-query retrieval if expanded queries are available
        multi_query_nodes = []
        if transformed_queries:
            for expanded_query in transformed_queries:
                # For numerical/table queries, focus on tables with expanded queries
                if is_numerical or is_table_related:
                    additional_table_markdown_nodes = self.table_markdown_retriever.retrieve(expanded_query)
                    additional_table_data_nodes = self.table_data_retriever.retrieve(expanded_query)
                    multi_query_nodes.extend(additional_table_markdown_nodes)
                    multi_query_nodes.extend(additional_table_data_nodes)
                else:
                    # For general queries, focus on text
                    additional_pre_text_nodes = self.pre_text_retriever.retrieve(expanded_query)
                    additional_post_text_nodes = self.post_text_retriever.retrieve(expanded_query)
                    multi_query_nodes.extend(additional_pre_text_nodes)
                    multi_query_nodes.extend(additional_post_text_nodes)
        
        # Combine all retrieved nodes
        all_nodes = []
        all_nodes.extend(pre_text_nodes)
        all_nodes.extend(post_text_nodes)
        all_nodes.extend(table_markdown_nodes)
        all_nodes.extend(table_data_nodes)
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
        
        # Adjust reranking strategy based on query type
        if is_numerical or is_table_related:
            # For numerical queries, prioritize table nodes
            table_nodes = [
                node for node in reranked_nodes 
                if node.node.metadata.get("is_table", False)
            ]
            non_table_nodes = [
                node for node in reranked_nodes 
                if not node.node.metadata.get("is_table", False)
            ]
            
            # Ensure we have enough table nodes at the top
            top_table_count = min(self.rerank_top_n, len(table_nodes))
            top_non_table_count = self.rerank_top_n - top_table_count
            
            top_nodes = table_nodes[:top_table_count] + non_table_nodes[:top_non_table_count]
        else:
            # For non-numerical queries, use standard reranking
            top_nodes = reranked_nodes[:self.rerank_top_n]
        
        # Always include at least one table node if available
        top_node_ids = set(node.node.node_id for node in top_nodes)
        table_nodes_to_add = [
            node for node in (table_markdown_nodes + table_data_nodes)
            if node.node.node_id not in top_node_ids
        ][:1]  # Add at most one additional table node
        
        # Combine everything
        final_nodes = top_nodes + table_nodes_to_add + additional_context[:2]
        
        # Return results
        return {
            "original_query": original_query,
            "transformed_query": query if query != original_query else None,
            "expanded_queries": transformed_queries,
            "retrieved_nodes": final_nodes,
            "is_numerical_query": is_numerical,
            "is_table_related_query": is_table_related,
            "pre_text_node_count": len(pre_text_nodes),
            "post_text_node_count": len(post_text_nodes),
            "table_markdown_node_count": len(table_markdown_nodes),
            "table_data_node_count": len(table_data_nodes),
            "context_node_count": len(context_nodes),
            "multi_query_node_count": len(multi_query_nodes),
            "final_node_count": len(final_nodes)
        }
    
    def retrieve_with_table_focus(self, query: str) -> List[NodeWithScore]:
        """
        Retrieve with a specific focus on table data, both in markdown and JSON formats.
        
        Args:
            query: User query.
            
        Returns:
            List of retrieved nodes with scores.
        """
        # Retrieve from table indices with higher similarity_top_k
        table_markdown_nodes = self.table_markdown_retriever.retrieve(query, similarity_top_k=10)
        table_data_nodes = self.table_data_retriever.retrieve(query, similarity_top_k=10)
        
        # Retrieve minimal text context
        pre_text_nodes = self.pre_text_retriever.retrieve(query, similarity_top_k=3)
        post_text_nodes = self.post_text_retriever.retrieve(query, similarity_top_k=3)
        
        # Combine with priority on tables
        combined_nodes = table_markdown_nodes + table_data_nodes + pre_text_nodes + post_text_nodes
        
        # Deduplicate
        unique_nodes_dict = {}
        for node in combined_nodes:
            if node.node.node_id not in unique_nodes_dict:
                unique_nodes_dict[node.node.node_id] = node
            else:
                if node.score > unique_nodes_dict[node.node.node_id].score:
                    unique_nodes_dict[node.node.node_id] = node
        
        return list(unique_nodes_dict.values())
    
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
        
        # Check if query is already identified as numerical
        is_numerical = retrieval_result.get("is_numerical_query", False)
        
        if is_numerical:
            # Filter nodes to prioritize those with numerical data
            numerical_nodes = []
            non_numerical_nodes = []
            
            for node in retrieved_nodes:
                if (node.node.metadata.get("contains_numbers", False) or 
                    node.node.metadata.get("is_table", False)):
                    numerical_nodes.append(node)
                else:
                    non_numerical_nodes.append(node)
            
            # Ensure we have enough numerical nodes by retrieving more if needed
            if len(numerical_nodes) < 3:
                # Retrieve more nodes from table indices
                additional_table_markdown_nodes = self.table_markdown_retriever.retrieve(query, similarity_top_k=10)
                additional_table_data_nodes = self.table_data_retriever.retrieve(query, similarity_top_k=10)
                
                # Only add tables not already included
                existing_ids = set(node.node.node_id for node in numerical_nodes)
                
                for node in additional_table_markdown_nodes + additional_table_data_nodes:
                    if node.node.node_id not in existing_ids:
                        numerical_nodes.append(node)
                        existing_ids.add(node.node.node_id)
                        if len(numerical_nodes) >= 5:
                            break
            
            # Combine, prioritizing numerical nodes
            return numerical_nodes + non_numerical_nodes[:5]
        
        return retrieved_nodes
    
    def retrieve_document_context(self, node_id: str) -> str:
        """
        Retrieve the full document context for a node.
        
        Args:
            node_id: ID of the node to get context for.
            
        Returns:
            Full content of the document containing the node.
        """
        # Extract base document ID from node ID
        doc_id_pattern = r'(.+?)(?:_pre_text|_table_markdown|_table_data|_post_text|_chunk_).*'
        match = re.match(doc_id_pattern, node_id)
        
        if not match:
            logger.warning(f"Could not extract base document ID from node ID: {node_id}")
            return ""
        
        base_doc_id = match.group(1)
        
        # Search for the document in the context index
        results = self.context_retriever.retrieve(base_doc_id, similarity_top_k=20)
        
        for node in results:
            if "full_content" in node.node.metadata:
                return node.node.metadata["full_content"]
        
        # If full_content not found, try to reconstruct from related nodes
        logger.info(f"Full content not found for document ID: {base_doc_id}. Attempting to reconstruct.")
        
        # Retrieve all nodes for this document ID
        all_related_nodes = []
        for retriever in [self.pre_text_retriever, self.post_text_retriever, 
                         self.table_markdown_retriever, self.table_data_retriever]:
            all_related_nodes.extend(retriever.retrieve(base_doc_id, similarity_top_k=20))
        
        # Extract content from related nodes
        pre_text = ""
        table_markdown = ""
        post_text = ""
        
        for node in all_related_nodes:
            if "pre_text" in node.node.id_:
                pre_text = node.node.text
            elif "table_markdown" in node.node.id_:
                table_markdown = node.node.text
            elif "post_text" in node.node.id_:
                post_text = node.node.text
        
        # Combine content
        full_content = f"{pre_text}\n\n{table_markdown}\n\n{post_text}".strip()
        return full_content if full_content else "Context not found" 