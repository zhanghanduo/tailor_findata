from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import re
import json
import colorama
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
import time

# Initialize colorama for colored terminal output
colorama.init()

logger = logging.getLogger(__name__)

# Define colors for terminal output
CYAN = colorama.Fore.CYAN
MAGENTA = colorama.Fore.MAGENTA
YELLOW = colorama.Fore.YELLOW
GREEN = colorama.Fore.GREEN
RESET = colorama.Fore.RESET

class HybridFinancialRetriever(BaseRetriever):
    """
    Hybrid retriever that combines vector search and BM25 for financial data.
    Handles different types of content (pre_text, post_text, table)
    and adapts retrieval strategies based on query analysis.
    """
    
    def __init__(
        self,
        pre_text_index: VectorStoreIndex,
        post_text_index: VectorStoreIndex,
        table_index: VectorStoreIndex,
        llm: Optional[OpenAI] = None,
        similarity_top_k: int = 5,
        rerank_top_n: int = 10,
        vector_weight: float = 0.5,  # Weight for vector search results (0-1)
        bm25_weight: float = 0.5,    # Weight for BM25 search results (0-1)
    ):
        """
        Initialize the hybrid financial retriever.
        
        Args:
            pre_text_index: VectorStoreIndex for pre_text nodes.
            post_text_index: VectorStoreIndex for post_text nodes.
            table_index: VectorStoreIndex for table nodes.
            llm: OpenAI LLM for query analysis.
            similarity_top_k: Number of top results to retrieve.
            rerank_top_n: Number of top results to rerank.
            vector_weight: Weight for vector search (0-1).
            bm25_weight: Weight for BM25 search (0-1).
        """
        super().__init__()
        self.pre_text_index = pre_text_index
        self.post_text_index = post_text_index
        self.table_index = table_index
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.rerank_top_n = rerank_top_n
        
        # Debug logging for index initialization
        logger.info("Initializing HybridFinancialRetriever")
        self._log_index_info(pre_text_index, "pre_text")
        self._log_index_info(post_text_index, "post_text")
        self._log_index_info(table_index, "table")
        
        # Ensure weights sum to 1
        total = vector_weight + bm25_weight
        self.vector_weight = vector_weight / total
        self.bm25_weight = bm25_weight / total
        
        # Create vector retrievers
        self.pre_text_vector_retriever = pre_text_index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        self.post_text_vector_retriever = post_text_index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        self.table_vector_retriever = table_index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        
        # Create BM25 retrievers
        self.pre_text_bm25_retriever = self._create_bm25_retriever(pre_text_index)
        self.post_text_bm25_retriever = self._create_bm25_retriever(post_text_index)
        self.table_bm25_retriever = self._create_bm25_retriever(table_index)
        
        # Configure postprocessors
        self.similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.6)
    
    def _log_index_info(self, index: VectorStoreIndex, name: str):
        """Debug log information about the index"""
        if index is None:
            logger.warning(f"{name} index is None")
            return
            
        # Check collection size
        if hasattr(index, '_vector_store') and hasattr(index._vector_store, '_collection'):
            collection = index._vector_store._collection
            try:
                count = collection.count()
                logger.info(f"{name} collection has {count} documents")
            except Exception as e:
                logger.warning(f"Error getting count from {name} collection: {str(e)}")
        else:
            logger.warning(f"{name} index has no _vector_store._collection attribute")
            
        # Check docstore
        if hasattr(index, '_storage_context') and hasattr(index._storage_context, 'docstore'):
            try:
                docstore = index._storage_context.docstore
                doc_count = len(docstore.docs) if hasattr(docstore, 'docs') else 0
                logger.info(f"{name} docstore has {doc_count} documents")
            except Exception as e:
                logger.warning(f"Error accessing {name} docstore: {str(e)}")
        else:
            logger.warning(f"{name} index has no _storage_context.docstore attribute")
    
    def refresh_collections(self):
        """Force refresh of collections and retrievers."""
        logger.info("Refreshing collections and retrievers")
        
        # Re-create BM25 retrievers
        self.pre_text_bm25_retriever = self._create_bm25_retriever(self.pre_text_index)
        self.post_text_bm25_retriever = self._create_bm25_retriever(self.post_text_index)
        self.table_bm25_retriever = self._create_bm25_retriever(self.table_index)
        
        # Log refresh results
        self._log_index_info(self.pre_text_index, "pre_text (after refresh)")
        self._log_index_info(self.post_text_index, "post_text (after refresh)")
        self._log_index_info(self.table_index, "table (after refresh)")
        
        return "Collections and retrievers refreshed"
    
    def _create_bm25_retriever(self, index: VectorStoreIndex) -> BM25Retriever:
        """
        Create a BM25 retriever for the given index.
        
        Args:
            index: The index to create a BM25 retriever for.
            
        Returns:
            BM25Retriever for the index.
        """
        # Get nodes from the vector store directly first (most reliable for Chroma)
        nodes = []
        
        # For Chroma Vector Store, extract documents directly
        try:
            if hasattr(index, '_vector_store') and hasattr(index._vector_store, '_collection'):
                collection = index._vector_store._collection
                logger.info(f"Accessing Chroma collection with {collection.count()} documents")
                
                # Get all documents from the collection
                results = collection.get(limit=10000)  # Use a high limit to ensure we get everything
                
                if results and 'documents' in results and results['documents']:
                    doc_count = len(results['documents'])
                    logger.info(f"Retrieved {doc_count} documents from Chroma collection")
                    
                    # Create text nodes from the documents
                    for i, doc in enumerate(results['documents']):
                        if i < len(results.get('metadatas', [])):
                            metadata = results['metadatas'][i]
                        else:
                            metadata = {}
                            
                        if i < len(results.get('ids', [])):
                            node_id = results['ids'][i]
                        else:
                            node_id = f"chroma_node_{i}"
                            
                        nodes.append(TextNode(
                            text=doc,
                            id_=node_id,
                            metadata=metadata
                        ))
                        
                    logger.info(f"Created {len(nodes)} TextNodes from Chroma documents")
        except Exception as e:
            logger.warning(f"Error accessing Chroma collection directly: {str(e)}")
            
        # If we couldn't get nodes from Chroma directly, try the docstore
        if not nodes:
            try:
                # Access the docstore
                if hasattr(index, '_storage_context') and index._storage_context is not None:
                    docstore = index._storage_context.docstore
                    if docstore is not None and hasattr(docstore, 'docs'):
                        nodes = list(docstore.docs.values())
                        logger.info(f"Found {len(nodes)} nodes in storage_context docstore")
                
                # Fallback to old method if needed
                if not nodes and hasattr(index, '_docstore') and index._docstore is not None:
                    if hasattr(index._docstore, 'docs'):
                        nodes = list(index._docstore.docs.values())
                        logger.info(f"Found {len(nodes)} nodes in _docstore")
            except Exception as e:
                logger.warning(f"Error accessing docstore: {str(e)}")
                
        # Check if we have valid nodes with content
        valid_nodes = []
        for node in nodes:
            try:
                # Try different ways to get content from node
                content = ""
                if hasattr(node, 'get_content') and callable(getattr(node, 'get_content')):
                    content = node.get_content()
                elif hasattr(node, 'text'):
                    content = node.text
                    
                if content and len(content.strip()) > 0:
                    valid_nodes.append(node)
            except Exception as e:
                logger.warning(f"Error getting content from node: {str(e)}")
                
        # If no valid nodes, create dummy nodes
        if not valid_nodes:
            logger.warning(f"No valid nodes found for BM25 retriever, creating dummy nodes")
            # Create a minimum set of dummy nodes
            min_nodes = max(10, self.similarity_top_k * 2)
            for i in range(min_nodes):
                dummy_node = TextNode(
                    text=f"Placeholder text {i} for BM25 indexing. This dummy node is created because no valid nodes were found in the index.",
                    id_=f"dummy_node_{i}"
                )
                valid_nodes.append(dummy_node)
        else:
            logger.info(f"Using {len(valid_nodes)} valid nodes for BM25 retriever")
            
        # Create BM25Retriever with the valid nodes
        try:
            # Create a SimpleDocumentStore for the BM25 retriever
            from llama_index.core.storage.docstore import SimpleDocumentStore
            docstore = SimpleDocumentStore()
            docstore.add_documents(valid_nodes)
            
            # Initialize the BM25 retriever
            retriever = BM25Retriever.from_defaults(
                docstore=docstore,
                similarity_top_k=min(self.similarity_top_k * 2, len(valid_nodes))
            )
            
            return retriever
        except Exception as e:
            logger.error(f"Failed to create BM25 retriever: {str(e)}")
            # Use a simplified fallback approach
            try:
                # Last resort fallback
                dummy_nodes = [
                    TextNode(text=f"Emergency fallback node {i}", id_=f"emergency_node_{i}")
                    for i in range(self.similarity_top_k * 2)
                ]
                docstore = SimpleDocumentStore()
                docstore.add_documents(dummy_nodes)
                retriever = BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=self.similarity_top_k)
                return retriever
            except Exception as e2:
                logger.error(f"Final fallback BM25 retriever creation failed: {str(e2)}")
                raise
    
    def _is_numerical_query(self, query: str) -> bool:
        """
        Determine if the query is looking for numerical information.
        
        Args:
            query: The query string.
            
        Returns:
            True if the query is numerical, False otherwise.
        """
        numerical_keywords = [
            r"\d+%", r"\d+\.?\d*", "percent", "percentage", "ratio", "growth", "increase", 
            "decrease", "change", "revenue", "profit", "loss", "margin", "expense", "cost", 
            "calculate", "compute", "total", "sum", "average", "mean", "median", "compare",
            # Financial terms that typically involve numbers
            "cash flow", "balance", "statement", "financial", "report", "quarter", "annual",
            "fiscal", "year", "month", "income", "asset", "liability", "equity", "dividend",
            "debt", "interest", "tax", "payment", "earning", "ebitda", "operating", "net",
            # Specific to cash flows
            "operating activities", "investing activities", "financing activities",
            "cash provided", "cash used", "from operations", "in investing", "in financing",
            # Working capital related
            "working capital", "accounts receivable", "inventory", "inventories", "accounts payable",
            "current assets", "current liabilities", "trade receivables", "trade payables",
            "changes in working capital", "increase", "decrease", "reduction", "days"
        ]
        
        for keyword in numerical_keywords:
            if re.search(r"\b" + keyword + r"\b", query, re.IGNORECASE):
                return True
        
        # Use LLM to determine if query is numerical if available
        if self.llm:
            try:
                prompt = f"""
                Analyze if the following query is asking for numerical information, financial calculations, financial statements, or data in tables:
                Query: {query}
                
                Respond with 'yes' if the query is numerical/financial/tabular/statement-related or 'no' if it's more conceptual/textual.
                """
                response = self.llm.complete(prompt)
                if "yes" in response.text.lower():
                    logger.info(f"LLM determined query '{query}' is numerical/financial")
                    return True
            except Exception as e:
                logger.warning(f"Error using LLM for numerical query detection: {str(e)}")
        
        return False
    
    def _is_table_related_query(self, query: str) -> bool:
        """
        Determine if the query is specifically looking for information in tables.
        
        Args:
            query: The query string.
            
        Returns:
            True if the query is table-related, False otherwise.
        """
        table_keywords = [
            "table", "row", "column", "cell", "chart", "graph", "figure", "spreadsheet",
            "matrix", "grid", "tabular", "data", "statement", "report", "balance sheet",
            "income statement", "cash flow statement", "financial statement", "quarterly report",
            "annual report", "10-K", "10-Q", "financial highlights", "summary", "total",
            # Items likely to be in tables
            "operating", "investing", "financing", "activities", "cash provided by",
            "cash used in", "net change", "net increase", "net decrease",
            # Additional financial table keywords
            "earnings", "revenue", "profit", "loss", "margin", "ebitda", "assets", "liabilities",
            "equity", "debt", "ratios", "percentage", "fiscal year", "fiscal quarter",
            # Working capital related
            "working capital", "accounts receivable", "inventory", "accounts payable",
            "current assets", "current liabilities", "changes in", "days sales outstanding",
            "days inventory outstanding", "days payable outstanding", "cash conversion cycle"
        ]
        
        for keyword in table_keywords:
            if re.search(r"\b" + keyword + r"\b", query, re.IGNORECASE):
                return True
        
        # Consider all numerical queries to be at least somewhat table-related
        if self._is_numerical_query(query):
            return True
                
        return False
    
    def _is_cash_flow_year_query(self, query: str) -> bool:
        """
        Detect if a query is asking about cash flow from operating activities for a specific year.
        
        Args:
            query: The query string.
            
        Returns:
            True if the query is about cash flow for a specific year, False otherwise.
        """
        # Check for cash flow terms
        cf_terms = ["cash flow", "operating activities", "net cash", "cash provided", "cash used"]
        has_cf_term = any(term in query.lower() for term in cf_terms) and "operating" in query.lower()
        
        # Check for year mentions
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        
        # Check for question about "what is" or "how much"
        is_inquiry = re.search(r'\b(what|how much|amount)\b', query.lower()) is not None
        
        return has_cf_term and bool(years) and is_inquiry
    
    def _merge_retrieved_nodes(
        self, 
        vector_nodes: List[NodeWithScore], 
        bm25_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """
        Merge vector and BM25 retrieved nodes with a weighted scoring approach.
        
        Args:
            vector_nodes: List of nodes from vector search.
            bm25_nodes: List of nodes from BM25 search.
            
        Returns:
            List of merged nodes with weighted scores.
        """
        all_nodes = {}
        
        # Process vector nodes with weight
        for node in vector_nodes:
            node_id = node.node.node_id
            weighted_score = node.score * self.vector_weight
            all_nodes[node_id] = {
                "node": node.node,
                "vector_score": node.score,
                "bm25_score": 0,
                "final_score": weighted_score
            }
        
        # Process BM25 nodes with weight
        for node in bm25_nodes:
            node_id = node.node.node_id
            weighted_score = node.score * self.bm25_weight
            
            if node_id in all_nodes:
                # Node exists in both results, combine scores
                all_nodes[node_id]["bm25_score"] = node.score
                all_nodes[node_id]["final_score"] += weighted_score
            else:
                # Node only in BM25 results
                all_nodes[node_id] = {
                    "node": node.node,
                    "vector_score": 0,
                    "bm25_score": node.score,
                    "final_score": weighted_score
                }
        
        # Apply metadata-based scoring boosts
        for node_id, node_info in all_nodes.items():
            node = node_info["node"]
            metadata = node.metadata or {}
            
            # Check table-related metadata for boosting
            if metadata.get("type") in ["table", "table_markdown"] or metadata.get("is_table") is True:
                # Basic boost for all tables
                node_info["final_score"] *= 1.1
                
                # Additional boost for tables with financial data
                if any(metadata.get(key) for key in [
                    "contains_net_cash_operating", 
                    "contains_net_income",
                    "contains_revenue",
                    "contains_balance_sheet",
                    "cash_flow_related",
                    "working_capital_related"
                ]):
                    node_info["final_score"] *= 1.25
                
                # Extra boost if years are explicitly mentioned in metadata
                if metadata.get("years_mentioned"):
                    node_info["final_score"] *= 1.15
            
            # Extract organization ID from node ID
            org_id = self._extract_organization_id(node_id)
            if org_id:
                # Store the organization ID in metadata for later use
                node.metadata["organization"] = org_id
        
        # Create combined node list sorted by final score
        result_nodes = [
            NodeWithScore(
                node=info["node"],
                score=info["final_score"]
            )
            for _, info in all_nodes.items()
        ]
        
        return sorted(result_nodes, key=lambda x: x.score, reverse=True)
    
    def _retrieve_from_index_hybrid(
        self,
        query_bundle: QueryBundle,
        vector_retriever: Optional[BaseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        **kwargs  # Add **kwargs to accept and ignore extra parameters
    ) -> List[NodeWithScore]:
        """Retrieve from vector and BM25 indices using a hybrid approach."""
        # Print statement for debugging
        import inspect
        frame = inspect.currentframe()
        args_info = inspect.getargvalues(frame)
        
        # Check if query has an organization identifier
        query_org_id = None
        org_match = re.search(r'\b([A-Z]{2,5})\b', query_bundle.query_str)
        if org_match:
            query_org_id = org_match.group(1)
            logger.info(f"Found organization ID {query_org_id} in query")
        
        # Handle cases where one of the retrievers is None
        results = []
        
        # Get vector search results
        vector_nodes = vector_retriever.retrieve(query_bundle) if vector_retriever else []
        
        # Check if BM25 retriever is properly initialized or marked as invalid
        if (hasattr(bm25_retriever, '_is_invalid') and bm25_retriever._is_invalid) or \
           (not hasattr(bm25_retriever, 'bm25') or bm25_retriever.bm25 is None or \
            not hasattr(bm25_retriever.bm25, 'corpus') or bm25_retriever.bm25.corpus is None):
            logger.warning("BM25 retriever's corpus is None or invalid. Using only vector results.")
            
            # Even with no BM25 results, check for organization matches in vector results
            if query_org_id:
                # Boost scores for matching organization IDs
                for i, node in enumerate(vector_nodes):
                    node_org_id = self._extract_organization_id(node.node.node_id)
                    if node_org_id == query_org_id:
                        logger.info(f"Boosting score for node from organization {node_org_id} matching query")
                        # Boost score by 30%
                        vector_nodes[i] = NodeWithScore(
                            node=node.node,
                            score=node.score * 1.3
                        )
            
            return sorted(vector_nodes, key=lambda x: x.score, reverse=True)
        
        # Check corpus size for BM25 and adjust top_k if necessary
        try:
            corpus_size = len(bm25_retriever.bm25.corpus)
            if corpus_size < bm25_retriever.similarity_top_k:
                # Set a sensible minimum (at least 1)
                adjusted_top_k = max(1, corpus_size)
                logger.warning(f"BM25 corpus size ({corpus_size}) is smaller than similarity_top_k. Adjusting to {adjusted_top_k}.")
                # Temporarily adjust the similarity_top_k
                original_top_k = bm25_retriever.similarity_top_k
                bm25_retriever.similarity_top_k = adjusted_top_k
                
                # Get BM25 search results with adjusted top_k
                try:
                    bm25_nodes = bm25_retriever.retrieve(query_bundle)
                except KeyError as ke:
                    logger.warning(f"KeyError in BM25 retrieval: {str(ke)}. Using only vector results.")
                    
                    # Check for organization matches in vector results
                    if query_org_id:
                        # Boost scores for matching organization IDs
                        for i, node in enumerate(vector_nodes):
                            node_org_id = self._extract_organization_id(node.node.node_id)
                            if node_org_id == query_org_id:
                                logger.info(f"Boosting score for node from organization {node_org_id} matching query")
                                # Boost score by 30%
                                vector_nodes[i] = NodeWithScore(
                                    node=node.node,
                                    score=node.score * 1.3
                                )
                    
                    return sorted(vector_nodes, key=lambda x: x.score, reverse=True)
                
                # Restore original top_k
                bm25_retriever.similarity_top_k = original_top_k
            else:
                # Normal case - corpus is large enough
                try:
                    bm25_nodes = bm25_retriever.retrieve(query_bundle)
                except KeyError as ke:
                    logger.warning(f"KeyError in BM25 retrieval: {str(ke)}. Using only vector results.")
                    
                    # Check for organization matches in vector results
                    if query_org_id:
                        # Boost scores for matching organization IDs
                        for i, node in enumerate(vector_nodes):
                            node_org_id = self._extract_organization_id(node.node.node_id)
                            if node_org_id == query_org_id:
                                logger.info(f"Boosting score for node from organization {node_org_id} matching query")
                                # Boost score by 30%
                                vector_nodes[i] = NodeWithScore(
                                    node=node.node,
                                    score=node.score * 1.3
                                )
                    
                    return sorted(vector_nodes, key=lambda x: x.score, reverse=True)
        except (TypeError, AttributeError, KeyError) as e:
            logger.warning(f"Error getting BM25 results: {str(e)}. Using only vector results.")
            
            # Check for organization matches in vector results
            if query_org_id:
                # Boost scores for matching organization IDs
                for i, node in enumerate(vector_nodes):
                    node_org_id = self._extract_organization_id(node.node.node_id)
                    if node_org_id == query_org_id:
                        logger.info(f"Boosting score for node from organization {node_org_id} matching query")
                        # Boost score by 30%
                        vector_nodes[i] = NodeWithScore(
                            node=node.node,
                            score=node.score * 1.3
                        )
            
            return sorted(vector_nodes, key=lambda x: x.score, reverse=True)
        
        # Merge results with organization boosting
        merged_nodes = self._merge_retrieved_nodes(vector_nodes, bm25_nodes)
        
        # Apply final organization boosting if needed
        if query_org_id:
            for i, node in enumerate(merged_nodes):
                node_org_id = self._extract_organization_id(node.node.node_id)
                if node_org_id == query_org_id:
                    logger.info(f"Applying final boost for node from organization {node_org_id} matching query")
                    # Boost score by 30%
                    merged_nodes[i] = NodeWithScore(
                        node=node.node,
                        score=node.score * 1.3
                    )
            
            # Re-sort after applying boosts
            merged_nodes = sorted(merged_nodes, key=lambda x: x.score, reverse=True)
        
        return merged_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Required implementation of abstract _retrieve method from BaseRetriever.
        This is the core retrieval method that will be called by the base class.
        
        Args:
            query_bundle: QueryBundle containing the query string and any additional info.
            
        Returns:
            List of retrieved nodes with scores.
        """
        # Convert QueryBundle to string for our retrieve method
        query_str = query_bundle.query_str
        
        # Use our main retrieve method with default parameters
        results = self.retrieve(query_str)
        
        # Extract nodes from results and return
        if results and "results" in results:
            nodes = []
            for result in results["results"]:
                # Create NodeWithScore from result
                node = TextNode(
                    text=result["text"],
                    metadata=result["metadata"]
                )
                score = result["metadata"].get("score", 0.0)
                nodes.append(NodeWithScore(node=node, score=score))
            return nodes
        
        # Return empty list if no results
        return []

    def _extract_organization_id(self, node_id: str) -> str:
        """
        Extract organization identifier from node ID.
        
        Args:
            node_id: The node ID string.
            
        Returns:
            Organization identifier or empty string if not found.
        """
        # Extract org ID from format like "Single_GS/2014/page_62.pdf-2" or "Double_AES/2016/page_98.pdf"
        match = re.search(r'(?:Single|Double)_([A-Z]+)/', node_id)
        if match:
            return match.group(1)
        return ""

    def _transform_query(self, query: str) -> str:
        """
        Transform a query to improve retrieval quality.
        
        Args:
            query: The original query.
            
        Returns:
            Transformed query.
        """
        # Extract organization identifier if present in the query
        org_match = re.search(r'\b([A-Z]{2,5})\b', query)
        org_id = ""
        if org_match:
            org_id = org_match.group(1)
            logger.info(f"Detected organization ID {org_id} in query")
        
        # For queries specifically about cash flow (highest priority - handle explicitly)
        if "cash" in query.lower() and "operating" in query.lower():
            # Extract year if present in the query
            year_match = re.search(r"\b(19|20)\d{2}\b", query)
            year_str = ""
            if year_match:
                year = year_match.group(0)
                # Add year explicitly if not already formatted
                if f"in {year}" not in query.lower() and f"for {year}" not in query.lower():
                    year_str = f" in {year}"
                logger.info(f"Detected year {year} in cash flow query")
            
            # Create expanded query with variations of the same concept
            if "net cash from operating activities" in query.lower():
                # Already has the main term but add year if needed
                if year_str:
                    base_query = f"net cash from operating activities{year_str} OR cash provided by operating activities{year_str} OR cash flows from operations{year_str}"
                    # Add organization if detected
                    if org_id:
                        return f"{base_query} {org_id}"
                    return base_query
                return query  # Already specific enough
            elif "cash" in query.lower() and "operating" in query.lower() and "activities" in query.lower():
                base_query = f"net cash from operating activities{year_str} OR cash provided by operating activities{year_str} OR cash flows from operations{year_str}"
                # Add organization if detected
                if org_id:
                    return f"{base_query} {org_id}"
                return base_query
            else:
                # General cash + operating query
                base_query = f"net cash from operating activities{year_str} OR cash provided by operating activities{year_str} OR operating cash flow{year_str} OR cash flows from operations{year_str}"
                # Add organization if detected
                if org_id:
                    return f"{base_query} {org_id}"
                return base_query
        
        # Simple transformation to add financial context if needed
        financial_markers = [
            "cash flow", "balance sheet", "income statement", "financial", 
            "revenue", "profit", "loss", "dividend", "earnings"
        ]
        
        has_financial_marker = any(marker in query.lower() for marker in financial_markers)
        
        if not has_financial_marker:
            # Try to infer more specific financial context with LLM
            if self.llm:
                # Add retry mechanism for resilience
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        prompt = f"""
                        Analyze this query and transform it to better retrieve financial information:
                        "{query}"
                        
                        Return only the transformed query with no additional text or explanation.
                        If this is already clearly a financial query, return it unchanged.
                        """
                        response = self.llm.complete(prompt)
                        transformed = response.text.strip().strip('"')
                        # Only use transformation if it's actually different
                        if transformed and transformed != query:
                            logger.info(f"LLM transformed query: {query} -> {transformed}")
                            return transformed
                        break  # Success, exit retry loop
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"Error using LLM for query transformation (attempt {retry_count}/{max_retries}): {str(e)}")
                        if retry_count < max_retries:
                            # Simple exponential backoff
                            wait_time = 1 * (2 ** (retry_count - 1))
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            logger.warning("Max retries reached for query transformation. Using original query.")
        
        # For queries looking for numerical data
        if re.search(r"\b(?:how much|what is|what was|what were|total|amount)\b", query.lower()):
            year_match = re.search(r"\b(in|for|during|of)?\s*(\d{4})\b", query.lower())
            if year_match:
                year = year_match.group(2)
                # Make year explicit if it's implicit in the query
                base_query = query
                if f"in {year}" not in query.lower() and f"for {year}" not in query.lower():
                    base_query = f"{query} for {year} financial year"
                # Add organization if detected
                if org_id and org_id not in base_query:
                    return f"{base_query} {org_id}"
                return base_query
        
        # For queries about cash flow specifically
        if "cash flow" in query.lower() or "cash flows" in query.lower():
            if "operating" not in query.lower() and "investing" not in query.lower() and "financing" not in query.lower():
                base_query = f"{query} (operating, investing, and financing activities)"
                # Add organization if detected
                if org_id and org_id not in base_query:
                    return f"{base_query} {org_id}"
                return base_query
        
        # For queries about working capital
        if "working capital" in query.lower():
            # If query mentions specific working capital components but not all of them
            wc_components = ["accounts receivable", "inventory", "inventories", "accounts payable"]
            has_component = any(comp in query.lower() for comp in wc_components)
            all_components = all(comp in query.lower() for comp in ["accounts receivable", "inventory", "accounts payable"])
            
            if not has_component:
                base_query = f"{query} (including accounts receivable, inventory, and accounts payable)"
                # Add organization if detected
                if org_id and org_id not in base_query:
                    return f"{base_query} {org_id}"
                return base_query
            elif has_component and not all_components:
                # Expand to include other working capital components
                expanded_components = []
                if "accounts receivable" not in query.lower():
                    expanded_components.append("accounts receivable")
                if "inventory" not in query.lower() and "inventories" not in query.lower():
                    expanded_components.append("inventory")
                if "accounts payable" not in query.lower():
                    expanded_components.append("accounts payable")
                
                if expanded_components:
                    components_str = ", ".join(expanded_components)
                    base_query = f"{query} (also related to {components_str})"
                    # Add organization if detected
                    if org_id and org_id not in base_query:
                        return f"{base_query} {org_id}"
                    return base_query
        
        # Check for individual working capital components without mentioning "working capital"
        if any(term in query.lower() for term in ["accounts receivable", "inventory", "inventories", "accounts payable"]):
            if "working capital" not in query.lower():
                base_query = f"{query} (working capital component)"
                # Add organization if detected
                if org_id and org_id not in base_query:
                    return f"{base_query} {org_id}"
                return base_query
        
        # If we have an organization ID but haven't added it yet, add it
        if org_id and org_id not in query:
            return f"{query} {org_id}"
        
        # Default: return original query
        return query
    
    def _parse_retrieval_results(self, nodes: List[NodeWithScore], query: str) -> Dict[str, Any]:
        """
        Parse retrieval results into a structured response.
        
        Args:
            nodes: List of retrieved nodes with scores.
            query: The original query.
            
        Returns:
            Dictionary containing parsed retrieval results.
        """
        if not nodes:
            return {
                "query": query,
                "results": [],
                "source_types": {},
                "total_results": 0
            }
        
        results = []
        source_types = {
            "pre_text": 0,
            "post_text": 0, 
            "table": 0
        }
        
        # Extract node metadata
        for node in nodes:
            node_type = node.node.metadata.get("type", "unknown")
            if "pre_text" in node_type:
                source_type = "pre_text"
                source_types["pre_text"] += 1
            elif "post_text" in node_type:
                source_type = "post_text"
                source_types["post_text"] += 1
            elif "table" in node_type or node.node.metadata.get("is_table", False):
                source_type = "table"
                source_types["table"] += 1
            else:
                source_type = "unknown"
            
            # Get organization ID
            org_id = self._extract_organization_id(node.node.node_id)
            
            # Collect metadata to include in results
            result_metadata = {
                "id": node.node.node_id,
                "type": source_type,
                "score": node.score,
                "contains_numbers": node.node.metadata.get("contains_numbers", False),
                "financial_keywords": node.node.metadata.get("financial_keywords", ""),
                "organization": org_id
            }
            
            # Add year information if available
            if "years_mentioned" in node.node.metadata:
                result_metadata["years_mentioned"] = node.node.metadata["years_mentioned"]
            
            results.append({
                "text": node.node.text,
                "metadata": result_metadata
            })
        
        return {
            "query": query,
            "results": results,
            "source_types": source_types,
            "total_results": len(results)
        }

    def _retrieve_tables_only(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve only table nodes for queries that specifically need tabular data.
        
        Args:
            query_bundle: Query bundle.
            
        Returns:
            List of retrieved table nodes with scores.
        """
        logger.info(f"Retrieving tables only for query: {query_bundle.query_str}")
        
        # Check for organization ID in query
        query_org_id = None
        org_match = re.search(r'\b([A-Z]{2,5})\b', query_bundle.query_str)
        if org_match:
            query_org_id = org_match.group(1)
            logger.info(f"Found organization ID {query_org_id} in table-only query")
        
        # Get results from table index with hybrid approach
        table_nodes = self._retrieve_from_index_hybrid(
            query_bundle, 
            self.table_vector_retriever,
            self.table_bm25_retriever
        )
        
        # Verify these are actually table nodes and boost their scores
        verified_tables = []
        for node in table_nodes:
            is_table = (
                node.node.metadata.get("type") == "table" or
                "table" in node.node.metadata.get("type", "") or
                node.node.metadata.get("is_table", False)
            )
            
            if is_table:
                # Apply boost to emphasize these are the tables we're looking for
                if node.score is not None:
                    node.score *= 1.5
                
                # Apply organization-specific boost if applicable
                node_org_id = self._extract_organization_id(node.node.node_id)
                if query_org_id and node_org_id == query_org_id:
                    logger.info(f"Boosting table node from matching organization: {node_org_id}")
                    node.score *= 1.5  # Additional 50% boost for organization match
                
                # Store organization ID in metadata
                if "metadata" not in node.node.__dict__:
                    node.node.metadata = {}
                node.node.metadata["organization"] = node_org_id
                
                verified_tables.append(node)
        
        # If we didn't find any tables through standard retrieval, search all docstores
        if not verified_tables:
            logger.warning("No tables found in table index, searching all docstores for tables")
            all_docstores = [
                self.pre_text_index._docstore,
                self.post_text_index._docstore,
                self.table_index._docstore
            ]
            
            # Search for any table nodes across all docstores
            for docstore in all_docstores:
                if not docstore or not hasattr(docstore, 'docs'):
                    continue
                    
                for node_id, node in docstore.docs.items():
                    is_table = (
                        node.metadata.get("type") == "table" or
                        "table" in node.metadata.get("type", "") or
                        node.metadata.get("is_table", False)
                    )
                    
                    if is_table:
                        # Extract organization ID
                        node_org_id = self._extract_organization_id(node_id)
                        
                        # Start with basic score
                        text_match_score = 0
                        
                        # Apply organization match boost
                        if query_org_id and node_org_id == query_org_id:
                            # Give significant boost for organization match
                            text_match_score += 2.0
                            logger.info(f"Found table from matching organization: {node_org_id}")
                        
                        # Do a simple text similarity match
                        if hasattr(node, 'text') and node.text:
                            query_text = query_bundle.query_str.lower()
                            
                            # Count matching words
                            query_words = set(re.findall(r'\b\w+\b', query_text))
                            node_text = node.text.lower()
                            
                            for word in query_words:
                                if word in node_text:
                                    text_match_score += 0.5
                                    # Extra points for exact word match
                                    if re.search(r'\b' + re.escape(word) + r'\b', node_text):
                                        text_match_score += 0.5
                        
                        # Store organization ID in metadata
                        if "metadata" not in node.__dict__:
                            node.metadata = {}
                        node.metadata["organization"] = node_org_id
                        
                        # Include any table with at least some match
                        if text_match_score > 0:
                            verified_tables.append(NodeWithScore(node=node, score=text_match_score))
        
        # Sort by score
        verified_tables.sort(key=lambda x: x.score or 0, reverse=True)
        
        # Apply table-specific reranking to prioritize the most relevant tables
        reranked_tables = self._rerank_for_table_query(verified_tables)
        
        # Log results with organization IDs
        logger.info(f"{MAGENTA}=== Final Selected Table Nodes ==={RESET}")
        for i, node in enumerate(reranked_tables[:5]):  # Log top 5
            try:
                node_org_id = self._extract_organization_id(node.node.node_id)
                logger.info(f"{MAGENTA}[{i+1}] Org={node_org_id}, Score={node.score:.4f}{RESET}")
            except Exception as e:
                logger.warning(f"Error logging table node details: {str(e)}")
        
        return reranked_tables[:self.similarity_top_k]

    def retrieve(self, query: str, use_query_transformation: bool = True, tables_only: bool = False) -> Dict[str, Any]:
        """
        Public method to retrieve nodes for a query.
        
        Args:
            query: User query.
            use_query_transformation: Whether to use query transformation.
            tables_only: If True, only retrieve table nodes.
            
        Returns:
            Dictionary containing retrieval results.
        """
        logger.info(f"Retrieving for query: {query} (tables_only={tables_only})")
        
        # Store the current query for reference in other methods
        self.current_query = query
        
        # Store original vector/bm25 weights to restore later
        original_vector_weight = self.vector_weight
        original_bm25_weight = self.bm25_weight
        
        # Check if query mentions financial metrics
        financial_terms = [
            "net cash", "operating activities", "cash flow", "net income", 
            "revenue", "balance sheet", "working capital", "accounts receivable",
            "inventory", "accounts payable"
        ]
        
        # Check if query has specific keywords or exact phrases that might benefit from BM25
        has_financial_terms = any(term in query.lower() for term in financial_terms)
        
        # Check if the query has year patterns which benefit from BM25
        has_year_pattern = bool(re.search(r'\b(19|20)\d{2}\b', query))
        
        # Check if query has an organization ID
        has_org_id = bool(re.search(r'\b([A-Z]{2,5})\b', query))
        
        # Adjust weights for financial queries with years
        if has_financial_terms and has_year_pattern:
            # Financial queries with specific years benefit from increased BM25 weight
            self.bm25_weight = 0.8
            self.vector_weight = 0.2
            logger.info(f"Adjusted weights for financial/year query: BM25={self.bm25_weight}, Vector={self.vector_weight}")
        elif has_financial_terms:
            # Financial queries without years still benefit from BM25 but less extremely
            self.bm25_weight = 0.7
            self.vector_weight = 0.3
            logger.info(f"Adjusted weights for financial query: BM25={self.bm25_weight}, Vector={self.vector_weight}")
        elif has_year_pattern:
            # Year-related queries benefit from slight BM25 boost
            self.bm25_weight = 0.6
            self.vector_weight = 0.4
            logger.info(f"Adjusted weights for year query: BM25={self.bm25_weight}, Vector={self.vector_weight}")
        elif has_org_id:
            # Organization-specific queries benefit from BM25
            self.bm25_weight = 0.65
            self.vector_weight = 0.35
            logger.info(f"Adjusted weights for organization-specific query: BM25={self.bm25_weight}, Vector={self.vector_weight}")
        
        try:
            # Use specialized financial query transformation if applicable
            if use_query_transformation:
                transformed_query = self._transform_query(query)
                if transformed_query != query:
                    logger.info(f"Transformed query: {transformed_query}")
                    query_bundle = QueryBundle(transformed_query)
                else:
                    query_bundle = QueryBundle(query)
            else:
                query_bundle = QueryBundle(query)
            
            # Check if looking for tables only
            if tables_only:
                retrieval_results = self._retrieve_tables_only(query_bundle)
                logger.info(f"Tables-only retrieval found {len(retrieval_results)} results")
                return self._parse_retrieval_results(retrieval_results, query)
            
            # Retrieve from all indices
            all_nodes = []
            
            # Get pre-text nodes
            pre_text_nodes = self._retrieve_from_index_hybrid(
                query_bundle,
                self.pre_text_vector_retriever,
                self.pre_text_bm25_retriever
            )
            all_nodes.extend(pre_text_nodes)
            
            # Get post-text nodes
            post_text_nodes = self._retrieve_from_index_hybrid(
                query_bundle,
                self.post_text_vector_retriever,
                self.post_text_bm25_retriever
            )
            all_nodes.extend(post_text_nodes)
            
            # Get table nodes
            table_nodes = self._retrieve_from_index_hybrid(
                query_bundle,
                self.table_vector_retriever,
                self.table_bm25_retriever
            )
            all_nodes.extend(table_nodes)
            
            # Remove duplicates
            unique_nodes = {}
            for node in all_nodes:
                node_id = node.node.node_id
                if node_id not in unique_nodes or node.score > unique_nodes[node_id].score:
                    unique_nodes[node_id] = node
            
            # Convert to list
            nodes = list(unique_nodes.values())
            logger.info(f"{CYAN}Combined {len(nodes)} unique nodes{RESET}")
            
            # Apply numerical reranking to prioritize nodes with numbers
            reranked_nodes = self._rerank_for_numerical_query(nodes)
            
            # Limit to top results
            final_nodes = reranked_nodes[:self.similarity_top_k]
            
            # Sort by score
            nodes.sort(key=lambda x: x.score or 0, reverse=True)
            
            # Print node organizations for debugging
            logger.info(f"{CYAN}Top nodes by organization:{RESET}")
            for i, node in enumerate(final_nodes[:5]):
                org_id = self._extract_organization_id(node.node.node_id)
                logger.info(f"{CYAN}Node {i+1}: Org={org_id}, Score={node.score}{RESET}")
            
            # Restore original weights
            self.vector_weight = original_vector_weight
            self.bm25_weight = original_bm25_weight
            
            return self._parse_retrieval_results(final_nodes, query)
        except Exception as e:
            logger.exception(f"Error in hybrid retrieval: {str(e)}")
            
            # Fallback to direct keyword search
            try:
                logger.info("Falling back to direct keyword search")
                nodes = self._direct_keyword_search(query)
                return self._parse_retrieval_results(nodes, query)
            except Exception as e2:
                logger.exception(f"Error in direct keyword search: {str(e2)}")
                
                # Final fallback - return empty results
                logger.warning("All retrieval methods failed, returning empty results")
                return {
                    "query": query,
                    "results": [],
                    "nodes": [],
                    "fallback_used": True
                }

    def _direct_keyword_search(self, query_str: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Perform a direct keyword search across all indices.
        This is a fallback search method that looks for exact matches.
        
        Args:
            query_str: The query string.
            top_k: Number of top results to return.
            
        Returns:
            List of nodes with scores.
        """
        exact_phrases = []
        
        # Extract key financial terms
        financial_terms = [
            "net cash from operating activities",
            "operating activities",
            "cash flow",
            "net income",
            "revenue",
            "balance sheet",
            "working capital"
        ]
        
        # Check for year in query (like 1990, 2010, etc.)
        year_match = re.search(r'\b(19|20)\d{2}\b', query_str)
        target_year = None
        year_str = ""
        
        if year_match:
            target_year = year_match.group(0)
            year_str = f" {target_year}"
            
            # Add phrases with the year
            for term in financial_terms:
                if term.lower() in query_str.lower():
                    exact_phrases.append(f"{term}{year_str}")
                    exact_phrases.append(f"{term} in{year_str}")
                    exact_phrases.append(f"{term} for{year_str}")
        
        # Check for organization ID in query
        org_match = re.search(r'\b([A-Z]{2,5})\b', query_str)
        target_org = None
        
        if org_match:
            target_org = org_match.group(1)
            logger.info(f"Detected organization ID {target_org} in query for direct keyword search")
        
        # Add original query terms to search for
        query_terms = query_str.lower().split()
        for i in range(2, len(query_terms) + 1):
            for j in range(len(query_terms) - i + 1):
                phrase = " ".join(query_terms[j:j+i])
                if len(phrase) > 5 and phrase not in exact_phrases:  # Avoid short phrases
                    exact_phrases.append(phrase)
        
        # Sort exact phrases by length (descending) to prioritize longer matches
        exact_phrases = sorted(exact_phrases, key=len, reverse=True)
        
        all_matched_nodes = []
        
        # Search for exact matches in all indices
        for index, collection_name in [
            (self.pre_text_index, "pre_text"),
            (self.post_text_index, "post_text"),
            (self.table_index, "table")
        ]:
            try:
                # Get all nodes from the index for searching
                nodes = []
                
                # Try to get all documents directly from the vector store
                if hasattr(index, '_vector_store') and hasattr(index._vector_store, '_collection'):
                    collection = index._vector_store._collection
                    results = collection.get(limit=10000)
                    
                    if results and 'documents' in results and results['documents']:
                        for i, doc in enumerate(results['documents']):
                            if i < len(results.get('metadatas', [])):
                                metadata = results['metadatas'][i]
                                node_id = results.get('ids', [])[i] if i < len(results.get('ids', [])) else f"node_{i}"
                                
                                # Create a node from the document
                                nodes.append(TextNode(
                                    text=doc,
                                    metadata=metadata,
                                    id_=node_id
                                ))
                
                if not nodes:
                    # Fallback: Try to use the index's docstore
                    if hasattr(index, 'docstore'):
                        docstore_nodes = index.docstore.docs.values()
                        nodes.extend(docstore_nodes)
                
                logger.info(f"Searching {len(nodes)} nodes in {collection_name} index for direct matches")
                
                for node in nodes:
                    node_text = node.text.lower()
                    metadata = node.metadata or {}
                    node_type = metadata.get("type", "")
                    
                    # Extract organization ID
                    node_org_id = self._extract_organization_id(node.node_id)
                    
                    # Initialize match score
                    match_score = 0
                    matching_phrases = []
                    
                    # Organization match bonus - significant boost if organization matches
                    if target_org and node_org_id == target_org:
                        match_score += 3
                        matching_phrases.append(f"organization match: {node_org_id}")
                    
                    # Check if this is a table with metadata about financial data that matches the query
                    if target_year and (node_type.startswith("table") or metadata.get("is_table")):
                        # Check for financial term metadata matches
                        if (
                            ("net cash from operating activities" in query_str.lower() and metadata.get("contains_net_cash_operating")) or
                            ("net income" in query_str.lower() and metadata.get("contains_net_income")) or
                            ("revenue" in query_str.lower() and metadata.get("contains_revenue")) or
                            (any(term in query_str.lower() for term in ["cash flow", "operating activities"]) and metadata.get("cash_flow_related")) or
                            (any(term in query_str.lower() for term in ["working capital", "accounts receivable", "accounts payable"]) and metadata.get("working_capital_related"))
                        ):
                            # Check if specific year is mentioned in the table
                            years_mentioned = metadata.get("years_mentioned", "")
                            if target_year in years_mentioned:
                                match_score += 5
                                matching_phrases.append(f"metadata match: financial term + year {target_year}")
                    
                    # Check exact phrase matches and their proximity
                    for phrase in exact_phrases:
                        if phrase in node_text:
                            # Score based on phrase length and specificity
                            phrase_score = len(phrase) / 10  # Longer phrases get higher scores
                            match_score += phrase_score
                            matching_phrases.append(phrase)
                            
                            # Give bonus for exact financial term matches
                            if any(term in phrase for term in financial_terms):
                                match_score += 1
                                
                            # Give big bonus if both financial term and year are present
                            if target_year and target_year in node_text and any(term in phrase for term in financial_terms):
                                match_score += 3
                    
                    # Only add nodes with matches
                    if match_score > 0:
                        node.metadata["organization"] = node_org_id
                        all_matched_nodes.append(NodeWithScore(
                            node=node,
                            score=match_score
                        ))
            
            except Exception as e:
                logger.error(f"Error in direct keyword search for {collection_name}: {str(e)}")
                logger.exception(e)
        
        # Sort and limit results
        all_matched_nodes = sorted(all_matched_nodes, key=lambda x: x.score, reverse=True)
        return all_matched_nodes[:top_k]

    def _rerank_for_numerical_query(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Rerank nodes for numerical queries by prioritizing nodes with numerical content.
        
        Args:
            nodes: List of retrieved nodes.
            
        Returns:
            Reranked list of nodes.
        """
        def numerical_density(text):
            # Count numbers in the text
            numbers = re.findall(r'\d+\.?\d*%?', text)
            if not text:
                return 0
            return len(numbers) / len(text) * 1000  # Scale for readability
        
        # Extract organization from query if present
        org_id_match = re.search(r'\b([A-Z]{2,5})\b', self.current_query if hasattr(self, 'current_query') else "")
        query_org_id = org_id_match.group(1) if org_id_match else None
        
        # Calculate numerical density for each node
        for i, node in enumerate(nodes):
            text = node.node.text
            score_boost = numerical_density(text)
            
            # Check if the node is a table
            is_table = (
                node.node.metadata.get("type") == "table"
            )
            
            # Give extra boost to tables
            if is_table:
                score_boost *= 1.5
            
            # Extract organization ID and provide a boost if it matches the query
            node_org_id = self._extract_organization_id(node.node.node_id)
            
            # Store the organization ID in metadata
            if "metadata" not in node.node.__dict__:
                node.node.metadata = {}
            node.node.metadata["organization"] = node_org_id
            
            # Apply organization matching boost
            if query_org_id and node_org_id == query_org_id:
                score_boost *= 2.0  # Double the boost for organization match
            
            # Update score
            new_score = (node.score or 0.0) + (score_boost * 0.1)  # Scale the boost
            nodes[i] = NodeWithScore(node=node.node, score=new_score)
        
        # Sort by new score
        nodes.sort(key=lambda x: x.score or 0, reverse=True)
        return nodes
    
    def _rerank_for_table_query(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Rerank nodes for table-related queries by prioritizing table nodes.
        
        Args:
            nodes: List of retrieved nodes.
            
        Returns:
            Reranked list of nodes.
        """
        # Extract organization from query if present
        org_id_match = re.search(r'\b([A-Z]{2,5})\b', self.current_query if hasattr(self, 'current_query') else "")
        query_org_id = org_id_match.group(1) if org_id_match else None
        
        for i, node in enumerate(nodes):
            # Check if the node is a table
            is_table = (
                node.node.metadata.get("type") == "table" or 
                "table" in node.node.metadata.get("type", "") or
                node.node.metadata.get("is_table", False)
            )
            
            # Initial boost multiplier
            boost_multiplier = 1.5 if is_table else 1.0  # 50% boost for tables
            
            # Extract organization ID
            node_org_id = self._extract_organization_id(node.node.node_id)
            
            # Store the organization ID in metadata
            if "metadata" not in node.node.__dict__:
                node.node.metadata = {}
            node.node.metadata["organization"] = node_org_id
            
            # Apply organization matching boost
            if query_org_id and node_org_id == query_org_id:
                logger.info(f"Boosting table with matching organization: {node_org_id}")
                boost_multiplier *= 1.5  # Additional 50% boost for organization match
            
            # Apply the combined boost
            new_score = (node.score or 0.0) * boost_multiplier
            nodes[i] = NodeWithScore(node=node.node, score=new_score)
        
        # Sort by new score
        nodes.sort(key=lambda x: x.score or 0, reverse=True)
        return nodes

    def retrieve_with_numerical_focus(self, query: str) -> Dict[str, Any]:
        """
        Specialized retrieval method focusing on numerical data in documents.
        Optimized for queries requiring numerical information or calculations.
        
        Args:
            query: The query string.
            
        Returns:
            Dictionary containing retrieval results optimized for numerical data.
        """
        logger.info(f"{CYAN}Retrieving with numerical focus for query: {query}{RESET}")
        
        # Store original vector/bm25 weights to restore later
        original_vector_weight = self.vector_weight
        original_bm25_weight = self.bm25_weight
        
        # For numerical queries, adjust weights to favor BM25 which can better match specific numbers
        self.bm25_weight = 0.7
        self.vector_weight = 0.3
        logger.info(f"{CYAN}Adjusted weights for numerical query: BM25={self.bm25_weight}, Vector={self.vector_weight}{RESET}")
        
        try:
            # Transform query to better capture numerical intent
            transformed_query = self._transform_query(query)
            if transformed_query != query:
                logger.info(f"{CYAN}Transformed numerical query: {transformed_query}{RESET}")
                query_bundle = QueryBundle(transformed_query)
            else:
                query_bundle = QueryBundle(query)
            
            # For numerical queries, prioritize tables first
            logger.info(f"{CYAN}Retrieving table nodes for numerical query{RESET}")
            table_nodes = self._retrieve_from_index_hybrid(
                query_bundle,
                self.table_vector_retriever,
                self.table_bm25_retriever
            )
            logger.info(f"{CYAN}Retrieved {len(table_nodes)} table nodes{RESET}")
            
            # Log retrieved table node IDs
            logger.info(f"{MAGENTA}--- Table Node IDs ---{RESET}")
            for i, node in enumerate(table_nodes[:5]):  # Limit to first 5
                node_id = node.node.node_id if hasattr(node.node, 'node_id') else node.node.id_
                
                # Format the ID to be user-friendly
                if '/' in node_id or '.pdf' in node_id:
                    friendly_id = node_id  # Already in a readable format
                else:
                    friendly_id = node_id.split('-')[0] if '-' in node_id else node_id
                    
                logger.info(f"{MAGENTA}Table {i+1}: ID={friendly_id} | Score={node.score:.4f}{RESET}")
            if len(table_nodes) > 5:
                logger.info(f"{MAGENTA}... and {len(table_nodes) - 5} more table nodes{RESET}")
            
            # Get text nodes as well
            logger.info(f"{CYAN}Retrieving pre-text nodes for numerical query{RESET}")
            pre_text_nodes = self._retrieve_from_index_hybrid(
                query_bundle,
                self.pre_text_vector_retriever,
                self.pre_text_bm25_retriever
            )
            logger.info(f"{CYAN}Retrieved {len(pre_text_nodes)} pre-text nodes{RESET}")
            
            logger.info(f"{CYAN}Retrieving post-text nodes for numerical query{RESET}")
            post_text_nodes = self._retrieve_from_index_hybrid(
                query_bundle,
                self.post_text_vector_retriever,
                self.post_text_bm25_retriever
            )
            logger.info(f"{CYAN}Retrieved {len(post_text_nodes)} post-text nodes{RESET}")
            
            # Combine all nodes
            all_nodes = []
            
            # Add table nodes first with a boosted score
            for node in table_nodes:
                # Boost table node scores for numerical queries
                if node.score is not None:
                    node.score *= 1.5
                all_nodes.append(node)
            
            # Add text nodes
            all_nodes.extend(pre_text_nodes)
            all_nodes.extend(post_text_nodes)
            
            # Remove duplicates
            unique_nodes = {}
            for node in all_nodes:
                node_id = node.node.node_id
                if node_id not in unique_nodes or node.score > unique_nodes[node_id].score:
                    unique_nodes[node_id] = node
            
            # Convert to list
            nodes = list(unique_nodes.values())
            logger.info(f"{CYAN}Combined {len(nodes)} unique nodes{RESET}")
            
            # Apply numerical reranking to prioritize nodes with numbers
            reranked_nodes = self._rerank_for_numerical_query(nodes)
            
            # Limit to top results
            final_nodes = reranked_nodes[:self.similarity_top_k]
            
            # Log final selected nodes
            logger.info(f"{MAGENTA}=== Final Selected Numerical Nodes ==={RESET}")
            for i, node in enumerate(final_nodes):
                try:
                    metadata = node.node.metadata
                    node_type = metadata.get("type", "unknown")
                    years = metadata.get("years_mentioned", "")
                    contains_numbers = metadata.get("contains_numbers", False)
                    
                    # Get the node ID and format it to be user-friendly
                    node_id = node.node.node_id if hasattr(node.node, 'node_id') else node.node.id_
                    
                    # Make ID user-friendly (like "Single_JKHY/2009/page_28.pdf-3")
                    if '/' in node_id or '.pdf' in node_id:
                        # This already looks like a file path ID format
                        friendly_id = node_id
                    else:
                        # Try to extract a more concise ID
                        friendly_id = node_id.split('-')[0] if '-' in node_id else node_id
                    
                    logger.info(f"{MAGENTA}[{i+1}] ID={friendly_id} | Type={node_type} | Score={node.score:.4f} | Years={years} | HasNumbers={contains_numbers}{RESET}")
                except Exception as e:
                    logger.warning(f"Error logging node details: {str(e)}")
            logger.info(f"{MAGENTA}======================================{RESET}")
            
            # Restore original weights
            self.vector_weight = original_vector_weight
            self.bm25_weight = original_bm25_weight
            
            return self._parse_retrieval_results(final_nodes, query)
        except Exception as e:
            logger.exception(f"{YELLOW}Error in numerical retrieval: {str(e)}{RESET}")
            
            # Restore weights before fallback
            self.vector_weight = original_vector_weight
            self.bm25_weight = original_bm25_weight
            
            # Fallback to standard retrieval
            try:
                logger.info(f"{YELLOW}Falling back to standard retrieval{RESET}")
                return self.retrieve(query)
            except Exception as e2:
                logger.exception(f"{YELLOW}Error in fallback retrieval: {str(e2)}{RESET}")
                
                # Final fallback - return empty results
                logger.warning(f"{YELLOW}All retrieval methods failed, returning empty results{RESET}")
                return {
                    "query": query,
                    "results": [],
                    "nodes": [],
                    "fallback_used": True
                } 