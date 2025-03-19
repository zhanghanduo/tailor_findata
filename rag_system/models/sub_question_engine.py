from typing import List, Dict, Any, Optional, Union
import logging
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore, QueryBundle
import json
import re
import colorama

from rag_system.utils.query_transformers import QueryTransformer
from rag_system.utils.numerical_operations import extract_and_compute_financial_data
from rag_system.models.hybrid_retriever import HybridFinancialRetriever

# Initialize colorama for colored terminal output
colorama.init()

logger = logging.getLogger(__name__)

# Add color constants for terminal output
BLUE = colorama.Fore.BLUE
GREEN = colorama.Fore.GREEN
YELLOW = colorama.Fore.YELLOW
RED = colorama.Fore.RED
RESET = colorama.Fore.RESET

class FinancialSubQuestionEngine:
    """
    SubQuestionQueryEngine for handling complex financial queries by breaking them down
    into simpler sub-questions.
    """
    
    def __init__(
        self,
        pre_text_index: VectorStoreIndex,
        post_text_index: VectorStoreIndex,
        table_index: VectorStoreIndex,
        llm: Optional[OpenAI] = None,
        vector_weight: float = 0.6,  # Weight for vector search
        bm25_weight: float = 0.4     # Weight for BM25 search
    ):
        """
        Initialize the financial sub-question engine.
        
        Args:
            pre_text_index: VectorStoreIndex for pre_text nodes.
            post_text_index: VectorStoreIndex for post_text nodes.
            table_index: VectorStoreIndex for table nodes.
            llm: OpenAI LLM for query transformation and answering.
            vector_weight: Weight for vector search (0-1).
            bm25_weight: Weight for BM25 search (0-1).
        """
        self.pre_text_index = pre_text_index
        self.post_text_index = post_text_index
        self.table_index = table_index
        self.llm = llm
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Initialize query transformer
        if llm:
            if hasattr(llm, 'model_name'):
                model_name = llm.model_name
            elif hasattr(llm, 'model'):
                model_name = llm.model
            else:
                model_name = "gpt-4o-mini"
            
            temperature = llm.temperature if hasattr(llm, 'temperature') else 0.1
        else:
            model_name = "gpt-4o-mini"
            temperature = 0.1
            
        self.query_transformer = QueryTransformer(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize hybrid retriever
        self.hybrid_retriever = HybridFinancialRetriever(
            pre_text_index=pre_text_index,
            post_text_index=post_text_index,
            table_index=table_index,
            llm=llm,
            similarity_top_k=5,
            rerank_top_n=10,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        
        # Create query engines for each index using the hybrid retriever
        self.pre_text_engine = self._create_query_engine(pre_text_index)
        self.post_text_engine = self._create_query_engine(post_text_index)
        self.table_engine = self._create_query_engine(table_index)
        
        # Create tools for sub-question engine
        self.tools = self._create_tools()
        
        # Create sub-question engine
        self.sub_question_engine = self._create_sub_question_engine()
    
    def _create_query_engine(self, index: VectorStoreIndex):
        """
        Create a query engine for an index using the hybrid retriever.
        
        Args:
            index: The index to create a query engine for.
            
        Returns:
            A query engine for the index.
        """
        return index.as_query_engine(
            similarity_top_k=5,
            llm=self.llm
        )
    
    def _create_tools(self) -> List[QueryEngineTool]:
        """
        Create tools for the sub-question query engine.
        
        Returns:
            List of QueryEngineTool objects.
        """
        # Create tools for each query engine
        pre_text_tool = QueryEngineTool.from_defaults(
            query_engine=self.pre_text_engine,
            name="pre_text",
            description=(
                "Useful for answering conceptual questions about financial information, "
                "including company background, industry trends, and qualitative analysis."
            )
        )
        
        post_text_tool = QueryEngineTool.from_defaults(
            query_engine=self.post_text_engine,
            name="post_text",
            description=(
                "Useful for answering questions about financial analysis, interpretations, "
                "and explanations of data in financial reports."
            )
        )
        
        table_tool = QueryEngineTool.from_defaults(
            query_engine=self.table_engine,
            name="tables",
            description=(
                "Useful for answering questions about numerical data, financial metrics, "
                "statistics, and tabular information in financial reports."
            )
        )
        
        return [pre_text_tool, post_text_tool, table_tool]
    
    def _create_sub_question_engine(self) -> SubQuestionQueryEngine:
        """
        Create the sub-question query engine.
        
        Returns:
            SubQuestionQueryEngine object.
        """
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            llm=self.llm
        )
        
        # Create sub-question engine
        return SubQuestionQueryEngine.from_defaults(
            query_engine_tools=self.tools,
            llm=self.llm,
            response_synthesizer=response_synthesizer,
            use_async=True,
            verbose=True
        )
    
    def _retrieve_nodes(self, query: str, source_type: str = None, use_query_transformation: bool = True) -> List[NodeWithScore]:
        """
        Retrieve nodes for a query, optionally filtered by source type.
        
        Args:
            query: The query string.
            source_type: Optional source type to filter results by.
            use_query_transformation: Whether to use query transformation.
            
        Returns:
            List of retrieved nodes.
        """
        # Get all retrieval results
        retrieval_results = self.hybrid_retriever.retrieve(query, use_query_transformation)
        
        # Extract nodes from the retrieval results
        if not retrieval_results or not retrieval_results.get("results"):
            logger.warning(f"No results found for query: {query}")
            return []
        
        # Convert the retrieval results back to NodeWithScore objects
        nodes = []
        
        # Log retrieval results with document IDs in blue
        logger.info(f"{BLUE}=============== RETRIEVED DOCUMENT IDS ==============={RESET}")
        logger.info(f"{BLUE}Query: {query}{RESET}")
        if source_type:
            logger.info(f"{BLUE}Source type: {source_type}{RESET}")
        
        for result in retrieval_results["results"]:
            # Skip if filtering by source type and type doesn't match
            if source_type and result["metadata"]["type"] != source_type:
                continue
                
            # Reconstruct a NodeWithScore object (best effort)
            try:
                from llama_index.core.schema import TextNode
                node = TextNode(
                    text=result["text"],
                    id_=result["metadata"]["id"],
                    metadata=result["metadata"]
                )
                score = result["metadata"].get("score", 0.0)
                nodes.append(NodeWithScore(node=node, score=score))
                
                # Log document ID and score with colored output - focus on user-friendly ID format
                doc_id = result["metadata"]["id"]
                # Extract the user-friendly part of the ID if it exists
                if '/' in doc_id or '.pdf' in doc_id:
                    # This looks like a file path ID format
                    doc_id_short = doc_id
                else:
                    doc_id_short = doc_id.split('-')[0] if '-' in doc_id else doc_id
                    
                doc_type = result["metadata"]["type"]
                logger.info(f"{BLUE}ID: {doc_id_short} | Type: {doc_type} | Score: {score:.4f}{RESET}")
            except Exception as e:
                logger.warning(f"Error reconstructing node: {str(e)}")
        
        logger.info(f"{BLUE}==================================================={RESET}")
        
        return nodes
    
    def _process_sub_question(self, question: str, source_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a sub-question, retrieving relevant nodes and generating a response.
        
        Args:
            question: The sub-question to process.
            source_type: Optional source type to filter retrieval by.
            
        Returns:
            Dictionary containing the sub-question response.
        """
        logger.info(f"Processing sub-question: {question}")
        
        # Retrieve relevant nodes for the sub-question
        nodes = self._retrieve_nodes(question, source_type, use_query_transformation=True)
        
        if not nodes:
            logger.warning(f"No nodes retrieved for sub-question: {question}")
            return {
                "question": question,
                "source_type": source_type,
                "response": "Empty Response",
                "nodes": []
            }
        
        # Extract texts from nodes with error handling
        texts = []
        for node in nodes:
            try:
                if hasattr(node, 'node') and hasattr(node.node, 'text'):
                    texts.append(node.node.text)
                elif hasattr(node, 'text'):
                    texts.append(node.text)
                elif isinstance(node, str):
                    texts.append(node)
            except Exception as e:
                logger.warning(f"Error extracting text from node: {str(e)}")
        
        # Calculate weighted interpolation of node texts based on scores
        interpolated_text = self._calculate_weighted_text(nodes)
        
        # Use LLM to generate a response from the retrieved texts
        prompt = f"""
        Answer the question "{question}" based on this information:
        
        {interpolated_text}
        
        Answer in a clear, concise way based strictly on the information provided.
        If the information doesn't directly answer the question, say "Cannot determine from the provided information."
        """
        
        response_text = "Empty Response"
        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
        
        # Return the sub-question, response, and any relevant nodes with error handling
        node_info = []
        for node in nodes[:3]:  # Include top 3 nodes
            try:
                if hasattr(node, 'node') and hasattr(node.node, 'text'):
                    node_text = node.node.text
                    node_score = node.score if hasattr(node, 'score') else 0
                elif hasattr(node, 'text'):
                    node_text = node.text
                    node_score = node.score if hasattr(node, 'score') else 0
                elif isinstance(node, str):
                    node_text = node
                    node_score = 0
                else:
                    continue
                    
                node_info.append({"text": node_text, "score": node_score})
            except Exception as e:
                logger.warning(f"Error processing node for response: {str(e)}")
                
        return {
            "question": question,
            "source_type": source_type,
            "response": response_text,
            "nodes": node_info
        }
    
    def _calculate_weighted_text(self, nodes: List[NodeWithScore], max_length: int = 4000) -> str:
        """
        Calculate a weighted text from nodes based on their scores.
        
        Args:
            nodes: List of nodes with scores.
            max_length: Maximum length of the resulting text.
            
        Returns:
            Weighted interpolated text.
        """
        if not nodes:
            return ""
        
        # Sort nodes by score with error handling
        def get_node_score(node):
            try:
                if hasattr(node, 'score'):
                    return node.score or 0
                return 0
            except:
                return 0
                
        sorted_nodes = sorted(nodes, key=get_node_score, reverse=True)
        
        # Take top nodes up to max_length
        text_chunks = []
        current_length = 0
        
        for node in sorted_nodes:
            try:
                # Extract text with robust error handling
                text = ""
                if hasattr(node, 'node') and hasattr(node.node, 'text'):
                    text = node.node.text
                elif hasattr(node, 'text'):
                    text = node.text
                elif isinstance(node, str):
                    text = node
                    
                if not text:
                    continue
                    
                if current_length + len(text) > max_length:
                    # Truncate if needed
                    remaining = max_length - current_length
                    if remaining > 100:  # Only add if we can add a meaningful chunk
                        text_chunks.append(text[:remaining])
                    break
                
                text_chunks.append(text)
                current_length += len(text)
            except Exception as e:
                logger.warning(f"Error processing node in weighted text calculation: {str(e)}")
                continue
        
        return "\n\n".join(text_chunks)
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a complex query by breaking it into sub-questions.
        
        Args:
            query: The complex query to process.
            
        Returns:
            Dictionary containing the response to the complex query.
        """
        logger.info(f"Processing complex query: {query}")
        
        # Analyze query to determine if it needs numerical operations
        needs_computation = self._needs_numerical_computation(query)
        logger.info(f"Query needs numerical computation: {needs_computation}")
        
        if needs_computation:
            return self.query_with_numerical_operations(query)
        
        # Check if query is simple enough to answer directly
        if self._is_simple_query(query):
            return self._process_direct_query(query)
        
        # Generate sub-questions
        sub_questions = self._generate_sub_questions(query)
        logger.info(f"Generated {len(sub_questions)} sub-questions")
        
        # Process each sub-question
        sub_question_responses = []
        for sub_q in sub_questions:
            question = sub_q["question"]
            source_type = sub_q.get("source_type")
            
            response = self._process_sub_question(question, source_type)
            sub_question_responses.append(response)
        
        # Generate final response from sub-question responses
        final_response = self._generate_final_response(query, sub_question_responses)
        
        # Return the complete response
        return {
            "query": query,
            "sub_questions": sub_question_responses,
            "response": final_response
        }
    
    def _is_simple_query(self, query: str) -> bool:
        """
        Determine if a query is simple enough to be answered directly.
        
        Args:
            query: The query to analyze.
            
        Returns:
            True if the query is simple, False otherwise.
        """
        # Check query complexity by looking for certain patterns
        complex_indicators = [
            "compare", "relationship", "difference between", 
            "how does", "explain", "analyze", "what factors", 
            "why did", "how would", "what are the implications"
        ]
        
        # Check if query contains any complex indicators
        contains_complex_indicator = any(indicator in query.lower() for indicator in complex_indicators)
        
        # Simple queries tend to be shorter
        is_short = len(query.split()) <= 15
        
        # Check if query asks for specific data that doesn't need computation
        simple_data_requests = [
            "what is the", "what was the", "how much", "how many",
            "when did", "where is", "who is", "which"
        ]
        asks_for_specific_data = any(request in query.lower() for request in simple_data_requests)
        
        # Simple queries typically ask for specific data points and are relatively short
        # without containing complex indicators
        return (asks_for_specific_data or is_short) and not contains_complex_indicator
    
    def _process_direct_query(self, query: str) -> Dict[str, Any]:
        """
        Process a simple query directly without breaking it into sub-questions.
        
        Args:
            query: The simple query to process.
            
        Returns:
            Dictionary containing the response to the simple query.
        """
        logger.info(f"Processing simple query directly: {query}")
        
        # Retrieve nodes for the direct query
        nodes = self._retrieve_nodes(query)
        
        # Get source text from nodes
        source_text = "\n\n".join([node.node.text for node in nodes[:5]])
        
        # Generate response
        prompt = f"""
        Answer the following question based only on the information provided:
        
        Question: {query}
        
        Information:
        {source_text}
        
        Instructions:
        1. Answer the question directly and concisely.
        2. If the answer is a specific number, percentage, or financial value, include it in <answer></answer> tags.
        3. If you cannot find the answer in the provided information, say so.
        4. Don't use phrases like "Based on the provided information" or "According to the text".
        5. For percentages, round to one decimal place (e.g., 14.1%, not 14.13%).
        """
        
        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
        except Exception as e:
            logger.error(f"Error generating response for direct query: {str(e)}")
            response_text = "Error processing query."
        
        return {
            "query": query,
            "sub_questions": [],
            "response": response_text
        }
    
    def _needs_numerical_computation(self, query: str) -> bool:
        """
        Determine if a query needs numerical computation.
        
        Args:
            query: The query to analyze.
            
        Returns:
            True if the query needs numerical computation, False otherwise.
        """
        analysis = self.query_transformer.analyze_numerical_query(query)
        return analysis.get("needs_computation", False)
    
    def _generate_sub_questions(self, query: str) -> List[Dict[str, Any]]:
        """
        Generate sub-questions from a complex query.
        
        Args:
            query: The complex query.
            
        Returns:
            List of sub-questions with source types.
        """
        # Use QueryTransformer to generate sub-questions
        sub_q_list = self.query_transformer.generate_sub_questions(query)
        
        # Convert simple strings to dictionaries with source types
        sub_questions = []
        for q in sub_q_list:
            # Detect if this is likely table-related
            if any(keyword in q.lower() for keyword in ["table", "row", "column", "statement", "report", "figure"]):
                source_type = "table"
            # Detect if this is likely numerical
            elif any(keyword in q.lower() for keyword in ["how much", "amount", "total", "percentage", "number", "value"]):
                source_type = "table"  # Numerical questions often need table data
            else:
                source_type = None  # Will search all sources
                
            sub_questions.append({"question": q, "source_type": source_type})
        
        # If no sub-questions were generated, create a default one
        if not sub_questions:
            logger.warning(f"No sub-questions generated for query: {query}")
            sub_questions = [{"question": query, "source_type": None}]
        
        return sub_questions
    
    def _generate_final_response(self, query: str, sub_question_responses: List[Dict[str, Any]]) -> str:
        """
        Generate a final response to the main query based on sub-question responses.
        
        Args:
            query: The main query.
            sub_question_responses: List of responses to sub-questions.
            
        Returns:
            Final response to the main query.
        """
        logger.info(f"Generating final response for: {query}")
        
        # Extract successful responses (non-empty)
        valid_responses = []
        for response in sub_question_responses:
            if response["response"] and response["response"] != "Empty Response" and response["response"] != "Cannot determine from the provided information.":
                valid_responses.append(response)
        
        if not valid_responses:
            return f"Unable to find sufficient information to answer the query: {query}"
        
        # Create a prompt with the sub-question responses
        prompt = f"""
        Answer the question "{query}" based on the following information from sub-questions:
        
        """
        
        for i, response in enumerate(valid_responses):
            prompt += f"Sub-question {i+1}: {response['question']}\n"
            prompt += f"Answer: {response['response']}\n\n"
        
        prompt += """
        Using the above information, provide a comprehensive answer to the main question.
        If the information is insufficient to provide a definitive answer, clearly state what can be determined and what cannot.
        
        If the answer involves a specific numerical value (especially a percentage or financial figure):
        1. Extract the final value and place it within <answer></answer> tags
        2. For percentages, round to one decimal place (e.g., 14.1%, not 14.13%)
        3. Include the '%' symbol for percentage values
        4. For currency values, include the appropriate symbol or code
        
        Example:
        If the question asks for a percentage increase and the calculated value is 14.13%, your answer should include <answer>14.1%</answer>
        """
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            # Fallback to a simpler approach
            return "\n".join([f"Based on sub-question '{r['question']}': {r['response']}" for r in valid_responses])
    
    def query_with_numerical_operations(self, query: str) -> Dict[str, Any]:
        """
        Process a numerical query that requires numerical operations on financial data.
        
        Args:
            query: The user's query.
            
        Returns:
            Dictionary containing the response to the query.
        """
        logger.info(f"Processing numerical query: {query}")
        
        # First, generate regular sub-questions to retrieve relevant data
        sub_questions = self._generate_sub_questions(query)
        
        # Process each sub-question with a focus on numerical data
        sub_question_responses = []
        retrieved_nodes = []
        
        logger.info(f"{YELLOW}=============== NUMERICAL RETRIEVAL ==============={RESET}")
        logger.info(f"{YELLOW}Main query: {query}{RESET}")
        
        for sub_q in sub_questions:
            question = sub_q["question"]
            source_type = sub_q.get("source_type")
            
            logger.info(f"{YELLOW}Processing sub-question: {question} (source: {source_type or 'all'}){RESET}")
            
            # For table data, use specialized retrieval
            if source_type == "table":
                retrieval_result = self.hybrid_retriever.retrieve_with_numerical_focus(question)
                # Convert the retrieval results to NodeWithScore objects
                table_nodes = []
                
                # Log retrieved table documents
                if retrieval_result and "results" in retrieval_result:
                    logger.info(f"{GREEN}--- Retrieved Table Documents ---{RESET}")
                    for i, result in enumerate(retrieval_result["results"]):
                        try:
                            doc_id = result["metadata"]["id"]
                            score = result["metadata"].get("score", 0.0)
                            
                            # Make ID user-friendly
                            if '/' in doc_id or '.pdf' in doc_id:
                                friendly_id = doc_id  # Already in a readable format
                            else:
                                friendly_id = doc_id.split('-')[0] if '-' in doc_id else doc_id
                                
                            logger.info(f"{GREEN}Table {i+1}: ID={friendly_id} | Score={score:.4f}{RESET}")
                            
                            from llama_index.core.schema import TextNode
                            node = TextNode(
                                text=result["text"],
                                id_=doc_id,
                                metadata=result["metadata"]
                            )
                            table_nodes.append(NodeWithScore(node=node, score=score))
                        except Exception as e:
                            logger.warning(f"Error reconstructing node from table result: {str(e)}")
                            
                retrieved_nodes.extend(table_nodes)
                logger.info(f"{GREEN}Retrieved {len(table_nodes)} table nodes{RESET}")
            else:
                nodes = self._retrieve_nodes(question, source_type)
                retrieved_nodes.extend(nodes)
                logger.info(f"{GREEN}Retrieved {len(nodes)} text nodes{RESET}")
            
            # Get a response for the sub-question
            response = self._process_sub_question(question, source_type)
            sub_question_responses.append(response)
        
        logger.info(f"{YELLOW}Total nodes for numerical operations: {len(retrieved_nodes)}{RESET}")
        
        # Log document IDs that will be used for numerical operations
        logger.info(f"{YELLOW}Documents used for numerical operations:{RESET}")
        for i, node in enumerate(retrieved_nodes[:10]):  # Limit to first 10 to avoid clutter
            try:
                # Get the node ID
                if hasattr(node, 'node') and hasattr(node.node, 'node_id'):
                    doc_id = node.node.node_id
                elif hasattr(node, 'node') and hasattr(node.node, 'id_'):
                    doc_id = node.node.id_
                elif hasattr(node, 'node_id'):
                    doc_id = node.node_id
                elif hasattr(node, 'id_'):
                    doc_id = node.id_
                else:
                    doc_id = f"unknown-{i}"
                
                # Make ID user-friendly (like "Single_JKHY/2009/page_28.pdf-3")
                if '/' in doc_id or '.pdf' in doc_id:
                    # This already looks like a file path ID format
                    friendly_id = doc_id
                else:
                    # Try to extract a more concise ID
                    friendly_id = doc_id.split('-')[0] if '-' in doc_id else doc_id
                
                logger.info(f"{YELLOW}[{i+1}] ID: {friendly_id}{RESET}")
            except Exception as e:
                logger.warning(f"Error getting node ID: {str(e)}")
        
        if len(retrieved_nodes) > 10:
            logger.info(f"{YELLOW}... and {len(retrieved_nodes) - 10} more{RESET}")
            
        logger.info(f"{YELLOW}==================================================={RESET}")
        
        # Perform numerical operations on the retrieved data
        numerical_result = extract_and_compute_financial_data(query, retrieved_nodes, self.llm)
        
        # Generate final response incorporating numerical results
        if numerical_result and "result" in numerical_result:
            numerical_response = numerical_result["result"]
            logger.info(f"{GREEN}Successfully performed numerical operations{RESET}")
        else:
            numerical_response = "Could not perform numerical operations with the available data."
            logger.warning(f"{RED}Failed to perform numerical operations{RESET}")
        
        # Generate a text response that includes the numerical results
        final_response = self._generate_final_response_with_numerical(query, sub_question_responses, numerical_response)
        
        # Return the complete response
        return {
            "query": query,
            "sub_questions": sub_question_responses,
            "numerical_response": numerical_response,
            "response": final_response
        }
    
    def _generate_final_response_with_numerical(
        self, 
        query: str, 
        sub_question_responses: List[Dict[str, Any]], 
        numerical_response: str
    ) -> str:
        """
        Generate a final response to the main query based on sub-question responses and numerical analysis.
        
        Args:
            query: The main query.
            sub_question_responses: List of responses to sub-questions.
            numerical_response: Response from numerical operations.
            
        Returns:
            Final response to the main query.
        """
        logger.info(f"Generating final response with numerical results for: {query}")
        
        # Extract successful responses (non-empty)
        valid_responses = []
        for response in sub_question_responses:
            if response["response"] and response["response"] != "Empty Response" and response["response"] != "Cannot determine from the provided information.":
                valid_responses.append(response)
        
        if not valid_responses and not numerical_response:
            return f"Unable to find sufficient information to answer the query: {query}"
        
        # For simple numerical queries, extract the key value directly
        if numerical_response and not valid_responses:
            # Try to extract the key numerical value
            value_match = re.search(r'([\d,.]+%?)', numerical_response)
            if value_match:
                return f"<answer>{value_match.group(1)}</answer>"
        
        # Create a prompt with the sub-question responses and numerical results
        prompt = f"""
        Answer the question "{query}" based on the following information:
        
        """
        
        for i, response in enumerate(valid_responses):
            prompt += f"Sub-question {i+1}: {response['question']}\n"
            prompt += f"Answer: {response['response']}\n\n"
        
        prompt += f"Numerical analysis: {numerical_response}\n\n"
        
        prompt += """
        Using the above information, provide a direct answer to the main question.
        Focus only on the specific answer that was asked for, without explanation.
        
        If the answer involves a specific numerical value (especially a percentage or financial figure):
        1. Extract ONLY the final numerical value and place it within <answer></answer> tags
        2. For percentages, round to one decimal place (e.g., 14.1%, not 14.13%)
        3. Include the % symbol for percentage values
        4. Include the appropriate currency symbol for monetary values
        
        Example:
        1. If the question asks for a percentage increase and the calculated value is 14.13%, your answer should be ONLY: <answer>14.1%</answer>
        2. If the question asks for a dollar amount and the value is $1,234,567.89, your answer should be ONLY: <answer>$1,234,567.89</answer>
        
        DO NOT include any explanations or additional context in your response.
        """
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating final response with numerical results: {str(e)}")
            # Fallback to numerical response if available
            if numerical_response:
                # Try to extract a number from the numerical response
                value_match = re.search(r'([\d,.]+%?)', numerical_response)
                if value_match:
                    return f"<answer>{value_match.group(1)}</answer>"
                return f"<answer>{numerical_response}</answer>"
            
            # Last fallback is to concatenate sub-question responses
            result = "\n".join([r['response'] for r in valid_responses if r['response']])
            return result or "Could not generate an answer." 