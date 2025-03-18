from typing import List, Dict, Any, Optional, Union
import logging
from llama_index.core import VectorStoreIndex, QueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.core.response_synthesizers import get_response_synthesizer

from rag_system.utils.query_transformers import QueryTransformer
from rag_system.utils.numerical_operations import perform_numerical_operations

logger = logging.getLogger(__name__)

class FinancialSubQuestionEngine:
    """
    SubQuestionQueryEngine for handling complex financial queries by breaking them down
    into simpler sub-questions.
    """
    
    def __init__(
        self,
        text_index: VectorStoreIndex,
        table_index: VectorStoreIndex,
        context_index: VectorStoreIndex,
        llm: Optional[OpenAI] = None
    ):
        """
        Initialize the financial sub-question engine.
        
        Args:
            text_index: VectorStoreIndex for text nodes.
            table_index: VectorStoreIndex for table nodes.
            context_index: VectorStoreIndex for all nodes with broader context.
            llm: OpenAI LLM for query transformation and answering.
        """
        self.text_index = text_index
        self.table_index = table_index
        self.context_index = context_index
        self.llm = llm
        
        # Initialize query transformer
        self.query_transformer = QueryTransformer(
            model_name=llm.model if llm else "gpt-3.5-turbo",
            temperature=llm.temperature if llm else 0.1
        ) if llm else None
        
        # Create query engines for each index
        self.text_engine = text_index.as_query_engine(
            similarity_top_k=5,
            llm=llm
        )
        
        self.table_engine = table_index.as_query_engine(
            similarity_top_k=5,
            llm=llm
        )
        
        self.context_engine = context_index.as_query_engine(
            similarity_top_k=5,
            llm=llm
        )
        
        # Create tools for sub-question engine
        self.tools = self._create_tools()
        
        # Create sub-question engine
        self.sub_question_engine = self._create_sub_question_engine()
    
    def _create_tools(self) -> List[QueryEngineTool]:
        """
        Create query engine tools for the sub-question engine.
        
        Returns:
            List of QueryEngineTool objects.
        """
        text_tool = QueryEngineTool.from_defaults(
            query_engine=self.text_engine,
            name="text_search",
            description="Useful for searching through textual financial information, narratives, and descriptions.",
        )
        
        table_tool = QueryEngineTool.from_defaults(
            query_engine=self.table_engine,
            name="table_search",
            description="Useful for searching through tables with financial data, metrics, and numerical information.",
        )
        
        context_tool = QueryEngineTool.from_defaults(
            query_engine=self.context_engine,
            name="context_search",
            description="Useful for finding broader context and background information about financial data.",
        )
        
        return [text_tool, table_tool, context_tool]
    
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
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a complex query by breaking it down into sub-questions.
        
        Args:
            query: User query.
            
        Returns:
            Dictionary containing the query results and metadata.
        """
        logger.info(f"Processing complex query: {query}")
        
        # Check if the query needs numerical computation
        numerical_analysis = None
        if self.query_transformer:
            numerical_analysis = self.query_transformer.analyze_numerical_query(query)
            needs_numerical = numerical_analysis.get("needs_numerical_computation", False)
            logger.info(f"Query needs numerical computation: {needs_numerical}")
        
        # Generate sub-questions if query transformer is available
        sub_questions = []
        if self.query_transformer:
            sub_questions = self.query_transformer.generate_sub_questions(query)
            logger.info(f"Generated {len(sub_questions)} sub-questions")
        
        # If no sub-questions were generated or only one, just use the original query
        if not sub_questions or len(sub_questions) <= 1:
            logger.info("No sub-questions generated, using original query")
            response = self.sub_question_engine.query(query)
            
            return {
                "query": query,
                "response": str(response),
                "sub_questions": [],
                "numerical_analysis": numerical_analysis
            }
        
        # Process each sub-question and collect responses
        sub_question_responses = []
        for i, sub_question in enumerate(sub_questions):
            logger.info(f"Processing sub-question {i+1}: {sub_question}")
            sub_response = self.sub_question_engine.query(sub_question)
            sub_question_responses.append({
                "question": sub_question,
                "response": str(sub_response)
            })
        
        # Get the final response by answering the original query with the sub-question responses
        context = "Based on the following information:\n\n"
        for i, resp in enumerate(sub_question_responses):
            context += f"Question {i+1}: {resp['question']}\n"
            context += f"Answer {i+1}: {resp['response']}\n\n"
        
        context += f"Please answer the original question: {query}"
        
        final_response = self.llm.complete(context)
        
        return {
            "query": query,
            "response": str(final_response),
            "sub_questions": sub_question_responses,
            "numerical_analysis": numerical_analysis
        }
    
    def query_with_numerical_operations(self, query: str) -> Dict[str, Any]:
        """
        Process a query that requires numerical operations.
        
        Args:
            query: User query.
            
        Returns:
            Dictionary containing the query results and numerical analysis.
        """
        logger.info(f"Processing numerical query: {query}")
        
        # First run the query through the standard sub-question engine
        result = self.query(query)
        
        # If the query requires numerical computation, perform additional operations
        if self.query_transformer:
            numerical_analysis = self.query_transformer.analyze_numerical_query(query)
            
            if numerical_analysis.get("needs_numerical_computation", False):
                # Retrieve nodes from all indices with a focus on numerical data
                text_nodes = self.text_index.as_retriever(similarity_top_k=10).retrieve(query)
                table_nodes = self.table_index.as_retriever(similarity_top_k=10).retrieve(query)
                
                # Extract nodes from NodeWithScore objects
                all_nodes = [node.node for node in text_nodes + table_nodes]
                
                # Perform numerical operations
                numerical_results = perform_numerical_operations(
                    all_nodes, 
                    query, 
                    numerical_analysis,
                    self.llm
                )
                
                # Add numerical results to the query result
                result["numerical_operations"] = numerical_results
                
                # If there's an LLM computation result, append it to the response
                if "llm_computation_result" in numerical_results:
                    result["numerical_response"] = numerical_results["llm_computation_result"]
        
        return result 