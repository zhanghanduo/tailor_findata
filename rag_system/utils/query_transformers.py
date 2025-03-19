from typing import List, Dict, Any, Optional
import logging
import re
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Prompt templates
QUERY_REWRITE_TEMPLATE = PromptTemplate(
    """You are an assistant that helps rewrite queries to be more effective for retrieving financial information from an RAG system.
    
The user's query is: {query}

Consider the following aspects in the financial domain:
1. Identify key financial terms, metrics, or calculations mentioned in the query
2. Consider time periods, years, or quarters referenced
3. Identify company names, ticker symbols, or specific financial statements mentioned
4. Look for numerical comparisons, trends, or specific data points requested

Your task is to rewrite the query to make it more specific, focusing on retrieving relevant financial information. If the query mentions calculations, tables, or numerical data, emphasize those aspects.

Rewritten query:"""
)

QUERY_EXPANSION_TEMPLATE = PromptTemplate(
    """You are an assistant that helps expand a financial query into multiple related queries to improve retrieval.
    
Original query: {query}

Generate 3-5 alternative queries that:
1. Rephrase the original query using different financial terminology
2. Break down complex queries into simpler, more focused queries
3. Include specific synonyms for financial terms
4. Explicitly ask for numerical data if the original query implies calculations
5. Specifically request information from tables if relevant

Return the alternative queries as a numbered list:"""
)

NUMERICAL_QUERY_ANALYSIS_TEMPLATE = PromptTemplate(
    """You are an assistant that analyzes financial queries to identify numerical operations that need to be performed.
    
User query: {query}

Analyze the query to identify:
1. What numerical values need to be extracted?
2. What calculations need to be performed?
3. Are there any time-based comparisons (year-over-year, quarter-over-quarter)?
4. Are there any specific financial metrics mentioned that require calculation?

Provide a structured analysis with the following fields:
- needs_numerical_computation: true/false
- values_to_extract: [list of specific values to extract]
- operations: [list of operations to perform]
- time_periods: [list of relevant time periods]
- financial_metrics: [list of financial metrics mentioned]"""
)

SUB_QUESTIONS_TEMPLATE = PromptTemplate(
    """You are an assistant that breaks down complex financial queries into smaller, more focused sub-questions.
    
Original query: {query}

First, determine if this query is simple enough to answer directly or needs to be broken down:
- Simple queries ask for a specific fact, number, or piece of information.
- Complex queries require understanding multiple pieces of information or performing calculations.

If the query is SIMPLE, return exactly ONE sub-question that is identical to the original query.

If the query is COMPLEX, break it down into 2-3 simpler sub-questions that:
1. Focus on retrieving specific pieces of information needed to answer the original query
2. Target specific tables or numerical data if needed
3. Are clear and directly answerable from financial documents

When you answer:
1. For numerical questions, provide exact numerical values with appropriate units
2. If a percentage or specific number is requested, extract ONLY that value in your final answer
3. For percentages, round to one decimal place (e.g., 14.1%, not 14.13%)
4. Put the final numerical answer in <answer></answer> tags

Sub-questions:"""
)

class QueryTransformer:
    """
    Class for transforming queries to improve retrieval performance.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        Initialize the query transformer.
        
        Args:
            model_name: Name of the LLM model to use.
            temperature: Temperature setting for the LLM.
        """
        self.llm = OpenAI(model=model_name, temperature=temperature)
    
    def transform(self, query: str) -> str:
        """
        Transform a query to make it more effective for retrieval.
        This is a wrapper around rewrite_query for compatibility.
        
        Args:
            query: Original user query.
            
        Returns:
            Transformed query.
        """
        return self.rewrite_query(query)
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a query to make it more effective for financial information retrieval.
        
        Args:
            query: Original user query.
            
        Returns:
            Rewritten query.
        """
        logger.info(f"Rewriting query: {query}")
        response = self.llm.complete(
            QUERY_REWRITE_TEMPLATE.format(query=query)
        )
        rewritten_query = str(response).strip()
        logger.info(f"Rewritten query: {rewritten_query}")
        return rewritten_query
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into multiple alternative queries.
        
        Args:
            query: Original user query.
            
        Returns:
            List of alternative queries.
        """
        logger.info(f"Expanding query: {query}")
        response = self.llm.complete(
            QUERY_EXPANSION_TEMPLATE.format(query=query)
        )
        
        # Extract numbered queries from response
        expanded_queries_text = str(response).strip()
        expanded_queries = []
        
        # Extract numbered lines using regex
        pattern = r"^\d+\.?\s*(.+)$"
        for line in expanded_queries_text.split("\n"):
            match = re.match(pattern, line.strip())
            if match:
                expanded_queries.append(match.group(1).strip())
        
        # If regex didn't work, just split by newlines and clean up
        if not expanded_queries:
            expanded_queries = [q.strip() for q in expanded_queries_text.split("\n") if q.strip()]
        
        logger.info(f"Generated {len(expanded_queries)} expanded queries")
        return expanded_queries
    
    def analyze_numerical_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to identify numerical operations needed.
        
        Args:
            query: User query.
            
        Returns:
            Dictionary with analysis results.
        """
        logger.info(f"Analyzing numerical aspects of query: {query}")
        response = self.llm.complete(
            NUMERICAL_QUERY_ANALYSIS_TEMPLATE.format(query=query)
        )
        
        analysis_text = str(response).strip()
        
        # Extract structured data from response
        analysis = {
            "needs_numerical_computation": "true" in analysis_text.lower(),
            "query": query,
            "analysis": analysis_text
        }
        
        # Try to extract specific fields
        patterns = {
            "values_to_extract": r"values_to_extract:?\s*\[(.*?)\]",
            "operations": r"operations:?\s*\[(.*?)\]",
            "time_periods": r"time_periods:?\s*\[(.*?)\]",
            "financial_metrics": r"financial_metrics:?\s*\[(.*?)\]"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
            if match:
                items = [item.strip().strip("'\"") for item in match.group(1).split(",")]
                analysis[key] = [item for item in items if item]
            else:
                analysis[key] = []
        
        logger.info(f"Numerical analysis: needs_computation={analysis['needs_numerical_computation']}")
        return analysis
    
    def generate_sub_questions(self, query: str) -> List[str]:
        """
        Break down a complex query into simpler sub-questions.
        
        Args:
            query: Original complex query.
            
        Returns:
            List of sub-questions.
        """
        logger.info(f"Generating sub-questions for: {query}")
        response = self.llm.complete(
            SUB_QUESTIONS_TEMPLATE.format(query=query)
        )
        
        sub_questions_text = str(response).strip()
        sub_questions = []
        
        # Extract numbered questions
        pattern = r"^\d+\.?\s*(.+)$"
        for line in sub_questions_text.split("\n"):
            match = re.match(pattern, line.strip())
            if match:
                sub_questions.append(match.group(1).strip())
        
        # If regex didn't work, just split by newlines and clean up
        if not sub_questions:
            sub_questions = [q.strip() for q in sub_questions_text.split("\n") if q.strip()]
        
        logger.info(f"Generated {len(sub_questions)} sub-questions")
        return sub_questions 