import os
import json
from dotenv import load_dotenv
import logging

from rag_system.main import FinancialRAGSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def print_separator():
    """Print a separator line."""
    print("\n" + "=" * 80 + "\n")

def main():
    """Run example queries on the financial RAG system."""
    # Initialize the RAG system
    rag_system = FinancialRAGSystem(
        data_path="data/sample_knowledge.json",
        persist_dir="output/chroma_db",
        rebuild_indices=True  # Set to False to use existing indices
    )
    
    # Initialize the system
    rag_system.initialize()
    
    # Sample queries to demonstrate various capabilities
    queries = [
        # Simple factual query
        "What was Republic Services' revenue in 2008?",
        
        # Numerical calculation query
        "What was the percentage change in Jack Henry's net income from 2008 to 2009?",
        
        # Query about tables
        "Show me the table with net income data for Jack Henry.",
        
        # Complex query requiring multiple sub-questions
        "Compare the financial performance of Republic Services and Jack Henry based on their income and revenue growth.",
        
        # Query with temporal comparison
        "How did Republic Services' revenue in 2008 compare to the previous year?",
        
        # Query requiring understanding of financial metrics
        "What factors affected Jack Henry's liquidity and capital resources according to the report?"
    ]
    
    # Process each query
    results = []
    for i, query in enumerate(queries):
        print_separator()
        print(f"Query {i+1}: {query}")
        print_separator()
        
        # Process the query
        result = rag_system.query(query)
        
        # Print the response
        print("Response:")
        print(result["response"])
        
        # Print sub-questions if available
        if result.get("sub_questions"):
            print("\nSub-questions and answers:")
            for j, sub_q in enumerate(result["sub_questions"]):
                print(f"\n{j+1}. {sub_q['question']}")
                print(f"   Answer: {sub_q['response']}")
        
        # Print numerical response if available
        if result.get("numerical_response"):
            print("\nNumerical analysis:")
            print(result["numerical_response"])
        
        # Add to results
        results.append({
            "query": query,
            "response": result["response"],
            "sub_questions": result.get("sub_questions", []),
            "numerical_response": result.get("numerical_response")
        })
        
        print_separator()
    
    # Save results to a file
    with open("rag_system/output/example_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(queries)} queries. Results saved to example_results.json")

if __name__ == "__main__":
    main() 