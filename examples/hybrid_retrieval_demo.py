#!/usr/bin/env python3
"""
Demonstration of hybrid retrieval capabilities in the Financial RAG system.
This script shows the impact of different vector/BM25 weight configurations.
"""

import os
import argparse
import logging
from dotenv import load_dotenv
from tabulate import tabulate
from rag_system.main import FinancialRAGSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_query_with_config(
    query: str, 
    data_path: str, 
    persist_dir: str, 
    vector_weight: float, 
    bm25_weight: float
) -> dict:
    """
    Run a query with a specific vector/BM25 weight configuration.
    
    Args:
        query: The query to run
        data_path: Path to the data
        persist_dir: Directory to persist indices
        vector_weight: Weight for vector search (0-1)
        bm25_weight: Weight for BM25 search (0-1)
        
    Returns:
        Dictionary with query results
    """
    # Normalize weights to ensure they sum to 1
    total = vector_weight + bm25_weight
    norm_vector_weight = vector_weight / total
    norm_bm25_weight = bm25_weight / total
    
    # Create RAG system with specified weights
    rag = FinancialRAGSystem(
        data_path=data_path,
        persist_dir=persist_dir,
        vector_weight=norm_vector_weight,
        bm25_weight=norm_bm25_weight,
        rebuild_indices=False  # Use existing indices
    )
    
    # Initialize the retriever only (no need to rebuild indices)
    rag.indices = rag.build_or_load_indices()
    rag.initialize_retriever()
    
    # Run query and return results
    logger.info(f"Running query with vector_weight={norm_vector_weight:.2f}, bm25_weight={norm_bm25_weight:.2f}")
    nodes = rag.retriever._retrieve(rag.retriever.QueryBundle(query))
    
    # Extract node information
    retrieved_nodes = []
    for i, node in enumerate(nodes[:10]):  # Top 10 nodes
        retrieved_nodes.append({
            "rank": i+1,
            "score": f"{node.score:.4f}" if node.score else "N/A",
            "node_type": node.node.metadata.get("type", "unknown"),
            "text_preview": node.node.text[:100] + "..." if len(node.node.text) > 100 else node.node.text
        })
    
    return {
        "vector_weight": norm_vector_weight,
        "bm25_weight": norm_bm25_weight,
        "nodes": retrieved_nodes
    }

def compare_configurations(query: str, data_path: str, persist_dir: str):
    """
    Compare different hybrid retrieval configurations for the same query.
    
    Args:
        query: The query to test
        data_path: Path to the data
        persist_dir: Directory to persist indices
    """
    # Test configurations with different vector vs BM25 weights
    configurations = [
        (1.0, 0.0),    # Pure vector search
        (0.8, 0.2),    # Mostly vector search
        (0.6, 0.4),    # Balanced but favoring vector search
        (0.5, 0.5),    # Equal weighting
        (0.4, 0.6),    # Balanced but favoring BM25
        (0.2, 0.8),    # Mostly BM25
        (0.0, 1.0)     # Pure BM25 search
    ]
    
    print(f"\n====== QUERY: '{query}' ======\n")
    
    # Run query with each configuration
    results = []
    for vector_weight, bm25_weight in configurations:
        result = run_query_with_config(
            query=query,
            data_path=data_path,
            persist_dir=persist_dir,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        results.append(result)
    
    # Display results
    for i, result in enumerate(results):
        config_name = f"Config {i+1}: Vector {result['vector_weight']:.1f}, BM25 {result['bm25_weight']:.1f}"
        print(f"\n{config_name}")
        print("=" * len(config_name))
        
        table_data = []
        for node in result["nodes"][:5]:  # Show top 5 for readability
            table_data.append([
                node["rank"],
                node["score"],
                node["node_type"],
                node["text_preview"]
            ])
        
        print(tabulate(
            table_data,
            headers=["Rank", "Score", "Type", "Preview"],
            tablefmt="grid"
        ))
    
    print("\nAnalysis: Look for differences in results between configurations.")
    print("- Pure vector search (Config 1) is better for semantic understanding.")
    print("- Pure BM25 search (Config 7) is better for keyword matching.")
    print("- Mixed configurations balance these approaches.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Demonstration")
    parser.add_argument("--query", type=str, default="What was the quarterly revenue growth?", 
                      help="Query to test")
    parser.add_argument("--data_path", type=str, help="Path to financial data")
    parser.add_argument("--persist_dir", type=str, help="Directory to persist indices")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get data path and persist dir from args or environment
    data_path = args.data_path or os.getenv("DATA_PATH")
    persist_dir = args.persist_dir or os.getenv("PERSIST_DIR")
    
    if not data_path or not persist_dir:
        print("ERROR: data_path and persist_dir must be provided via arguments or environment variables")
        return
    
    # Compare different hybrid retrieval configurations
    compare_configurations(args.query, data_path, persist_dir)

if __name__ == "__main__":
    main() 