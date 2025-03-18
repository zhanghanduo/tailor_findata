#!/usr/bin/env python3
import os
import json
import argparse
from tqdm import tqdm
import sys

from evaluate_rag import load_data, evaluate_dialogue, calculate_metrics
from rag_system import query_rag

def setup_directory_structure():
    """Create necessary directories if they don't exist."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def copy_sample_data(train_path, knowledge_path):
    """Copy sample data files to the data directory if they don't exist."""
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found. Please ensure the file exists.")
    
    if not os.path.exists(knowledge_path):
        print(f"Warning: {knowledge_path} not found. Please ensure the file exists.")

def main():
    parser = argparse.ArgumentParser(description='Run RAG system evaluation')
    parser.add_argument('--train_path', type=str, default='data/sample_train.json',
                        help='Path to the training data JSON')
    parser.add_argument('--knowledge_path', type=str, default='data/sample_knowledge.json',
                        help='Path to the knowledge data JSON')
    parser.add_argument('--output_path', type=str, default='results/evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed evaluation information')
    
    args = parser.parse_args()
    
    # Setup directory structure
    setup_directory_structure()
    
    # Check if files exist
    copy_sample_data(args.train_path, args.knowledge_path)
    
    # Load data
    try:
        train_data, knowledge_dict = load_data(args.train_path, args.knowledge_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data files exist and are in the correct location.")
        sys.exit(1)
    
    print(f"Loaded {len(train_data)} examples from {args.train_path}")
    print(f"Loaded {len(knowledge_dict)} knowledge items from {args.knowledge_path}")
    
    # Run evaluation for each example
    all_results = []
    for example in tqdm(train_data, desc="Evaluating examples"):
        result = evaluate_dialogue(example, knowledge_dict, args.verbose)
        all_results.append(result)
        
        # Print per-example results if verbose
        if args.verbose:
            print(f"\nExample ID: {result['example_id']}")
            print(f"Accuracy: {result['accuracy']:.2f} ({result['correct_count']}/{result['total_turns']})")
    
    # Calculate overall metrics
    metrics = calculate_metrics(all_results)
    
    # Print summary results
    print("\n============================================")
    print("              EVALUATION RESULTS            ")
    print("============================================")
    print(f"Examples evaluated: {metrics['examples_evaluated']}")
    print(f"Total turns evaluated: {metrics['total_turns']}")
    print(f"Overall turn accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Average example accuracy: {metrics['avg_example_accuracy']:.4f}")
    print(f"First turn accuracy: {metrics['first_turn_accuracy']:.4f}")
    print(f"Last turn accuracy (final answer): {metrics['last_turn_accuracy']:.4f}")
    print("============================================")
    
    # Save detailed results
    output = {
        'metrics': metrics,
        'example_results': all_results
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to {args.output_path}")

if __name__ == "__main__":
    main() 