import json
import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse
from tqdm import tqdm

# Import your RAG system here
# from rag_system import query_rag

# Mock function - replace with actual RAG system
def query_rag(question: str, context: str, history: List[Tuple[str, str]] = None) -> str:
    """
    Query the RAG system with a question and context.
    
    Args:
        question: The user's question
        context: The retrieved context
        history: Previous QA turns [(q1, a1), (q2, a2), ...]
    
    Returns:
        The RAG system's answer
    """
    # This is a mock function - implement with your actual RAG system
    pass

def load_data(train_path: str, knowledge_path: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Load the training and knowledge data
    
    Args:
        train_path: Path to the training data JSON
        knowledge_path: Path to the knowledge data JSON
    
    Returns:
        Tuple of (train_data, knowledge_dict)
    """
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    with open(knowledge_path, 'r') as f:
        knowledge_data = json.load(f)
    
    # Convert knowledge data to a dictionary for easy lookup
    knowledge_dict = {item['id']: item for item in knowledge_data}
    
    return train_data, knowledge_dict

def extract_answer(rag_response: str) -> str:
    """
    Extract the final answer from the RAG system response.
    
    This function extracts the answer portion from the RAG response,
    handling different answer formats.
    
    Args:
        rag_response: The full response from the RAG system
    
    Returns:
        The extracted answer
    """
    # Look for explicit answer tag
    answer_match = re.search(r'<answer>(.*?)</answer>', rag_response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Look for "The answer is" pattern
    answer_is_match = re.search(r'(?:The answer is|Answer:)\s*(.*?)(?:\.|$)', rag_response, re.DOTALL)
    if answer_is_match:
        return answer_is_match.group(1).strip()
    
    # If no specific pattern found, return the full response (might need refinement)
    return rag_response.strip()

def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    
    Args:
        answer: The answer to normalize
    
    Returns:
        Normalized answer
    """
    # Convert to string if it's not already
    answer = str(answer)
    
    # Remove whitespace
    answer = answer.strip()
    
    # Handle percentage format
    if '%' in answer:
        # Try to convert to float for comparison
        try:
            # Remove % sign and convert to float
            value = float(answer.replace('%', '').strip())
            return f"{value:.1f}%"
        except ValueError:
            pass
    
    # Handle dollar amounts
    if '$' in answer:
        # Remove $ and commas, then try to convert to float
        try:
            value = float(answer.replace('$', '').replace(',', '').strip())
            return f"{value}"
        except ValueError:
            pass
    
    # Try to convert numeric answers to float for comparison
    try:
        value = float(answer)
        # Format with 1 decimal place for consistent comparison
        if value.is_integer():
            return f"{int(value)}"
        else:
            return f"{value:.4f}".rstrip('0').rstrip('.') 
    except ValueError:
        # If not numeric, return as is
        return answer.lower()

def is_answer_correct(predicted: str, gold: str, tolerance: float = 0.01) -> bool:
    """
    Check if the predicted answer matches the gold answer.
    
    Args:
        predicted: The predicted answer from the RAG system
        gold: The gold standard answer
        tolerance: Tolerance for numerical comparison (percentage)
    
    Returns:
        True if the answer is correct, False otherwise
    """
    # Normalize both answers
    norm_predicted = normalize_answer(predicted)
    norm_gold = normalize_answer(gold)
    
    # Exact string match
    if norm_predicted == norm_gold:
        return True
    
    # Try numerical comparison for percentage and numeric values
    try:
        # Convert to float, handling percentage
        pred_val = float(norm_predicted.replace('%', ''))
        gold_val = float(norm_gold.replace('%', ''))
        
        # Calculate relative difference
        if gold_val != 0:
            rel_diff = abs(pred_val - gold_val) / abs(gold_val)
            return rel_diff <= tolerance
        else:
            return abs(pred_val - gold_val) <= tolerance
    except ValueError:
        # If conversion fails, they're not numeric, so use exact match
        return False

def evaluate_dialogue(example: Dict, knowledge_dict: Dict[str, Dict], 
                     verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single dialogue example.
    
    Args:
        example: The dialogue example
        knowledge_dict: Dictionary of knowledge items
        verbose: Whether to print detailed info
    
    Returns:
        Dictionary with evaluation results
    """
    dialogue_break = example['annotation']['dialogue_break']
    answer_list = example['annotation']['answer_list']
    
    # Get the relevant knowledge context
    example_id = example['id']
    context = knowledge_dict.get(example_id, {}).get('content', '')
    
    results = []
    history = []
    
    # Process each turn in the dialogue
    for i, (question, gold_answer) in enumerate(zip(dialogue_break, answer_list)):
        # Query the RAG system
        prompt = f"""
You are an expert financial analyst assistant. Answer the following question based only on the provided context.
Be concise and to the point. For numerical answers, provide just the number without explanation.

CONTEXT:
{context}

QUESTION:
{question}

PREVIOUS CONVERSATION:
{history}

Answer with just the final result. Format your answer like this:
<answer>Your concise answer here</answer>
"""
        
        # Get response from RAG system
        rag_response = query_rag(question, context, history)
        
        # Extract and normalize the answer
        extracted_answer = extract_answer(rag_response)
        
        # Check if the answer is correct
        is_correct = is_answer_correct(extracted_answer, gold_answer)
        
        # Add to results
        result = {
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': extracted_answer,
            'is_correct': is_correct
        }
        results.append(result)
        
        # Update conversation history
        history.append((question, extracted_answer))
        
        if verbose:
            print(f"Q: {question}")
            print(f"Gold: {gold_answer}")
            print(f"Predicted: {extracted_answer}")
            print(f"Correct: {is_correct}")
            print()
    
    # Calculate accuracy for this example
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = correct_count / len(results) if results else 0
    
    return {
        'example_id': example_id,
        'results': results,
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_turns': len(results)
    }

def calculate_metrics(all_results: List[Dict]) -> Dict[str, float]:
    """
    Calculate overall metrics from all evaluation results.
    
    Args:
        all_results: List of results from evaluate_dialogue
    
    Returns:
        Dictionary of metrics
    """
    # Overall accuracy across all turns
    total_correct = sum(r['correct_count'] for r in all_results)
    total_turns = sum(r['total_turns'] for r in all_results)
    overall_accuracy = total_correct / total_turns if total_turns > 0 else 0
    
    # Example-level accuracy
    example_accuracies = [r['accuracy'] for r in all_results]
    avg_example_accuracy = np.mean(example_accuracies) if example_accuracies else 0
    
    # First turn accuracy
    first_turn_correct = sum(1 for r in all_results if r['results'] and r['results'][0]['is_correct'])
    first_turn_accuracy = first_turn_correct / len(all_results) if all_results else 0
    
    # Last turn accuracy (final answer)
    last_turn_correct = sum(1 for r in all_results if r['results'] and r['results'][-1]['is_correct'])
    last_turn_accuracy = last_turn_correct / len(all_results) if all_results else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'avg_example_accuracy': avg_example_accuracy,
        'first_turn_accuracy': first_turn_accuracy,
        'last_turn_accuracy': last_turn_accuracy,
        'total_correct': total_correct,
        'total_turns': total_turns,
        'examples_evaluated': len(all_results)
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system on multi-turn QA')
    parser.add_argument('--train_path', type=str, default='data/sample_train.json',
                        help='Path to the training data JSON')
    parser.add_argument('--knowledge_path', type=str, default='data/sample_knowledge.json',
                        help='Path to the knowledge data JSON')
    parser.add_argument('--output_path', type=str, default='evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed evaluation information')
    
    args = parser.parse_args()
    
    # Load data
    train_data, knowledge_dict = load_data(args.train_path, args.knowledge_path)
    
    # Run evaluation for each example
    all_results = []
    for example in tqdm(train_data, desc="Evaluating examples"):
        result = evaluate_dialogue(example, knowledge_dict, args.verbose)
        all_results.append(result)
    
    # Calculate overall metrics
    metrics = calculate_metrics(all_results)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Examples evaluated: {metrics['examples_evaluated']}")
    print(f"Total turns: {metrics['total_turns']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2f}")
    print(f"Average example accuracy: {metrics['avg_example_accuracy']:.2f}")
    print(f"First turn accuracy: {metrics['first_turn_accuracy']:.2f}")
    print(f"Last turn accuracy: {metrics['last_turn_accuracy']:.2f}")
    
    # Save detailed results
    output = {
        'metrics': metrics,
        'example_results': all_results
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to {args.output_path}")

if __name__ == "__main__":
    main() 