import json
import argparse
from pathlib import Path
import numpy as np
import re


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate metrics for RAG system results")
    parser.add_argument("--results_file", type=str, required=True, 
                        help="Path to the JSON file containing results")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the metrics (optional)")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Tolerance for numeric comparison (default: 0.01 or 1%)")
    
    return parser.parse_args()


def normalize_numeric_value(value):
    """
    Normalize numeric values, handling percentage formats.
    
    Args:
        value: Value to normalize
        
    Returns:
        float: Normalized value or None if not numeric
    """
    if not value or not isinstance(value, str):
        return None
    
    # Remove whitespace and handle commas in numbers
    value = value.strip().replace(',', '')
    
    # Check for percentage format (e.g., "1%", "1.5%")
    percentage_match = re.match(r'^([-+]?\d*\.?\d+)\s*%$', value)
    if percentage_match:
        return float(percentage_match.group(1)) / 100
    
    # Try to convert to float directly
    try:
        return float(value)
    except ValueError:
        return None


def is_numeric(value):
    """Check if a value is numeric (can be converted to float)."""
    return normalize_numeric_value(value) is not None


def answers_match(predicted, golden, tolerance=0.01):
    """
    Check if predicted answer matches the golden answer within tolerance.
    
    Args:
        predicted: Predicted answer (string)
        golden: Golden answer (string)
        tolerance: Allowed percentage difference (default: 0.01 or 1%)
        
    Returns:
        bool: True if answers match, False otherwise
    """
    # Skip empty golden answers
    if not golden or golden.strip() == "":
        return None
    
    # If predicted is empty but golden is not, they don't match
    if not predicted or predicted.strip() == "":
        return False
    
    # Try to convert both to float for numeric comparison
    pred_num = normalize_numeric_value(predicted)
    gold_num = normalize_numeric_value(golden)
    
    if pred_num is not None and gold_num is not None:
        # Handle division by zero
        if gold_num == 0:
            # If both are zero, they match
            if pred_num == 0:
                return True
            # Otherwise, they don't match
            return False
        
        # Calculate relative difference
        rel_diff = abs(pred_num - gold_num) / abs(gold_num)
        
        # Check if within tolerance
        return rel_diff <= tolerance
    
    # For non-numeric values, check exact string match
    return predicted.strip().lower() == golden.strip().lower()


def calculate_metrics(results_file, tolerance=0.01):
    """
    Calculate metrics for RAG system results.
    
    Args:
        results_file: Path to the JSON file containing results
        tolerance: Tolerance for numeric comparison
        
    Returns:
        dict: Dictionary containing metrics
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Initialize counters
    total = 0
    matches = 0
    numeric_total = 0
    numeric_matches = 0
    empty_golden = 0
    
    # Process each result
    for result in results:
        qa = result.get("qa", {})
        predicted = qa.get("answer", "")
        golden = qa.get("golden_answer", "")
        
        # Skip if golden answer is empty
        if not golden or golden.strip() == "":
            empty_golden += 1
            continue
        
        # Increment total count
        total += 1
        
        # Check if answers match
        match_result = answers_match(predicted, golden, tolerance)
        if match_result:
            matches += 1
        
        # Track numeric answers separately
        if is_numeric(predicted) and is_numeric(golden):
            numeric_total += 1
            if match_result:
                numeric_matches += 1
    
    # Calculate metrics
    match_rate = matches / total if total > 0 else 0
    numeric_match_rate = numeric_matches / numeric_total if numeric_total > 0 else 0
    
    # Return metrics
    return {
        "total_examples": len(results),
        "valid_examples": total,
        "empty_golden_answers": empty_golden,
        "matched_examples": matches,
        "match_rate": match_rate,
        "numeric_examples": numeric_total,
        "numeric_matched": numeric_matches,
        "numeric_match_rate": numeric_match_rate,
    }


def main():
    """Main function to calculate and display metrics."""
    args = parse_args()
    
    # Calculate metrics
    print(f"Calculating metrics for {args.results_file} with {args.tolerance * 100}% tolerance...")
    metrics = calculate_metrics(args.results_file, args.tolerance)
    
    # Display metrics
    print("\nMetrics:")
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Valid examples (non-empty golden answers): {metrics['valid_examples']}")
    print(f"Empty golden answers (skipped): {metrics['empty_golden_answers']}")
    print(f"Matched examples: {metrics['matched_examples']}")
    print(f"Match rate: {metrics['match_rate']:.4f} ({metrics['match_rate'] * 100:.2f}%)")
    print(f"\nNumeric examples: {metrics['numeric_examples']}")
    print(f"Numeric matched: {metrics['numeric_matched']}")
    print(f"Numeric match rate: {metrics['numeric_match_rate']:.4f} ({metrics['numeric_match_rate'] * 100:.2f}%)")
    
    # Save metrics to file if specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {args.output_file}")


if __name__ == "__main__":
    main()