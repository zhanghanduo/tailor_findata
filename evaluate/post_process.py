import re
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset


def extract_program_tokens(text):
    """
    Extract program tokens from model output.
    Looks for content between <begin_of_program> and <end_of_program> tags.
    """
    program_pattern = r'<begin_of_program>\s*(.*?)\s*<end_of_program>'
    match = re.search(program_pattern, text, re.DOTALL)
    
    if match:
        program_text = match.group(1).strip()
        # Split by whitespace to get tokens
        tokens = program_text.split()
        
        # Ensure the last token is EOF
        if tokens and tokens[-1] != "EOF":
            tokens.append("EOF")
        
        return tokens
    
    # If no program found, return just EOF
    return ["EOF"]


def extract_answer(text):
    """
    Extract the final answer from model output.
    Looks for content between <begin_of_answer> and <end_of_answer> tags.
    """
    answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
    match = re.search(answer_pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return "N/A"


def format_predictions_for_evaluation(predictions, example_ids):
    """
    Format predictions for evaluation according to ConvFinQA requirements.
    
    Args:
        predictions: List of model outputs
        example_ids: List of example IDs corresponding to predictions
        
    Returns:
        List of dictionaries with 'id' and 'predicted' fields
    """
    formatted_predictions = []
    
    for pred, example_id in zip(predictions, example_ids):
        program_tokens = extract_program_tokens(pred)
        
        formatted_predictions.append({
            "id": example_id,
            "predicted": program_tokens
        })
    
    return formatted_predictions


def process_model_outputs(model_outputs_file, output_file, dataset_file=None):
    """
    Process model outputs and format them for evaluation.
    
    Args:
        model_outputs_file: File containing model outputs
        output_file: File to save formatted predictions
        dataset_file: Optional file containing example IDs
    """
    # Load model outputs
    with open(model_outputs_file, 'r') as f:
        model_outputs = json.load(f)
    
    # Get example IDs
    example_ids = []
    if dataset_file:
        # Load dataset to get example IDs
        dataset = load_dataset('json', data_files=dataset_file, split='train')
        example_ids = [example['id'] for example in dataset]
    else:
        # If no dataset file provided, assume IDs are in model_outputs
        example_ids = [output.get('id', f'example_{i}') for i, output in enumerate(model_outputs)]
    
    # Extract predictions
    predictions = [output.get('output', output.get('prediction', '')) for output in model_outputs]
    
    # Format predictions for evaluation
    formatted_predictions = format_predictions_for_evaluation(predictions, example_ids)
    
    # Save formatted predictions
    with open(output_file, 'w') as f:
        json.dump(formatted_predictions, f, indent=2)
    
    print(f"Formatted predictions saved to {output_file}")
    return formatted_predictions


def main():
    parser = argparse.ArgumentParser(description="Process model outputs for ConvFinQA evaluation")
    parser.add_argument("--model_outputs", type=str, required=True, help="File containing model outputs")
    parser.add_argument("--output_file", type=str, required=True, help="File to save formatted predictions")
    parser.add_argument("--dataset_file", type=str, help="Optional file containing example IDs")
    
    args = parser.parse_args()
    process_model_outputs(args.model_outputs, args.output_file, args.dataset_file)


if __name__ == "__main__":
    main() 