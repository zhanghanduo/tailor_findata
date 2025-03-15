import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from post_process import extract_program_tokens, extract_answer, format_predictions_for_evaluation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on ConvFinQA test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data (JSON file or processed dataset)")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save predictions")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--load_in_4bit", action="store_true", help="Whether to load in 4-bit quantization")
    
    return parser.parse_args()


def prepare_test_data(test_data_path):
    """
    Prepare test data for inference.
    
    Args:
        test_data_path: Path to test data (JSON file or processed dataset)
        
    Returns:
        test_dataset: Dataset containing test examples
        example_ids: List of example IDs
    """
    # Check if test_data_path is a directory (processed dataset) or a file (JSON)
    if os.path.isdir(test_data_path):
        # Load processed dataset
        test_dataset = load_from_disk(test_data_path)
        if isinstance(test_dataset, dict) and "test" in test_dataset:
            test_dataset = test_dataset["test"]
    else:
        # Load JSON file
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Convert to dataset format
        test_examples = []
        for example in test_data:
            # Extract necessary information
            example_id = example["id"]
            pre_text = example.get("pre_text", "")
            post_text = example.get("post_text", "")
            table = example.get("table", "")
            
            # Create context
            context = f"{pre_text}\n\n{table}\n\n{post_text}"
            
            # Get the question (first turn of the conversation)
            question = example["annotation"]["dialogue_break"][0] if "dialogue_break" in example["annotation"] else ""
            
            test_examples.append({
                "id": example_id,
                "context": context,
                "question": question
            })
        
        # Create a dataset from the examples
        from datasets import Dataset
        test_dataset = Dataset.from_list(test_examples)
    
    # Extract example IDs
    example_ids = [example["id"] for example in test_dataset] if "id" in test_dataset.column_names else [f"example_{i}" for i in range(len(test_dataset))]
    
    return test_dataset, example_ids


def run_inference(model, tokenizer, test_dataset, args):
    """
    Run inference on test dataset.
    
    Args:
        model: The model to use for inference
        tokenizer: The tokenizer to use for inference
        test_dataset: Dataset containing test examples
        args: Command line arguments
        
    Returns:
        predictions: List of model outputs
    """
    predictions = []
    
    # Create dataloader
    for i in tqdm(range(0, len(test_dataset), args.batch_size), desc="Running inference"):
        batch = test_dataset[i:i+args.batch_size]
        
        # Prepare inputs
        if "conversations" in test_dataset.column_names:
            # Dataset is already in the right format
            inputs = tokenizer(
                [example["text"] for example in batch], 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=args.max_length
            ).to(args.device)
        else:
            # Format inputs for inference
            prompts = []
            for example in batch:
                context = example.get("context", "")
                question = example.get("question", "")
                
                prompt = f"I'm looking at some financial data. Here's the context:\n\n{context}\n\n{question}"
                prompts.append(prompt)
            
            inputs = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=args.max_length
            ).to(args.device)
        
        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode outputs
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Add to predictions
        predictions.extend(decoded_outputs)
    
    return predictions


def main():
    """Main function to run inference and save predictions."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    if args.load_in_4bit:
        # Load in 4-bit quantization
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            quantization_config=bnb_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Prepare test data
    print(f"Preparing test data from {args.test_data}")
    test_dataset, example_ids = prepare_test_data(args.test_data)
    
    # Run inference
    print("Running inference...")
    predictions = run_inference(model, tokenizer, test_dataset, args)
    
    # Save raw predictions
    raw_predictions_file = os.path.join(args.output_dir, "raw_predictions.json")
    with open(raw_predictions_file, 'w') as f:
        json.dump([{"id": id, "output": pred} for id, pred in zip(example_ids, predictions)], f, indent=2)
    
    print(f"Raw predictions saved to {raw_predictions_file}")
    
    # Format predictions for evaluation
    formatted_predictions = format_predictions_for_evaluation(predictions, example_ids)
    
    # Save formatted predictions
    formatted_predictions_file = os.path.join(args.output_dir, "formatted_predictions.json")
    with open(formatted_predictions_file, 'w') as f:
        json.dump(formatted_predictions, f, indent=2)
    
    print(f"Formatted predictions saved to {formatted_predictions_file}")
    
    # Print sample predictions
    print("\nSample predictions:")
    for i in range(min(3, len(predictions))):
        print(f"Example ID: {example_ids[i]}")
        print(f"Raw prediction: {predictions[i][:100]}...")
        print(f"Extracted program: {formatted_predictions[i]['predicted']}")
        print()


if __name__ == "__main__":
    main()  