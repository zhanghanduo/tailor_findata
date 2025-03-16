import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from post_process import extract_program_tokens, extract_answer, format_predictions_for_evaluation
from bs4 import BeautifulSoup
import datetime
import traceback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on ConvFinQA test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data (JSON file or processed dataset)")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save predictions")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--load_in_4bit", action="store_true", help="Whether to load in 4-bit quantization")
    
    return parser.parse_args()


def format_table_for_llm(table_data):
    """
    Convert table data to a markdown-style text table that's easier for LLMs to understand.
    
    Args:
        table_data: List of lists representing table rows and columns
        
    Returns:
        Formatted table as a string
    """
    if not table_data or not isinstance(table_data, list) or len(table_data) == 0:
        return "No table data found."
    
    # Process the table data
    # Convert all values to strings and handle None values
    processed_table = []
    for row in table_data:
        if not isinstance(row, list):
            # Skip non-list rows
            continue
        processed_row = []
        for cell in row:
            if cell is None:
                processed_row.append("")
            else:
                processed_row.append(str(cell).strip())
        if processed_row:  # Only add non-empty rows
            processed_table.append(processed_row)
    
    if not processed_table:
        return "No valid table data found."
    
    # Ensure all rows have the same number of columns by padding with empty strings
    max_cols = max(len(row) for row in processed_table)
    padded_table = [row + [''] * (max_cols - len(row)) for row in processed_table]
    
    # Determine column widths for proper alignment
    col_widths = [max(len(row[i]) for row in padded_table) for i in range(max_cols)]
    
    # Create header row with column names
    header_row = padded_table[0]
    header_separator = ['-' * width for width in col_widths]
    
    # Format the table with proper alignment
    formatted_table = []
    formatted_table.append(' | '.join(header_row[i].ljust(col_widths[i]) for i in range(max_cols)))
    formatted_table.append(' | '.join(header_separator[i] for i in range(max_cols)))
    
    # Add data rows
    for row in padded_table[1:]:
        formatted_table.append(' | '.join(row[i].ljust(col_widths[i]) for i in range(max_cols)))
    
    return '\n'.join(formatted_table)


def format_html_table_for_llm(html_table):
    """Convert HTML table to a markdown-style text table that's easier for LLMs to understand"""
    if not html_table or html_table.strip() == '':
        return "No table data found."
        
    soup = BeautifulSoup(html_table, 'html.parser')
    
    # Extract all rows
    rows = soup.find_all('tr')
    if not rows:
        return "No table data found."
    
    # Process the table data
    table_data = []
    for row in rows:
        cells = row.find_all(['td', 'th'])
        # Skip the first cell if it's just a row number
        if cells and cells[0].text.strip().isdigit():
            row_data = [cell.text.strip() for cell in cells[1:]]
        else:
            row_data = [cell.text.strip() for cell in cells]
        table_data.append(row_data)
    
    return format_table_for_llm(table_data)


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
        for i, example in enumerate(test_data):
            # Generate an example ID if not present
            example_id = example.get("id", f"example_{i}")
            
            # Handle different data formats
            # Check if the example has the new format with pre_text, post_text, and table_ori
            if "pre_text" in example and "post_text" in example and "table_ori" in example:
                # Join pre_text paragraphs
                pre_text = ' '.join(example.get("pre_text", []))
                
                # Join post_text paragraphs
                post_text = ' '.join(example.get("post_text", []))
                
                # Format table
                table_data = example.get("table_ori", [])
                formatted_table = format_table_for_llm(table_data)
                
                # Get the question (if available)
                question = example.get("question", "")
                
                # Create context
                context = f'{pre_text}\n\n{formatted_table}\n\n{post_text}'
            
            # Check if the example has the old format with annotation
            elif "annotation" in example:
                # Extract context information from annotation
                annos = example.get('annotation', {})
                pre_text = annos.get('amt_pre_text', '')
                post_text = annos.get('amt_post_text', '')
                html_table = annos.get('amt_table', '').replace('<td></td>', '<td>-</td>')
                
                # Format table for better LLM understanding
                formatted_table = format_html_table_for_llm(html_table)
                
                # Create context
                context = f'{pre_text}\n\n{formatted_table}\n\n{post_text}'
                
                # Get the question (first turn of the conversation)
                question = annos.get("dialogue_break", [""])[0] if annos.get("dialogue_break") else ""
            
            # If neither format is present, use whatever is available
            else:
                context = example.get("context", "")
                question = example.get("question", "")
            
            test_examples.append({
                "id": example_id,
                "context": context,
                "question": question
            })
        
        # Create a dataset from the examples
        from datasets import Dataset
        test_dataset = Dataset.from_list(test_examples)
        
        # Debug: Print dataset info
        print(f"Created dataset with {len(test_dataset)} examples")
        print(f"Dataset features: {test_dataset.features}")
        print(f"First example: {test_dataset[0] if len(test_dataset) > 0 else 'No examples'}")
    
    # Extract example IDs
    example_ids = [example["id"] for example in test_dataset] if "id" in test_dataset.column_names else [f"example_{i}" for i in range(len(test_dataset))]
    
    return test_dataset, example_ids


def get_system_prompt():
    """Return the system prompt for ConvFinQA task."""
    return """Your role is to solve financial questions by generating both the program tokens that represent the calculation and the final answer. 
For each question, ONLY provide:
1. The program tokens that represent the calculation using <begin_of_program> and <end_of_program> tags
2. The final answer using <begin_of_answer> and <end_of_answer> tags

The program tokens should follow this EXACT format:
<begin_of_program>
operation_name( number1 number2 ) EOF
<end_of_program>

<begin_of_answer>
numerical_result
<end_of_answer>

Examples of operations:
- For addition: add( number1 number2 ) EOF
- For subtraction: subtract( number1 number2 ) EOF
- For multiplication: multiply( number1 number2 ) EOF
- For division: divide( number1 number2 ) EOF

IMPORTANT: 
- Always include the # symbol before reference numbers (e.g., #0, #1)
- Never omit any part of the format
- Always end program tokens with the EOF token
- The answer should be ONLY the numerical result without any additional text, units, or explanations
- DO NOT include any financial context, table data, or explanations in your response
- DO NOT include any text outside of the specified tags

Your response should ONLY contain the program tokens and answer within their respective tags."""


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
    
    # Get system prompt
    system_prompt = get_system_prompt()
    
    # Debug: Print dataset info
    print(f"Dataset type: {type(test_dataset)}")
    print(f"Dataset columns: {test_dataset.column_names if hasattr(test_dataset, 'column_names') else 'No column_names attribute'}")
    print(f"Dataset length: {len(test_dataset)}")
    
    # Process examples one by one to avoid batch processing issues
    for i in tqdm(range(len(test_dataset)), desc="Running inference"):
        try:
            # Get example
            example = test_dataset[i]
            
            # Extract context and question
            context = ""
            question = ""
            
            if isinstance(example, dict):
                context = example.get("context", "")
                question = example.get("question", "")
            else:
                # Try to access as dataset item
                context = test_dataset[i]["context"] if "context" in test_dataset.column_names else ""
                question = test_dataset[i]["question"] if "question" in test_dataset.column_names else ""
            
            # If question is empty, create a default financial analysis question
            if not question or question.strip() == "":
                question = "Based on the financial information provided, what calculations can be performed to analyze the data?"
            
            # Create prompt with system prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"I'm looking at some financial data. Here's the context:\n\n{context}\n\nQuestion: {question}"}
            ]
            
            # Format messages for the model
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
                pad_to_multiple_of=8
            ).to(args.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            try:
                # Try to extract the assistant's response using the chat template
                assistant_prefix = tokenizer.apply_chat_template([{"role": "assistant", "content": ""}], tokenize=False)
                user_prefix = tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=False)
                
                # Find where the assistant's response starts
                if assistant_prefix in decoded_output:
                    assistant_response = decoded_output.split(assistant_prefix)[-1]
                    # Remove any trailing user message if present
                    if user_prefix in assistant_response:
                        assistant_response = assistant_response.split(user_prefix)[0]
                else:
                    # Fallback: just use the entire output
                    assistant_response = decoded_output
            except Exception as e:
                print(f"Error extracting assistant response for example {i}: {e}")
                assistant_response = decoded_output
            
            predictions.append(assistant_response)
            
            # Print sample output occasionally
            if i % 10 == 0:
                print(f"\nSample output for example {i}:")
                print(f"Question: {question[:100]}...")
                print(f"Response: {assistant_response[:100]}...")
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            # Add empty prediction
            predictions.append("")
    
    return predictions


def main():
    """Main function to run inference and save predictions."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "inference_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Inference started at: {datetime.datetime.now()}\n")
        f.write(f"Arguments: {args}\n\n")
    
    try:
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
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare test data
        print(f"Preparing test data from {args.test_data}")
        test_dataset, example_ids = prepare_test_data(args.test_data)
        
        # Log dataset info
        with open(log_file, 'a') as f:
            f.write(f"Dataset loaded with {len(test_dataset)} examples\n")
            f.write(f"First few example IDs: {example_ids[:5]}\n\n")
        
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
        
        # Log completion
        with open(log_file, 'a') as f:
            f.write(f"Inference completed at: {datetime.datetime.now()}\n")
            f.write(f"Total examples processed: {len(predictions)}\n")
            f.write(f"Results saved to: {args.output_dir}\n")
    
    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(f"{error_msg}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")
        raise


if __name__ == "__main__":
    main()