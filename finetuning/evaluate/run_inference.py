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
import re
import transformers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on ConvFinQA test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--base_model_name", type=str, default=None, help="Name or path of the base model for PEFT adapters. If not provided, will try to infer from adapter_config.json")
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
            
            # Extract golden answer if available
            golden_answer = ""
            question = ""
            
            # Check for qa field
            if "qa" in example and isinstance(example["qa"], dict):
                if "answer" in example["qa"]:
                    golden_answer = str(example["qa"]["answer"]) if example["qa"]["answer"] is not None else ""
                if "question" in example["qa"]:
                    question = str(example["qa"]["question"]) if example["qa"]["question"] is not None else ""
            # Check for qa_0, qa_1, etc. fields
            elif any(key.startswith("qa_") for key in example.keys()):
                # Get all qa_* fields sorted numerically
                qa_keys = sorted([k for k in example.keys() if k.startswith("qa_") and k[3:].isdigit()], 
                                key=lambda x: int(x[3:]))
                
                # Use the first qa_* entry for the answer and question
                if qa_keys and isinstance(example[qa_keys[0]], dict):
                    first_qa = example[qa_keys[0]]
                    if "answer" in first_qa:
                        golden_answer = str(first_qa["answer"]) if first_qa["answer"] is not None else ""
                    if "question" in first_qa:
                        question = str(first_qa["question"]) if first_qa["question"] is not None else ""
                    
                    # Store all qa_* entries in a list under a single "qa" key
                    all_qa_entries = []
                    for qa_key in qa_keys:
                        if isinstance(example[qa_key], dict):
                            all_qa_entries.append(example[qa_key])
                    
                    # Replace the qa field with the combined list
                    example["qa"] = {"entries": all_qa_entries}
            # Fall back to direct answer/question fields
            else:
                if "answer" in example:
                    golden_answer = str(example["answer"]) if example["answer"] is not None else ""
                if "question" in example:
                    question = str(example["question"]) if example["question"] is not None else ""
            
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
                
                # Get the question (first turn of the conversation) if not already set
                if not question:
                    question = annos.get("dialogue_break", [""])[0] if annos.get("dialogue_break") else ""
            
            # If neither format is present, use whatever is available
            else:
                context = example.get("context", "")
            
            test_examples.append({
                "id": str(example_id),
                "context": str(context),
                "question": str(question),
                "answer": str(golden_answer),
                "qa": json.dumps(example.get("qa", {})) if isinstance(example.get("qa"), dict) else str(example.get("qa", "{}"))
            })
        
        # Create a dataset from the examples
        from datasets import Dataset, Features, Value
        
        # Define explicit schema to prevent type inference issues
        features = Features({
            "id": Value("string"),
            "context": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "qa": Value("string")  # Convert the qa dict to string to avoid schema issues
        })
        
        # Convert qa dictionaries to strings
        for example in test_examples:
            if isinstance(example["qa"], dict):
                example["qa"] = json.dumps(example["qa"])
        
        test_dataset = Dataset.from_list(test_examples, features=features)
        
        # Debug: Print dataset info
        print(f"Created dataset with {len(test_dataset)} examples")
        print(f"Dataset features: {test_dataset.features}")
        print(f"First example: {test_dataset[0] if len(test_dataset) > 0 else 'No examples'}")
    
    # Randomly split the dataset to 1/5 of its original size
    total_size = len(test_dataset)
    subset_size = total_size // 5
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random indices for the subset
    subset_indices = np.random.choice(total_size, size=subset_size, replace=False)
    subset_indices = sorted(subset_indices)  # Sort indices for consistency
    
    # Select the subset
    test_dataset = test_dataset.select(subset_indices)
    
    print(f"\nReduced dataset size from {total_size} to {len(test_dataset)} examples (1/5 random sample)")
    
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
- The numerical result must be a single number with no line breaks, spaces, or extra characters
- Format decimal numbers properly (e.g., 44.0 not 44\n0)
- DO NOT include any financial context, table data, or explanations in your response
- DO NOT include any text outside of the specified tags

Examples of correct answer formats:
<begin_of_answer>
44.0
<end_of_answer>

<begin_of_answer>
-1889.0
<end_of_answer>

Your response should ONLY contain the program tokens and answer within their respective tags.
"""


def is_valid_output(output):
    """
    Check if the model output is valid.
    
    Args:
        output: The model output to check
        
    Returns:
        bool: True if the output is valid, False otherwise
    """
    # Check if the output contains both program and answer tags
    has_program_tags = "<begin_of_program>" in output and "<end_of_program>" in output
    has_answer_tags = "<begin_of_answer>" in output and "<end_of_answer>" in output
    
    # Check if the output contains operation tokens
    operation_pattern = r'(add|subtract|multiply|divide)\s*\('
    has_operation = re.search(operation_pattern, output, re.IGNORECASE) is not None
    
    # Check if the output contains EOF token
    has_eof = "EOF" in output
    
    # Check if the output contains a numerical answer
    number_pattern = r'<begin_of_answer>\s*(-?\d+\.?\d*)\s*<end_of_answer>'
    has_number = re.search(number_pattern, output, re.DOTALL) is not None
    
    # Check for invalid content like "and" as the entire program or answer
    invalid_content = False
    if has_program_tags and has_answer_tags:
        # Extract program content
        program_pattern = r'<begin_of_program>\s*(.*?)\s*<end_of_program>'
        program_match = re.search(program_pattern, output, re.DOTALL)
        
        # Extract answer content
        answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
        answer_match = re.search(answer_pattern, output, re.DOTALL)
        
        if program_match and answer_match:
            program_content = program_match.group(1).strip()
            answer_content = answer_match.group(1).strip()
            
            # Check if either content is just "and" or other invalid words
            invalid_words = ["and", "or", "the", "is", "a", "an"]
            if (program_content.lower() in invalid_words or 
                answer_content.lower() in invalid_words):
                invalid_content = True
    
    # Return True if the output has program tags and either operation tokens or EOF,
    # and doesn't contain invalid content
    return has_program_tags and has_answer_tags and (has_operation or has_eof) and not invalid_content


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
    
    # Check if we're using a Gemma3 model
    is_gemma3 = "gemma3" in args.model_path.lower() or "gemma-3" in args.model_path.lower()
    
    # Check if we're using a Llama 3.3 model
    is_llama3_3 = "llama3.3" in args.model_path.lower() or "llama-3.3" in args.model_path.lower()
    
    # Check if we're using PEFT adapter
    is_peft_model = hasattr(model, "get_base_model") and hasattr(model, "peft_config")
    if is_peft_model:
        print("Running inference with PEFT adapter model")
        base_model_name = model.peft_config['default'].base_model_name_or_path.lower()
        is_gemma3 = is_gemma3 or "gemma3" in base_model_name or "gemma-3" in base_model_name
        is_llama3_3 = is_llama3_3 or "llama3.3" in base_model_name or "llama-3.3" in base_model_name
        if is_gemma3:
            print("Detected Gemma 3 as base model")
        if is_llama3_3:
            print("Detected Llama 3.3 as base model")
    
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
                if "question" in example:
                    question = example.get("question", "")
                elif "qa" in example:
                    # Check if qa is a string (JSON)
                    if isinstance(example["qa"], str):
                        try:
                            qa_dict = json.loads(example["qa"])
                            if isinstance(qa_dict, dict):
                                if "question" in qa_dict:
                                    question = qa_dict["question"]
                                # Handle entries list from qa_0, qa_1 etc.
                                elif "entries" in qa_dict and isinstance(qa_dict["entries"], list) and len(qa_dict["entries"]) > 0:
                                    first_entry = qa_dict["entries"][0]
                                    if isinstance(first_entry, dict) and "question" in first_entry:
                                        question = first_entry["question"]
                        except json.JSONDecodeError:
                            pass
                    # Check if qa is a dict
                    elif isinstance(example["qa"], dict) and "question" in example["qa"]:
                        question = example["qa"]["question"]
            else:
                # Try to access as dataset item
                if "context" in test_dataset.column_names:
                    context = test_dataset[i]["context"]
                
                if "question" in test_dataset.column_names:
                    question = test_dataset[i]["question"]
                elif "qa" in test_dataset.column_names:
                    # Check if qa is a string (JSON)
                    if isinstance(test_dataset[i]["qa"], str):
                        try:
                            qa_dict = json.loads(test_dataset[i]["qa"])
                            if isinstance(qa_dict, dict):
                                if "question" in qa_dict:
                                    question = qa_dict["question"]
                                # Handle entries list from qa_0, qa_1 etc.
                                elif "entries" in qa_dict and isinstance(qa_dict["entries"], list) and len(qa_dict["entries"]) > 0:
                                    first_entry = qa_dict["entries"][0]
                                    if isinstance(first_entry, dict) and "question" in first_entry:
                                        question = first_entry["question"]
                        except json.JSONDecodeError:
                            pass
                    # Check if qa is a dict
                    elif isinstance(test_dataset[i]["qa"], dict) and "question" in test_dataset[i]["qa"]:
                        question = test_dataset[i]["qa"]["question"]
            
            # Ensure context and question are strings
            context = str(context) if context is not None else ""
            question = str(question) if question is not None else ""
            
            # If question is empty, create a default financial analysis question
            if not question or question.strip() == "":
                question = "Based on the financial information provided, what calculations can be performed to analyze the data?"
            
            # Create user content
            user_content = f"I'm looking at some financial data. Here's the context:\n\n{context}\n\nQuestion: {question}"
            
            # Handle Gemma3 models differently
            if is_gemma3:
                # Format messages for Gemma3
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": f"{system_prompt}\n\n{user_content}"}]
                }]
                
                # Apply chat template with add_generation_prompt=True
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(args.device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=0.1,  # Lower temperature for more deterministic outputs
                        top_k=64,
                        top_p=0.95,
                    )
            # Handle Llama 3.3 models differently
            elif is_llama3_3:
                # Format messages for Llama 3.3
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                
                # Apply chat template with add_generation_prompt=True
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(args.device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        do_sample=False,
                        temperature=1.5,  # Lower temperature for more deterministic outputs
                        min_p = 0.1,
                    )
            else:
                # Create prompt with system prompt for non-Gemma3 models
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
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
                        eos_token_id=tokenizer.eos_token_id,
                        temperature=0.1,  # Lower temperature for more deterministic outputs
                        num_beams=1,  # Use greedy decoding
                        repetition_penalty=1.2,  # Slight penalty for repetition
                        # Stop generation at specific tokens
                        stopping_criteria=transformers.StoppingCriteriaList([
                            transformers.StoppingCriteria() if not hasattr(tokenizer, 'eos_token_id') else 
                            transformers.StoppingCriteriaList([])
                        ])
                    )
            
            # Decode
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            try:
                # Check if we're using a Qwen model
                is_qwen = "qwen" in args.model_path.lower()
                
                # For Llama 3.3 models, extract the model response
                if is_llama3_3:
                    # For Llama 3.3, the response should be after the prompt
                    assistant_response = decoded_output
                    
                    # Debug: Print raw output
                    if i % 10 == 0:
                        print(f"\nLlama 3.3 raw output for example {i} (first 200 chars):")
                        print(f"{decoded_output[:200]}...")
                    
                    # Try to extract just the generated part (after the prompt)
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False
                    )
                    if assistant_response.startswith(prompt_text):
                        assistant_response = assistant_response[len(prompt_text):].strip()
                        if i % 10 == 0:
                            print(f"Extracted response after prompt (first 100 chars): {assistant_response[:100]}...")
                    
                    # Look for program and answer tags
                    program_pattern = r'<begin_of_program>(.*?)<end_of_program>'
                    program_match = re.search(program_pattern, assistant_response, re.DOTALL)
                    
                    answer_pattern = r'<begin_of_answer>(.*?)<end_of_answer>'
                    answer_match = re.search(answer_pattern, assistant_response, re.DOTALL)
                    
                    if program_match and answer_match:
                        program_content = program_match.group(1).strip()
                        answer_content = answer_match.group(1).strip()
                        
                        clean_response = (
                            f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                            f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
                        )
                        assistant_response = clean_response
                        if i % 10 == 0:
                            print(f"Found program and answer tags in Llama 3.3 response")
                    else:
                        if i % 10 == 0:
                            print(f"No program/answer tags found in extracted response")
                    
                    # If no tags found, try to extract from the full output
                    if not is_valid_output(assistant_response):
                        if i % 10 == 0:
                            print(f"Trying to find tags in full output...")
                        program_match = re.search(program_pattern, decoded_output, re.DOTALL)
                        answer_match = re.search(answer_pattern, decoded_output, re.DOTALL)
                        
                        if program_match and answer_match:
                            program_content = program_match.group(1).strip()
                            answer_content = answer_match.group(1).strip()
                            
                            clean_response = (
                                f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                                f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
                            )
                            assistant_response = clean_response
                            if i % 10 == 0:
                                print(f"Found program and answer tags in full output")
                        else:
                            if i % 10 == 0:
                                print(f"No program/answer tags found in full output either")
                            
                            # Try to find operation tokens directly
                            operation_pattern = r'(add|subtract|multiply|divide)\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)'
                            match = re.search(operation_pattern, decoded_output, re.IGNORECASE)
                            
                            if match:
                                if i % 10 == 0:
                                    print(f"Found operation pattern: {match.group(0)}")
                                operation = match.group(1).lower()
                                num1 = match.group(2)
                                num2 = match.group(3)
                                
                                # Try to calculate the answer
                                try:
                                    if operation == "add":
                                        answer = str(float(num1) + float(num2))
                                    elif operation == "subtract":
                                        answer = str(float(num1) - float(num2))
                                    elif operation == "multiply":
                                        answer = str(float(num1) * float(num2))
                                    elif operation == "divide":
                                        answer = str(float(num1) / float(num2))
                                    else:
                                        answer = "N/A"
                                        
                                    # Create a valid output
                                    assistant_response = (
                                        f"<begin_of_program>\n{operation}( {num1} {num2} ) EOF\n<end_of_program>\n\n"
                                        f"<begin_of_answer>\n{answer}\n<end_of_answer>"
                                    )
                                    if i % 10 == 0:
                                        print(f"Created valid output from operation pattern")
                                except Exception as e:
                                    if i % 10 == 0:
                                        print(f"Error calculating answer: {e}")
                            else:
                                if i % 10 == 0:
                                    print(f"No operation pattern found either")
                                
                                # Llama 3.3 specific fallback: Look for any calculation pattern
                                calculation_pattern = r'(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)'
                                calc_match = re.search(calculation_pattern, decoded_output)
                                
                                if calc_match:
                                    if i % 10 == 0:
                                        print(f"Found calculation pattern: {calc_match.group(0)}")
                                    
                                    num1 = calc_match.group(1)
                                    operator = calc_match.group(2)
                                    num2 = calc_match.group(3)
                                    result = calc_match.group(4)
                                    
                                    # Map operator to operation name
                                    operation_map = {
                                        '+': 'add',
                                        '-': 'subtract',
                                        '*': 'multiply',
                                        '/': 'divide'
                                    }
                                    
                                    operation = operation_map.get(operator, 'add')
                                    
                                    # Create a valid output
                                    assistant_response = (
                                        f"<begin_of_program>\n{operation}( {num1} {num2} ) EOF\n<end_of_program>\n\n"
                                        f"<begin_of_answer>\n{result}\n<end_of_answer>"
                                    )
                                    if i % 10 == 0:
                                        print(f"Created valid output from calculation pattern")
                                else:
                                    # Last resort: Look for any numbers in the output
                                    number_pattern = r'(\d+\.?\d*)'
                                    numbers = re.findall(number_pattern, decoded_output)
                                    
                                    if len(numbers) >= 3:  # Need at least 2 operands and 1 result
                                        if i % 10 == 0:
                                            print(f"Found numbers: {numbers[:3]}")
                                        
                                        # Assume addition as default operation
                                        num1 = numbers[0]
                                        num2 = numbers[1]
                                        result = numbers[2]
                                        
                                        # Create a valid output
                                        assistant_response = (
                                            f"<begin_of_program>\nadd( {num1} {num2} ) EOF\n<end_of_program>\n\n"
                                            f"<begin_of_answer>\n{result}\n<end_of_answer>"
                                        )
                                        if i % 10 == 0:
                                            print(f"Created valid output from extracted numbers")
                
                # For Gemma3 models, extract the model response
                elif is_gemma3:
                    # For Gemma3, the response should be after the prompt
                    assistant_response = decoded_output
                    
                    # Try to extract just the generated part (after the prompt)
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    if assistant_response.startswith(prompt_text):
                        assistant_response = assistant_response[len(prompt_text):].strip()
                    
                    # Look for program and answer tags
                    program_pattern = r'<begin_of_program>(.*?)<end_of_program>'
                    program_match = re.search(program_pattern, assistant_response, re.DOTALL)
                    
                    answer_pattern = r'<begin_of_answer>(.*?)<end_of_answer>'
                    answer_match = re.search(answer_pattern, assistant_response, re.DOTALL)
                    
                    if program_match and answer_match:
                        program_content = program_match.group(1).strip()
                        answer_content = answer_match.group(1).strip()
                        
                        clean_response = (
                            f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                            f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
                        )
                        assistant_response = clean_response
                    
                    # If no tags found, try to extract from the full output
                    if not is_valid_output(assistant_response):
                        program_match = re.search(program_pattern, decoded_output, re.DOTALL)
                        answer_match = re.search(answer_pattern, decoded_output, re.DOTALL)
                        
                        if program_match and answer_match:
                            program_content = program_match.group(1).strip()
                            answer_content = answer_match.group(1).strip()
                            
                            clean_response = (
                                f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                                f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
                            )
                            assistant_response = clean_response
                # For Qwen models, use their specific format
                elif is_qwen:
                    # Extract using Qwen's specific format
                    assistant_pattern = r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)'
                    assistant_matches = re.findall(assistant_pattern, decoded_output, re.DOTALL)
                    
                    if assistant_matches:
                        # Use the last assistant response
                        assistant_response = assistant_matches[-1].strip()
                        print(f"Extracted Qwen assistant response (first 50 chars): {assistant_response[:50]}...")
                        
                        # Now look for program and answer tags within the assistant response
                        program_pattern = r'<begin_of_program>(.*?)<end_of_program>'
                        program_match = re.search(program_pattern, assistant_response, re.DOTALL)
                        
                        answer_pattern = r'<begin_of_answer>(.*?)<end_of_answer>'
                        answer_match = re.search(answer_pattern, assistant_response, re.DOTALL)
                        
                        if program_match and answer_match:
                            # If both tags are found, create a clean response with just these elements
                            program_content = program_match.group(1).strip()
                            answer_content = answer_match.group(1).strip()
                            
                            clean_response = (
                                f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                                f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
                            )
                            assistant_response = clean_response
                            print(f"Extracted program and answer from Qwen response: {assistant_response[:100]}...")
                    else:
                        # Fallback to standard extraction
                        assistant_response = decoded_output
                else:
                    # First try to extract using program and answer tags directly
                    program_pattern = r'<begin_of_program>(.*?)<end_of_program>'
                    program_match = re.search(program_pattern, decoded_output, re.DOTALL)
                    
                    answer_pattern = r'<begin_of_answer>(.*?)<end_of_answer>'
                    answer_match = re.search(answer_pattern, decoded_output, re.DOTALL)
                    
                    if program_match and answer_match:
                        # If both tags are found, create a clean response with just these elements
                        program_content = program_match.group(1).strip()
                        answer_content = answer_match.group(1).strip()
                        
                        clean_response = (
                            f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                            f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
                        )
                        assistant_response = clean_response
                    else:
                        # If tags aren't found, try to extract the assistant's response using the chat template
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
                            
                        # Try to clean up the response by removing system prompt and user message
                        if "system" in assistant_response.lower() and "user" in assistant_response.lower():
                            # Look for program and answer tags in the messy output
                            program_match = re.search(program_pattern, assistant_response, re.DOTALL)
                            answer_match = re.search(answer_pattern, assistant_response, re.DOTALL)
                            
                            if program_match and answer_match:
                                program_content = program_match.group(1).strip()
                                answer_content = answer_match.group(1).strip()
                                
                                clean_response = (
                                    f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                                    f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
                                )
                                assistant_response = clean_response
                
                # Check if the output is valid
                if not is_valid_output(assistant_response):
                    # Try to find operation tokens directly
                    operation_pattern = r'(add|subtract|multiply|divide)\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)'
                    match = re.search(operation_pattern, decoded_output, re.IGNORECASE)
                    
                    if match:
                        operation = match.group(1).lower()
                        num1 = match.group(2)
                        num2 = match.group(3)
                        
                        # Try to calculate the answer
                        try:
                            if operation == "add":
                                answer = str(float(num1) + float(num2))
                            elif operation == "subtract":
                                answer = str(float(num1) - float(num2))
                            elif operation == "multiply":
                                answer = str(float(num1) * float(num2))
                            elif operation == "divide":
                                answer = str(float(num1) / float(num2))
                            else:
                                answer = "N/A"
                                
                            # Create a valid output
                            assistant_response = (
                                f"<begin_of_program>\n{operation}( {num1} {num2} ) EOF\n<end_of_program>\n\n"
                                f"<begin_of_answer>\n{answer}\n<end_of_answer>"
                            )
                        except:
                            pass
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
        
        # Check if we're dealing with a PEFT adapter model
        is_peft_model = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
        
        if is_peft_model:
            print("Detected PEFT adapter model. Loading base model first...")
            
            # Check if PEFT is installed
            try:
                from peft import PeftModel, PeftConfig
            except ImportError:
                raise ImportError(
                    "PEFT library is required to load adapter models but was not found. "
                    "Please install it with: pip install peft"
                )
            
            # Get the base model name, either from the argument or from the adapter config
            if args.base_model_name:
                base_model_name = args.base_model_name
                print(f"Using provided base model: {base_model_name}")
            else:
                # Get the adapter config to find the base model name
                peft_config = PeftConfig.from_pretrained(args.model_path)
                base_model_name = peft_config.base_model_name_or_path
                print(f"Using base model from adapter config: {base_model_name}")
            
            # Load the base model
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
                    base_model_name,
                    device_map="auto",
                    quantization_config=bnb_config
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(base_model_name).to(args.device)
            
            # Load the adapter on top of the base model
            model = PeftModel.from_pretrained(model, args.model_path)
            
            # Get tokenizer from the base model
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            # Load regular model
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
        
        # Ensure proper padding settings
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Prepare test data
        print(f"Preparing test data from {args.test_data}")
        test_dataset, example_ids = prepare_test_data(args.test_data)
        
        # Debug: Print the structure of the first example
        if len(test_dataset) > 0:
            print("\nDebug - First example structure:")
            first_example = test_dataset[0]
            if isinstance(first_example, dict):
                for key, value in first_example.items():
                    if isinstance(value, dict):
                        print(f"{key}: {type(value)} with keys {list(value.keys())}")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"{key}: list of length {len(value)}, first item type: {type(value[0])}")
                    else:
                        print(f"{key}: {type(value)}")
            else:
                print(f"Example features: {test_dataset.features}")
                print(f"Example keys: {test_dataset[0].keys()}")
        
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
        try:
            formatted_predictions = format_predictions_for_evaluation(predictions, example_ids)
            
            # Save formatted predictions
            formatted_predictions_file = os.path.join(args.output_dir, "formatted_predictions.json")
            with open(formatted_predictions_file, 'w') as f:
                json.dump(formatted_predictions, f, indent=2)
            
            print(f"Formatted predictions saved to {formatted_predictions_file}")
        except Exception as e:
            error_msg = f"Error during formatting predictions: {str(e)}"
            print(error_msg)
            with open(log_file, 'a') as f:
                f.write(f"{error_msg}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
            # Continue with a default empty formatted_predictions
            formatted_predictions = [{"id": id, "predicted": "", "gold": ""} for id in example_ids]
        
        # Create final.json with the specified format
        final_predictions = []
        for i, example_id in enumerate(example_ids):
            try:
                # Get the question from the test dataset
                question = ""
                if isinstance(test_dataset[i], dict):
                    if "question" in test_dataset[i]:
                        question = test_dataset[i].get("question", "")
                    elif "qa" in test_dataset[i]:
                        # Check if qa is a string (JSON)
                        if isinstance(test_dataset[i]["qa"], str):
                            try:
                                qa_dict = json.loads(test_dataset[i]["qa"])
                                if isinstance(qa_dict, dict):
                                    if "question" in qa_dict:
                                        question = qa_dict["question"]
                                    # Handle entries list from qa_0, qa_1 etc.
                                    elif "entries" in qa_dict and isinstance(qa_dict["entries"], list) and len(qa_dict["entries"]) > 0:
                                        first_entry = qa_dict["entries"][0]
                                        if isinstance(first_entry, dict) and "question" in first_entry:
                                            question = first_entry["question"]
                            except json.JSONDecodeError:
                                pass
                        # Check if qa is a dict
                        elif isinstance(test_dataset[i]["qa"], dict) and "question" in test_dataset[i]["qa"]:
                            question = test_dataset[i]["qa"]["question"]
                else:
                    if "question" in test_dataset.column_names:
                        question = test_dataset[i]["question"]
                    elif "qa" in test_dataset.column_names:
                        # Check if qa is a string (JSON)
                        if isinstance(test_dataset[i]["qa"], str):
                            try:
                                qa_dict = json.loads(test_dataset[i]["qa"])
                                if isinstance(qa_dict, dict):
                                    if "question" in qa_dict:
                                        question = qa_dict["question"]
                                    # Handle entries list from qa_0, qa_1 etc.
                                    elif "entries" in qa_dict and isinstance(qa_dict["entries"], list) and len(qa_dict["entries"]) > 0:
                                        first_entry = qa_dict["entries"][0]
                                        if isinstance(first_entry, dict) and "question" in first_entry:
                                            question = first_entry["question"]
                            except json.JSONDecodeError:
                                pass
                        # Check if qa is a dict
                        elif isinstance(test_dataset[i]["qa"], dict) and "question" in test_dataset[i]["qa"]:
                            question = test_dataset[i]["qa"]["question"]
                
                # Ensure question is a string
                question = str(question) if question is not None else ""
                
                # Extract the answer from the prediction
                answer = extract_answer(predictions[i])
                # Ensure answer is a string
                answer = str(answer) if answer is not None else ""
                
                # Try to get the golden answer from the test dataset
                golden_answer = ""
                if isinstance(test_dataset[i], dict):
                    if "answer" in test_dataset[i]:
                        golden_answer = str(test_dataset[i]["answer"]) if test_dataset[i]["answer"] is not None else ""
                    elif "qa" in test_dataset[i]:
                        # Check if qa is a string (JSON)
                        if isinstance(test_dataset[i]["qa"], str):
                            try:
                                qa_dict = json.loads(test_dataset[i]["qa"])
                                if isinstance(qa_dict, dict) and "answer" in qa_dict:
                                    golden_answer = str(qa_dict["answer"]) if qa_dict["answer"] is not None else ""
                            except json.JSONDecodeError:
                                pass
                        # Check if qa is a dict
                        elif isinstance(test_dataset[i]["qa"], dict) and "answer" in test_dataset[i]["qa"]:
                            golden_answer = str(test_dataset[i]["qa"]["answer"]) if test_dataset[i]["qa"]["answer"] is not None else ""
                else:
                    if "answer" in test_dataset.column_names:
                        golden_answer = str(test_dataset[i]["answer"]) if test_dataset[i]["answer"] is not None else ""
                    elif "qa" in test_dataset.column_names:
                        # Check if qa is a string (JSON)
                        if isinstance(test_dataset[i]["qa"], str):
                            try:
                                qa_dict = json.loads(test_dataset[i]["qa"])
                                if isinstance(qa_dict, dict) and "answer" in qa_dict:
                                    golden_answer = str(qa_dict["answer"]) if qa_dict["answer"] is not None else ""
                            except json.JSONDecodeError:
                                pass
                        # Check if qa is a dict
                        elif isinstance(test_dataset[i]["qa"], dict) and "answer" in test_dataset[i]["qa"]:
                            golden_answer = str(test_dataset[i]["qa"]["answer"]) if test_dataset[i]["qa"]["answer"] is not None else ""
                
                # Create the final prediction entry
                final_prediction = {
                    "qa": {
                        "id": example_id,
                        "question": question,
                        "answer": answer,
                        "golden_answer": golden_answer
                    }
                }
                
                final_predictions.append(final_prediction)
            except Exception as e:
                print(f"Error creating final prediction for example {example_id}: {e}")
                # Add a placeholder entry
                final_predictions.append({
                    "qa": {
                        "id": example_id,
                        "question": "",
                        "answer": "",
                        "golden_answer": ""
                    }
                })
        
        # Save final predictions
        final_predictions_file = os.path.join(args.output_dir, "final.json")
        with open(final_predictions_file, 'w') as f:
            json.dump(final_predictions, f, indent=2)
        
        print(f"Final predictions saved to {final_predictions_file}")
        
        # Print sample predictions
        print("\nSample predictions:")
        for i in range(min(3, len(predictions))):
            print(f"Example ID: {example_ids[i]}")
            print(f"Raw prediction: {predictions[i][:100]}...")
            print(f"Extracted program: {formatted_predictions[i]['predicted']}")
            print(f"Final format:")
            print(f"  - Question: {final_predictions[i]['qa']['question']}")
            print(f"  - Predicted Answer: {final_predictions[i]['qa']['answer']}")
            print(f"  - Golden Answer: {final_predictions[i]['qa']['golden_answer']}")
            print()
        
        # Log completion
        with open(log_file, 'a') as f:
            f.write(f"Inference completed at: {datetime.datetime.now()}\n")
            f.write(f"Total examples processed: {len(predictions)}\n")
            f.write(f"Results saved to: {args.output_dir}\n")
            f.write(f"Output files: raw_predictions.json, formatted_predictions.json, final.json\n")
    
    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(f"{error_msg}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")
        raise


if __name__ == "__main__":
    main()