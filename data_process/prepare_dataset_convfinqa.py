import json
import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import re
from bs4 import BeautifulSoup
import huggingface_hub
import argparse


def format_table_for_llm(table_data):
    """
    Convert table data to a markdown-style text table that's easier for LLMs to understand.
    
    Args:
        table_data: List of lists representing table rows and columns or HTML table string
        
    Returns:
        Formatted table as a string
    """
    # Check if input is an HTML string
    if isinstance(table_data, str) and ("<table" in table_data.lower() or "<tr" in table_data.lower()):
        return format_html_table_for_llm(table_data)
    
    # Handle list-based table data
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
    
    # Use the list-based table formatter
    return format_table_for_llm(table_data)


def format_program_tokens(program_str):
    """
    Convert program string to a list of tokens.
    Example: "subtract(5829, 5735)" -> ["subtract(", "5829", "5735", ")", "EOF"]
    """
    if not program_str or program_str.lower() == "none":
        return ["EOF"]
    
    # Split by commas and handle parentheses
    tokens = []
    current_token = ""
    
    for char in program_str:
        if char == '(':
            tokens.append(current_token + '(')
            current_token = ""
        elif char == ')':
            if current_token:
                tokens.append(current_token.strip())
                current_token = ""
            tokens.append(')')
        elif char == ',':
            if current_token:
                tokens.append(current_token.strip())
                current_token = ""
        else:
            current_token += char
    
    if current_token:
        tokens.append(current_token.strip())
    
    # Add EOF token
    tokens.append("EOF")
    
    return tokens


def program_tokens_to_string(tokens):
    """
    Convert program tokens to a readable string format for training.
    Example: ["subtract(", "5829", "5735", ")", "EOF"] -> "subtract( 5829 5735 ) EOF"
    """
    return " ".join(tokens)


def clean_model_output(model_output):
    """
    Clean up model output to extract only the program and answer tags.
    
    Args:
        model_output (str): Raw model output
        
    Returns:
        str: Cleaned output with only program and answer tags
    """
    # Extract program section
    program_pattern = r'<begin_of_program>(.*?)<end_of_program>'
    program_match = re.search(program_pattern, model_output, re.DOTALL)
    program_content = program_match.group(1).strip() if program_match else "EOF"
    
    # Extract answer section
    answer_pattern = r'<begin_of_answer>(.*?)<end_of_answer>'
    answer_match = re.search(answer_pattern, model_output, re.DOTALL)
    answer_content = answer_match.group(1).strip() if answer_match else "N/A"
    
    # Format clean output
    clean_output = (
        f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
        f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
    )
    
    return clean_output


def get_convfinqa_dataset(json_file, is_test=False):
    """
    Convert ConvFinQA dataset to a format that includes both program tokens and answers.
    Uses <begin_of_program> and <begin_of_answer> tags.
    
    Args:
        json_file: Path to the JSON file containing the dataset
        is_test: Whether this is test data (affects how we handle missing answers/programs)
    
    Returns:
        Dataset object containing the processed conversations
    """
    conversations = []
    
    # System prompt for structured thinking and program generation
    system_prompt = """Your role is to solve financial questions by generating both the program tokens that represent the calculation and the final answer. 
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

Your response should ONLY contain the program tokens and answer within their respective tags.
"""
    
    samples = json.load(open(json_file))
    for sample in samples:
        # Extract context information based on data format
        pre_text = ""
        post_text = ""
        formatted_table = ""
        questions = []
        answers = []
        programs = []
        
        # First check if the sample has the annotation field (higher priority)
        if "annotation" in sample:
            annos = sample['annotation']
            pre_text = annos.get('amt_pre_text', '')
            post_text = annos.get('amt_post_text', '')
            html_table = annos.get('amt_table', '').replace('<td></td>', '<td>-</td>')
            formatted_table = format_html_table_for_llm(html_table)
            
            # Get dialogue components
            questions = annos.get('dialogue_break', [])
            
            # Get answers if available
            if 'exe_ans_list' in annos:
                answers = annos.get('exe_ans_list', [])
            
            # Get programs if available
            if 'program_list' in annos:
                programs = annos.get('program_list', [])
            elif 'turn_program' in annos:
                programs = annos.get('turn_program', [])
        
        # If annotation doesn't have the required fields or doesn't exist, 
        # check if the sample has the direct fields
        if not questions or not answers or not programs:
            # Check if the sample has the new format with pre_text, post_text, and table_ori
            if "pre_text" in sample and "post_text" in sample and "table_ori" in sample:
                # Only update these fields if they weren't set from annotation
                if not pre_text:
                    pre_text = ' '.join(sample.get("pre_text", []))
                
                if not post_text:
                    post_text = ' '.join(sample.get("post_text", []))
                
                if not formatted_table:
                    table_data = sample.get("table_ori", [])
                    formatted_table = format_table_for_llm(table_data)
                
                # Get questions and answers if not already set
                if not questions:
                    questions = sample.get("questions", [])
                    # If no questions are provided but a single question is available
                    if not questions and "question" in sample:
                        questions = [sample["question"]]
                
                if not answers:
                    answers = sample.get("answers", [])
                
                if not programs:
                    programs = sample.get("programs", [])
        
        # For test data, we might only have a single question without answers/programs
        if is_test and not questions and "question" in sample:
            questions = [sample["question"]]
        
        # Skip if no questions are available
        if not questions:
            continue
        
        # Create the financial context
        financial_context = f'{pre_text}\n\n{formatted_table}\n\n{post_text}\n'
        
        # Create conversation with proper turns
        conversation = []
        
        # Process each turn
        for i in range(len(questions)):
            # First turn includes the financial context
            if i == 0:
                question_text = f"I'm looking at some financial data. Here's the context:\n\n{financial_context}\n\n{questions[i]}"
            else:
                # For subsequent turns, just include the question
                question_text = questions[i]
            
            conversation.append({"role": "human", "content": question_text})
            
            # Skip assistant response for test data if no answers/programs are available
            if is_test and (i >= len(answers) or not answers):
                continue
            
            # Get program tokens for this turn
            program_tokens = []
            if i < len(programs) and programs[i]:
                program_tokens = format_program_tokens(programs[i])
            else:
                # If no program is available, use a placeholder
                program_tokens = ["EOF"]
            
            # Format program tokens as a string
            program_str = program_tokens_to_string(program_tokens)
            
            # Get the answer and ensure it's properly formatted
            answer = answers[i] if i < len(answers) else 'N/A'
            # Try to convert to float and back to string to ensure numerical format
            try:
                # Remove any non-numeric characters except decimal point and minus sign
                answer_str = str(answer).replace('$', '').replace('%', '').replace(',', '')
                # If it can be converted to float, use the clean numeric format
                float_answer = float(answer_str)
                answer = str(float_answer)
            except (ValueError, TypeError):
                # If not convertible to float, use as is
                pass
                        
            # Format the assistant's response with all components
            assistant_response = (
                f"<begin_of_program>\n{program_str}\n<end_of_program>\n\n"
                f"<begin_of_answer>\n{answer}\n<end_of_answer>"
            )
            
            conversation.append({"role": "assistant", "content": assistant_response})
        
        # Only add conversations that have at least one complete turn
        if len(conversation) >= 2 or (is_test and len(conversation) >= 1):
            conversations.append({
                "system": system_prompt,
                "conversations": conversation
            })
    
    return Dataset.from_list(conversations)


def prepare_convfinqa_datasets(data_dir="data", output_dir="processed_data", include_test=False):
    """
    Prepare ConvFinQA datasets for training, validation, and testing.
    
    Args:
        data_dir: Directory containing the data files
        output_dir: Directory to save the processed datasets
        include_test: Whether to include test data in the dataset (default: False)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train and dev sets
    train_file = os.path.join(data_dir, "train.json")
    dev_file = os.path.join(data_dir, "dev.json")
    test_file = os.path.join(data_dir, "test_private.json")
    
    # Process training data
    train_dataset = None
    if os.path.exists(train_file):
        print(f"Processing training data from {train_file}")
        try:
            train_dataset = get_convfinqa_dataset(train_file, is_test=False)
            print(f"Successfully processed training data with {len(train_dataset)} examples")
        except Exception as e:
            print(f"Error processing training data: {e}")
            print("Training data will not be included in the dataset")
    else:
        print(f"Training file not found: {train_file}")
    
    # Process validation data
    dev_dataset = None
    if os.path.exists(dev_file):
        print(f"Processing validation data from {dev_file}")
        try:
            dev_dataset = get_convfinqa_dataset(dev_file, is_test=False)
            print(f"Successfully processed validation data with {len(dev_dataset)} examples")
        except Exception as e:
            print(f"Error processing validation data: {e}")
            print("Validation data will not be included in the dataset")
    else:
        print(f"Validation file not found: {dev_file}")
    
    # Only process test data if explicitly requested
    test_dataset = None
    if include_test and os.path.exists(test_file):
        print(f"Processing test data from {test_file}")
        try:
            test_dataset = get_convfinqa_dataset(test_file, is_test=True)
            print(f"Successfully processed test data with {len(test_dataset)} examples")
        except Exception as e:
            print(f"Error processing test data: {e}")
            print("Test data will not be included in the dataset")
    elif include_test:
        print(f"Test file not found: {test_file}")
    
    # Create dataset dictionary
    dataset_dict = {}
    if train_dataset:
        dataset_dict["train"] = train_dataset
        print(f"Train dataset: {len(train_dataset)} examples")
    if dev_dataset:
        dataset_dict["validation"] = dev_dataset
        print(f"Validation dataset: {len(dev_dataset)} examples")
    if test_dataset:
        dataset_dict["test"] = test_dataset
        print(f"Test dataset: {len(test_dataset)} examples")
    
    if not dataset_dict:
        print("No datasets were successfully processed. Check the input files and try again.")
        return None
    
    # Save processed datasets
    dataset = DatasetDict(dataset_dict)
    output_path = os.path.join(output_dir, "convfinqa_sharegpt_refine")
    dataset.save_to_disk(output_path)
    
    print(f"Processed datasets saved to {output_path}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ConvFinQA multi-turn datasets for training")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the data files")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Directory to save the processed datasets")
    parser.add_argument("--include_test", action="store_true", help="Whether to include test data in the dataset")
    
    args = parser.parse_args()
    
    dataset_out = prepare_convfinqa_datasets(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        include_test=args.include_test
    )
    # Uncomment to push to HuggingFace Hub
    # dataset_out.push_to_hub("christlurker/convfinqa_sharegpt_refine")
