import json
import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import re
from bs4 import BeautifulSoup
import huggingface_hub


def format_table_for_llm(html_table):
    """Convert HTML table to a markdown-style text table that's easier for LLMs to understand"""
    soup = BeautifulSoup(html_table, 'html.parser')
    
    # Extract all rows
    rows = soup.find_all('tr')
    if not rows:
        return "No table data found."
    
    # Process the table data
    table_data = []
    for row in rows:
        cells = row.find_all('td')
        # Skip the first cell if it's just a row number
        if cells and cells[0].text.strip().isdigit():
            row_data = [cell.text.strip() for cell in cells[1:]]
        else:
            row_data = [cell.text.strip() for cell in cells]
        table_data.append(row_data)
    
    # Format as a clean text table
    if not table_data:
        return "Empty table."
    
    # Determine column widths for proper alignment
    col_widths = [max(len(row[i]) for row in table_data if i < len(row)) 
                    for i in range(max(len(row) for row in table_data))]
    
    # Create header row with column names
    header_row = table_data[0]
    header_separator = ['-' * width for width in col_widths]
    
    # Format the table with proper alignment
    formatted_table = []
    formatted_table.append(' | '.join(header_row[i].ljust(col_widths[i]) for i in range(len(header_row))))
    formatted_table.append(' | '.join(header_separator[i] for i in range(len(header_separator))))
    
    # Add data rows
    for row in table_data[1:]:
        formatted_table.append(' | '.join(row[i].ljust(col_widths[i]) if i < len(row) else ' ' * col_widths[i] 
                                        for i in range(len(col_widths))))
    
    return '\n'.join(formatted_table)


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


def get_convfinqa_dataset(json_file):
    """
    Convert ConvFinQA dataset to a format that includes both program tokens and answers.
    Uses <begin_of_step> and <begin_of_answer> tags.
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
        annos = sample['annotation']

        # Extract context information
        pre_text = annos.get('amt_pre_text', '')
        post_text = annos.get('amt_post_text', '')
        html_table = annos.get('amt_table', '').replace('<td></td>', '<td>-</td>')
        formatted_table = format_table_for_llm(html_table)
        
        # Create the financial context
        financial_context = f'{pre_text}\n\n{formatted_table}\n\n{post_text}\n'
        
        # Get dialogue components
        questions = annos.get('dialogue_break', [])
        answers = annos.get('exe_ans_list', [])
        
        # Get programs if available
        programs = annos.get('program_list', [])
        if not programs and 'turn_program' in annos:
            programs = annos.get('turn_program', [])
              
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
        
        conversations.append({
            "system": system_prompt,
            "conversations": conversation
        })
    
    return Dataset.from_list(conversations)


def prepare_convfinqa_datasets(data_dir="data", output_dir="processed_data"):
    """
    Prepare ConvFinQA datasets for training, validation, and testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train, dev, and test sets
    train_file = os.path.join(data_dir, "train.json")
    dev_file = os.path.join(data_dir, "dev.json")
    test_file = os.path.join(data_dir, "test_private.json")
    
    train_dataset = get_convfinqa_dataset(train_file) if os.path.exists(train_file) else None
    dev_dataset = get_convfinqa_dataset(dev_file) if os.path.exists(dev_file) else None
    test_dataset = get_convfinqa_dataset(test_file) if os.path.exists(test_file) else None
    
    # Create dataset dictionary
    dataset_dict = {}
    if train_dataset:
        dataset_dict["train"] = train_dataset
    if dev_dataset:
        dataset_dict["validation"] = dev_dataset
    if test_dataset:
        dataset_dict["test"] = test_dataset
    
    # Save processed datasets
    dataset = DatasetDict(dataset_dict)
    dataset.save_to_disk(os.path.join(output_dir, "convfinqa_sharegpt_refine"))
    
    print(f"Processed datasets saved to {os.path.join(output_dir, 'convfinqa_sharegpt_refine')}")
    return dataset


if __name__ == "__main__":
    dataset_out = prepare_convfinqa_datasets() 
    # dataset_out.push_to_hub("christlurker/convfinqa_sharegpt_refine")
