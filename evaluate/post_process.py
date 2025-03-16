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
    # Check if this is a Qwen model output and extract the assistant response first
    if "<|im_start|>" in text and "<|im_end|>" in text:
        # Extract the assistant's response
        assistant_pattern = r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)'
        assistant_matches = re.findall(assistant_pattern, text, re.DOTALL)
        
        if assistant_matches:
            # Use the last assistant response
            text = assistant_matches[-1].strip()
    
    # First try to find the program section
    program_pattern = r'<begin_of_program>\s*(.*?)\s*<end_of_program>'
    match = re.search(program_pattern, text, re.DOTALL)
    
    if match:
        program_text = match.group(1).strip()
        
        # Check if the program text contains operation tokens
        operation_pattern = r'(add|subtract|multiply|divide)\s*\('
        if re.search(operation_pattern, program_text):
            # Split by whitespace to get tokens
            tokens = []
            # Process tokens to handle parentheses correctly
            in_operation = False
            
            # First, normalize spacing around parentheses and ensure proper spacing
            program_text = re.sub(r'\(\s+', '(', program_text)  # Remove space after opening parenthesis
            program_text = re.sub(r'\s+\)', ')', program_text)  # Remove space before closing parenthesis
            program_text = re.sub(r'([a-zA-Z]+)\(', r'\1 (', program_text)  # Add space between operation and (
            
            # Split by whitespace
            raw_tokens = program_text.split()
            
            i = 0
            while i < len(raw_tokens):
                token = raw_tokens[i]
                
                # Skip system prompt related tokens
                if "system" in token.lower() or "user" in token.lower() or "assistant" in token.lower():
                    i += 1
                    continue
                # Skip instruction related tokens
                if "[inst]" in token.lower() or "[/inst]" in token.lower():
                    i += 1
                    continue
                
                # Handle operation names
                if token.lower() in ["add", "subtract", "multiply", "divide"]:
                    # Check if next token is a parenthesis
                    if i + 1 < len(raw_tokens) and raw_tokens[i + 1] == "(":
                        tokens.append(token.lower() + "(")
                        i += 2  # Skip the operation and the opening parenthesis
                        in_operation = True
                    else:
                        tokens.append(token.lower() + "(")  # Always append with parenthesis
                        i += 1
                # Handle opening parenthesis
                elif token == "(" and i > 0 and raw_tokens[i-1].lower() in ["add", "subtract", "multiply", "divide"]:
                    # Already handled with the operation
                    i += 1
                # Handle standalone opening parenthesis
                elif token == "(":
                    # Check if previous token was an operation
                    if i > 0 and raw_tokens[i-1].lower() in ["add", "subtract", "multiply", "divide"]:
                        # Already handled with the operation
                        i += 1
                    else:
                        tokens.append(token)
                        i += 1
                        in_operation = True
                # Handle closing parenthesis
                elif token == ")":
                    tokens.append(token)
                    i += 1
                    in_operation = False
                # Handle tokens with attached parentheses
                elif "(" in token or ")" in token:
                    # Check if it's an operation with attached parenthesis
                    op_match = re.match(r'(add|subtract|multiply|divide)(\(.*)', token, re.IGNORECASE)
                    if op_match:
                        tokens.append(op_match.group(1).lower() + "(")
                        
                        # Process the rest of the token
                        rest = op_match.group(2)[1:]  # Remove the opening parenthesis
                        if rest:
                            # Split by any remaining parentheses
                            parts = []
                            current_part = ""
                            for char in rest:
                                if char in "()":
                                    if current_part:
                                        parts.append(current_part)
                                        current_part = ""
                                    parts.append(char)
                                else:
                                    current_part += char
                            if current_part:
                                parts.append(current_part)
                            
                            # Add each part as a separate token
                            for part in parts:
                                if part == "(":
                                    in_operation = True
                                elif part == ")":
                                    in_operation = False
                                    tokens.append(part)
                                else:
                                    tokens.append(part)
                    else:
                        # Split the token by parentheses
                        parts = []
                        current_part = ""
                        for char in token:
                            if char in "()":
                                if current_part:
                                    parts.append(current_part)
                                    current_part = ""
                                parts.append(char)
                            else:
                                current_part += char
                        if current_part:
                            parts.append(current_part)
                        
                        # Add each part as a separate token
                        for part in parts:
                            if part == "(":
                                in_operation = True
                            elif part == ")":
                                in_operation = False
                            tokens.append(part)
                    
                    i += 1
                # Handle reference numbers (with # symbol)
                elif token.startswith("#"):
                    tokens.append(token)
                    i += 1
                # Handle regular numbers and other tokens
                else:
                    tokens.append(token)
                    i += 1
            
            # Ensure the last token is EOF
            if tokens and tokens[-1].upper() != "EOF":
                tokens.append("EOF")
            
            # If we have no valid tokens, return just EOF
            if not tokens:
                return ["EOF"]
                
            return tokens
        else:
            # If no operation pattern found, try to extract any meaningful tokens
            # Look for operation tokens in a more flexible way
            operation_words = ["add", "subtract", "multiply", "divide"]
            for op in operation_words:
                if op in program_text.lower():
                    # Try to reconstruct a valid operation
                    parts = program_text.lower().split(op)
                    if len(parts) > 1:
                        # Find numbers after the operation
                        number_pattern = r'[-+]?\d*\.?\d+'
                        numbers = re.findall(number_pattern, parts[1])
                        if len(numbers) >= 2:
                            # Construct a valid operation token sequence
                            return [f"{op}(", numbers[0], numbers[1], ")", "EOF"]
            
            # If still no operation found, try to extract any tokens
            tokens = program_text.split()
            
            # Filter out any non-program tokens
            valid_tokens = []
            for token in tokens:
                # Skip system prompt related tokens
                if "system" in token.lower() or "user" in token.lower() or "assistant" in token.lower():
                    continue
                # Skip instruction related tokens
                if "[inst]" in token.lower() or "[/inst]" in token.lower():
                    continue
                valid_tokens.append(token)
            
            # Ensure the last token is EOF
            if valid_tokens and valid_tokens[-1].upper() != "EOF":
                valid_tokens.append("EOF")
            
            # If we have no valid tokens, return just EOF
            if not valid_tokens:
                return ["EOF"]
                
            return valid_tokens
    
    # If no program found, try to find operation tokens directly in the text
    operation_pattern = r'(add|subtract|multiply|divide)\s*\(\s*([-+]?\d*\.?\d+)\s*([-+]?\d*\.?\d+)\s*\)'
    match = re.search(operation_pattern, text, re.IGNORECASE)
    if match:
        operation = match.group(1).lower()
        num1 = match.group(2)
        num2 = match.group(3)
        return [f"{operation}(", num1, num2, ")", "EOF"]
    
    # If still nothing found, return just EOF
    return ["EOF"]


def extract_answer(text):
    """
    Extract the final answer from model output.
    Looks for content between <begin_of_answer> and <end_of_answer> tags.
    """
    # Check if this is a Qwen model output and extract the assistant response first
    if "<|im_start|>" in text and "<|im_end|>" in text:
        # Extract the assistant's response
        assistant_pattern = r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)'
        assistant_matches = re.findall(assistant_pattern, text, re.DOTALL)
        
        if assistant_matches:
            # Use the last assistant response
            text = assistant_matches[-1].strip()
    
    # Look for answer tags
    answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
    match = re.search(answer_pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no answer tags found, try to find numerical answers in the text
    # Look for patterns like "The answer is X" or "= X"
    alt_patterns = [
        r'(?:answer|result|value)(?:\s+is|\s*[:=])\s*([-+]?\d*\.?\d*)',
        r'(?:=|equals)\s*([-+]?\d*\.?\d*)',
        r'(?:[\$£€])\s*([-+]?\d*\.?\d*)'
    ]
    
    for pattern in alt_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return "N/A"


def clean_model_output(text):
    """
    Clean up model output to extract only the program and answer tags.
    Removes system prompt, user messages, and other extraneous text.
    """
    # First, check if the text contains system/user messages
    contains_system = "system" in text.lower()
    contains_user = "user" in text.lower()
    
    # Check if this is a Qwen model output
    is_qwen_output = "<|im_start|>" in text and "<|im_end|>" in text
    
    # For Qwen models, extract the assistant response first
    if is_qwen_output:
        # Extract the assistant's response
        assistant_pattern = r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)'
        assistant_matches = re.findall(assistant_pattern, text, re.DOTALL)
        
        if assistant_matches:
            # Use the last assistant response
            text = assistant_matches[-1].strip()
            print(f"Extracted Qwen assistant response for cleaning (first 50 chars): {text[:50]}...")
            
            # Now look for program and answer tags within the assistant response
            program_pattern = r'<begin_of_program>(.*?)<end_of_program>'
            program_match = re.search(program_pattern, text, re.DOTALL)
            
            answer_pattern = r'<begin_of_answer>(.*?)<end_of_answer>'
            answer_match = re.search(answer_pattern, text, re.DOTALL)
            
            if program_match and answer_match:
                # If both tags are found, create a clean response with just these elements
                program_content = program_match.group(1).strip()
                answer_content = answer_match.group(1).strip()
                
                clean_output = (
                    f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                    f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
                )
                return clean_output
    
    # Extract program section
    program_pattern = r'<begin_of_program>(.*?)<end_of_program>'
    program_match = re.search(program_pattern, text, re.DOTALL)
    
    # Extract answer section
    answer_pattern = r'<begin_of_answer>(.*?)<end_of_answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    # If we found both program and answer sections
    if program_match and answer_match:
        program_content = program_match.group(1).strip()
        answer_content = answer_match.group(1).strip()
        
        # Check if program content contains operation tokens
        operation_pattern = r'(add|subtract|multiply|divide)\s*\('
        if not re.search(operation_pattern, program_content) and contains_system:
            # Try to find operation tokens in the text
            operation_match = re.search(operation_pattern, text, re.IGNORECASE)
            if operation_match:
                # Extract a better program section
                op_start = text.find(operation_match.group(0))
                op_end = text.find("EOF", op_start)
                if op_end > op_start:
                    # Extract the operation and surrounding content
                    better_program = text[op_start:op_end+3].strip()
                    program_content = better_program
        
        # Format clean output
        clean_output = (
            f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
            f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
        )
        
        return clean_output
    
    # If we didn't find both sections, try to extract them from the text
    if contains_system or contains_user:
        # This is likely a chat template output
        # Try to find operation tokens with their arguments
        operation_pattern = r'(add|subtract|multiply|divide)\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)'
        operation_match = re.search(operation_pattern, text, re.IGNORECASE)
        
        if operation_match:
            # Extract the full operation
            operation = operation_match.group(1).lower()
            num1 = operation_match.group(2)
            num2 = operation_match.group(3)
            program_content = f"{operation}( {num1} {num2} ) EOF"
                
            # Try to find numerical answer
            numerical_pattern = r'<begin_of_answer>\s*(-?\d+\.?\d*)\s*<end_of_answer>'
            numerical_match = re.search(numerical_pattern, text, re.DOTALL)
            
            if numerical_match:
                answer_content = numerical_match.group(1).strip()
            else:
                # Try alternative patterns for numerical answers
                alt_patterns = [
                    r'(?:answer|result|value)(?:\s+is|\s*[:=])\s*(-?\d+\.?\d*)',
                    r'(?:=|equals)\s*(-?\d+\.?\d*)',
                    r'(?:[\$£€])\s*(-?\d+\.?\d*)',
                    r'<begin_of_answer>.*?(\d+\.?\d*).*?<end_of_answer>',
                    r'(\d+\.?\d*)'  # Last resort: find any number
                ]
                
                for pattern in alt_patterns:
                    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        answer_content = match.group(1).strip()
                        break
                else:
                    # If no numerical answer found, try to calculate it
                    try:
                        if operation == "add":
                            answer_content = str(float(num1) + float(num2))
                        elif operation == "subtract":
                            answer_content = str(float(num1) - float(num2))
                        elif operation == "multiply":
                            answer_content = str(float(num1) * float(num2))
                        elif operation == "divide":
                            answer_content = str(float(num1) / float(num2))
                        else:
                            answer_content = "N/A"
                    except:
                        answer_content = "N/A"
            
            # Format clean output
            clean_output = (
                f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
            )
            
            return clean_output
        
        # Try a more flexible pattern for operations
        flexible_op_pattern = r'(add|subtract|multiply|divide).*?(\d+\.?\d*).*?(\d+\.?\d*).*?EOF'
        flexible_match = re.search(flexible_op_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if flexible_match:
            operation = flexible_match.group(1).lower()
            num1 = flexible_match.group(2)
            num2 = flexible_match.group(3)
            program_content = f"{operation}( {num1} {num2} ) EOF"
            
            # Try to find or calculate the answer
            try:
                if operation == "add":
                    answer_content = str(float(num1) + float(num2))
                elif operation == "subtract":
                    answer_content = str(float(num1) - float(num2))
                elif operation == "multiply":
                    answer_content = str(float(num1) * float(num2))
                elif operation == "divide":
                    answer_content = str(float(num1) / float(num2))
                else:
                    answer_content = "N/A"
            except:
                # Try to find a numerical answer in the text
                number_pattern = r'<begin_of_answer>\s*(-?\d+\.?\d*)\s*<end_of_answer>'
                number_match = re.search(number_pattern, text, re.DOTALL)
                
                if number_match:
                    answer_content = number_match.group(1).strip()
                else:
                    answer_content = "N/A"
            
            # Format clean output
            clean_output = (
                f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
            )
            
            return clean_output
    
    # If all else fails, return the original text with minimal cleaning
    if program_match:
        program_content = program_match.group(1).strip()
    else:
        # Try to find any operation pattern
        operation_pattern = r'(add|subtract|multiply|divide)\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)'
        operation_match = re.search(operation_pattern, text, re.IGNORECASE)
        
        if operation_match:
            operation = operation_match.group(1).lower()
            num1 = operation_match.group(2)
            num2 = operation_match.group(3)
            program_content = f"{operation}( {num1} {num2} ) EOF"
        else:
            program_content = "EOF"
    
    if answer_match:
        answer_content = answer_match.group(1).strip()
    else:
        # Try to find any numerical answer
        numerical_pattern = r'(\d+\.?\d*)'
        numerical_match = re.search(numerical_pattern, text)
        
        if numerical_match:
            answer_content = numerical_match.group(1).strip()
        else:
            answer_content = "N/A"
    
    # Format clean output
    clean_output = (
        f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
        f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
    )
    
    return clean_output


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
    
    for i, (pred, example_id) in enumerate(zip(predictions, example_ids)):
        try:
            # Check if this is a Qwen model output
            is_qwen_output = "<|im_start|>" in pred and "<|im_end|>" in pred
            
            if is_qwen_output:
                # Extract the assistant's response first
                assistant_pattern = r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)'
                assistant_matches = re.findall(assistant_pattern, pred, re.DOTALL)
                
                if assistant_matches:
                    # Use the last assistant response
                    pred = assistant_matches[-1].strip()
                    if i < 3:  # Debug info for first few examples
                        print(f"\nExtracted Qwen assistant response for example {i} (first 50 chars): {pred[:50]}...")
                    
                    # Look for program and answer tags directly in the assistant response
                    program_pattern = r'<begin_of_program>(.*?)<end_of_program>'
                    program_match = re.search(program_pattern, pred, re.DOTALL)
                    
                    answer_pattern = r'<begin_of_answer>(.*?)<end_of_answer>'
                    answer_match = re.search(answer_pattern, pred, re.DOTALL)
                    
                    if program_match and answer_match:
                        # If both tags are found, create a clean response with just these elements
                        program_content = program_match.group(1).strip()
                        answer_content = answer_match.group(1).strip()
                        
                        # Create a clean output
                        pred = (
                            f"<begin_of_program>\n{program_content}\n<end_of_program>\n\n"
                            f"<begin_of_answer>\n{answer_content}\n<end_of_answer>"
                        )
                        if i < 3:  # Debug info for first few examples
                            print(f"Extracted program and answer from Qwen response for example {i}: {pred[:100]}...")
            
            # First clean the output
            cleaned_output = clean_model_output(pred)
            
            # Then extract program tokens
            program_tokens = extract_program_tokens(cleaned_output)
            
            # Validate the program tokens
            valid_operations = ["add(", "subtract(", "multiply(", "divide("]
            has_valid_operation = any(op in program_tokens for op in valid_operations)
            
            # If no valid operation found, try to extract from the original prediction
            if not has_valid_operation and len(program_tokens) <= 2:
                # Try to find operation tokens directly in the prediction
                operation_pattern = r'(add|subtract|multiply|divide)\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)'
                match = re.search(operation_pattern, pred, re.IGNORECASE)
                
                if match:
                    operation = match.group(1).lower()
                    num1 = match.group(2)
                    num2 = match.group(3)
                    program_tokens = [f"{operation}(", num1, num2, ")", "EOF"]
            
            # Ensure the program tokens are properly formatted
            if len(program_tokens) > 1 and program_tokens[-1].upper() != "EOF":
                program_tokens.append("EOF")
            
            # Format tokens to match the official format
            formatted_tokens = []
            for token in program_tokens:
                # Handle operation tokens
                if token.lower() in ["add", "subtract", "multiply", "divide"]:
                    formatted_tokens.append(token.lower() + "(")
                elif token.lower() in ["add(", "subtract(", "multiply(", "divide("]:
                    formatted_tokens.append(token.lower())
                else:
                    formatted_tokens.append(token)
            
            # Add the formatted prediction
            formatted_predictions.append({
                "id": example_id,
                "predicted": formatted_tokens
            })
            
            # Print debug info for the first few examples
            if i < 3:
                print(f"\nExample {i} (ID: {example_id}):")
                print(f"Original prediction (first 100 chars): {pred[:100]}...")
                print(f"Cleaned output (first 100 chars): {cleaned_output[:100]}...")
                print(f"Extracted program tokens: {program_tokens}")
                print(f"Formatted tokens: {formatted_tokens}")
                
        except Exception as e:
            print(f"Error formatting prediction for example {example_id}: {e}")
            # Add a default prediction
            formatted_predictions.append({
                "id": example_id,
                "predicted": ["EOF"]
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