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
        
        # Special case: Check if the program text is just "and" (Llama 3.3 issue)
        if program_text.lower() == "and":
            # Try to find numbers in the text to construct a valid operation
            number_pattern = r'[-+]?\d*\.?\d+'
            numbers = re.findall(number_pattern, text)
            
            if len(numbers) >= 2:
                # Construct a valid addition operation as fallback
                return ["add(", numbers[0], numbers[1], ")", "EOF"]
            else:
                # If no numbers found, return EOF
                return ["EOF"]
        
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
                
                # Handle operation names with attached parenthesis
                if any(op in token.lower() for op in ["add(", "subtract(", "multiply(", "divide("]):
                    # Extract the operation name
                    op_name = re.match(r'(add|subtract|multiply|divide)', token.lower()).group(1)
                    tokens.append(op_name + "(")
                    
                    # Check if there's more content after the opening parenthesis
                    rest = token[len(op_name)+1:]
                    if rest:
                        # Process the rest of the token
                        for char in rest:
                            if char.isdigit() or char == '.':
                                # Start a new number token
                                tokens.append(char)
                            elif char == ')':
                                tokens.append(')')
                    
                    i += 1
                    in_operation = True
                # Handle operation names without attached parenthesis
                elif token.lower() in ["add", "subtract", "multiply", "divide"]:
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
                    # Check if the previous token already has a parenthesis attached
                    elif i > 0 and tokens and tokens[-1].endswith("("):
                        # Skip this parenthesis as it's redundant
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
                                    # Skip duplicate opening parenthesis
                                    if not (tokens and tokens[-1].endswith("(")):
                                        in_operation = True
                                        tokens.append(part)
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
                                # Skip duplicate opening parenthesis
                                if not (tokens and tokens[-1].endswith("(")):
                                    in_operation = True
                                    tokens.append(part)
                            elif part == ")":
                                in_operation = False
                                tokens.append(part)
                            else:
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
            
            # Post-process to remove any duplicate parentheses
            i = 0
            while i < len(tokens) - 1:
                if tokens[i].endswith("(") and tokens[i+1] == "(":
                    tokens.pop(i+1)  # Remove the duplicate "("
                else:
                    i += 1
                
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
    # Look for common operation patterns in different formats
    operation_patterns = [
        # Standard format: add( 320 177 )
        r'(add|subtract|multiply|divide)\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)',
        # Alternative format: add(320, 177)
        r'(add|subtract|multiply|divide)\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)',
        # Math expression format: 320 + 177
        r'([-+]?\d*\.?\d+)\s*[+\-*/]\s*([-+]?\d*\.?\d+)',
        # Calculation format: 320 plus 177
        r'([-+]?\d*\.?\d+)\s*(plus|minus|times|divided by)\s*([-+]?\d*\.?\d+)'
    ]
    
    for pattern in operation_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            
            # Handle standard operation format
            if groups[0].lower() in ["add", "subtract", "multiply", "divide"]:
                operation = groups[0].lower()
                num1 = groups[1]
                num2 = groups[2]
                return [f"{operation}(", num1, num2, ")", "EOF"]
            
            # Handle math expression format
            elif groups[1] in ["+", "-", "*", "/"]:
                op_map = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
                operation = op_map[groups[1]]
                num1 = groups[0]
                num2 = groups[2]
                return [f"{operation}(", num1, num2, ")", "EOF"]
            
            # Handle calculation format
            elif groups[1].lower() in ["plus", "minus", "times", "divided by"]:
                op_map = {"plus": "add", "minus": "subtract", "times": "multiply", "divided by": "divide"}
                operation = op_map[groups[1].lower()]
                num1 = groups[0]
                num2 = groups[2]
                return [f"{operation}(", num1, num2, ")", "EOF"]
    
    # Look for numbers in the text as a last resort
    number_pattern = r'[-+]?\d*\.?\d+'
    numbers = re.findall(number_pattern, text)
    
    # If we found at least two numbers, assume it's an addition
    if len(numbers) >= 2:
        return ["add(", numbers[0], numbers[1], ")", "EOF"]
    
    # If still nothing found, return just EOF
    return ["EOF"]


def extract_answer(text):
    """
    Extract the final answer from model output.
    Looks for content between <begin_of_answer> and <end_of_answer> tags.
    Returns a string representation of the answer.
    """
    # If input is None, return empty string
    if text is None:
        return ""
    
    # Ensure text is a string
    text = str(text)
    
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
        answer_text = match.group(1).strip()
        
        # Special case: Check if the answer text is just "and" (Llama 3.3 issue)
        if answer_text.lower() == "and":
            # Try to find numerical values in the text
            number_pattern = r'[-+]?\d*\.?\d+'
            numbers = re.findall(number_pattern, text)
            
            if numbers:
                # Use the last number as the answer (likely the result)
                return str(numbers[-1])
            else:
                # If no numbers found, return N/A
                return "N/A"
                
        return str(answer_text)
    
    # If no answer tags found, try to find numerical answers in the text
    # Look for patterns like "The answer is X" or "= X"
    alt_patterns = [
        r'(?:answer|result|value)(?:\s+is|\s*[:=])\s*([-+]?\d*\.?\d*)',
        r'(?:=|equals)\s*([-+]?\d*\.?\d*)',
        r'(?:[\$£€])\s*([-+]?\d*\.?\d*)',
        r'(?:final answer|final result)(?:\s+is|\s*[:=])\s*([-+]?\d*\.?\d*)',
        r'(?:calculated|computation|calculation)(?:\s+result|\s+answer)(?:\s+is|\s*[:=])\s*([-+]?\d*\.?\d*)',
        r'(?:therefore|thus|so)(?:,|\s+the\s+answer\s+is|\s+the\s+result\s+is)\s*([-+]?\d*\.?\d*)',
        r'(?:answer|result)(?:\s*:|\s+is)\s*([-+]?\d*\.?\d*)',
        # Look for a standalone number at the end of the text
        r'(?:^|\n|\.\s+)(\d+\.?\d*)(?:\s*$|\n)'
    ]
    
    for pattern in alt_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()
    
    # Look for operation results
    operation_result_patterns = [
        # Standard format: add(320, 177) = 497
        r'(?:add|subtract|multiply|divide)\s*\([^)]*\)\s*=\s*([-+]?\d*\.?\d*)',
        # Math expression format: 320 + 177 = 497
        r'\d+\s*[+\-*/]\s*\d+\s*=\s*([-+]?\d*\.?\d*)',
        # Last number in the text (as a fallback)
        r'([-+]?\d*\.?\d+)(?:\s*$|\n\s*$)'
    ]
    
    for pattern in operation_result_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()
    
    # If all else fails, extract all numbers and return the last one
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return str(numbers[-1])
    
    # Default return if no answer found
    if text.strip() == "":
        return "N/A"
    
    # Final fallback - just return the first 50 characters of the text
    return str(text[:50]) + ("..." if len(text) > 50 else "")


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
        List of formatted predictions
    """
    formatted_predictions = []
    
    for i, (pred, example_id) in enumerate(zip(predictions, example_ids)):
        try:
            # Skip empty predictions
            if not pred or pred.strip() == "":
                print(f"Warning: Empty prediction for example {example_id}")
                formatted_predictions.append({
                    "id": example_id,
                    "predicted": ["EOF"]
                })
                continue
            
            # Extract program tokens
            program_tokens = extract_program_tokens(pred)
            
            # Post-process tokens to remove duplicate parentheses
            if len(program_tokens) > 1:
                # Check for the pattern ["add(", "(", ...] and remove the second "("
                if program_tokens[0].endswith("(") and program_tokens[1] == "(":
                    program_tokens.pop(1)  # Remove the duplicate "("
            
            # Extract answer
            answer = extract_answer(pred)
            
            # Ensure program tokens are strings
            program_tokens = [str(token) for token in program_tokens]
            
            # Create formatted prediction
            formatted_prediction = {
                "id": str(example_id),
                "predicted": program_tokens
            }
            
            # Debug output
            if i % 10 == 0:  # Print debug info for every 10th example
                print(f"Example ID: {example_id}")
                print(f"Program tokens: {program_tokens}")
                print(f"Answer: {answer}")
            
            formatted_predictions.append(formatted_prediction)
            
        except Exception as e:
            print(f"Error formatting prediction for example {example_id}: {e}")
            # Add a placeholder entry
            formatted_predictions.append({
                "id": str(example_id),
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