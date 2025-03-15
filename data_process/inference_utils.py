import re
from typing import Dict, List, Union, Any

def clean_model_output(model_output: str) -> str:
    """
    Clean up model output to extract only the program and answer tags.
    This function should be used during inference to remove any extraneous text.
    
    Args:
        model_output (str): Raw model output from the model
        
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


def post_process_batch(model_outputs: List[str]) -> List[str]:
    """
    Apply post-processing to a batch of model outputs.
    
    Args:
        model_outputs (List[str]): List of raw model outputs
        
    Returns:
        List[str]: List of cleaned model outputs
    """
    return [clean_model_output(output) for output in model_outputs]


def extract_program_and_answer(model_output: str) -> Dict[str, str]:
    """
    Extract program and answer from model output.
    
    Args:
        model_output (str): Model output (can be raw or cleaned)
        
    Returns:
        Dict[str, str]: Dictionary with program and answer
    """
    # Clean the output first to ensure consistent format
    cleaned_output = clean_model_output(model_output)
    
    # Extract program
    program_pattern = r'<begin_of_program>(.*?)<end_of_program>'
    program_match = re.search(program_pattern, cleaned_output, re.DOTALL)
    program = program_match.group(1).strip() if program_match else "EOF"
    
    # Extract answer
    answer_pattern = r'<begin_of_answer>(.*?)<end_of_answer>'
    answer_match = re.search(answer_pattern, cleaned_output, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else "N/A"
    
    return {
        "program": program,
        "answer": answer
    }


def evaluate_answer_accuracy(predicted_answer: str, ground_truth_answer: str, 
                            tolerance: float = 0.01) -> bool:
    """
    Evaluate the accuracy of a predicted answer against the ground truth.
    
    Args:
        predicted_answer (str): Predicted answer from the model
        ground_truth_answer (str): Ground truth answer
        tolerance (float): Tolerance for numerical comparison (default: 0.01)
        
    Returns:
        bool: True if the answer is correct, False otherwise
    """
    try:
        # Try to convert both to float for numerical comparison
        pred_float = float(predicted_answer)
        truth_float = float(ground_truth_answer)
        
        # Check if the difference is within tolerance
        return abs(pred_float - truth_float) <= tolerance
    except (ValueError, TypeError):
        # If conversion fails, do string comparison
        return predicted_answer.strip() == ground_truth_answer.strip() 