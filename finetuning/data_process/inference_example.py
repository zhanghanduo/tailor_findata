import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference_utils import clean_model_output, extract_program_and_answer, evaluate_answer_accuracy

def load_model_and_tokenizer(model_path):
    """
    Load the model and tokenizer from the specified path.
    
    Args:
        model_path (str): Path to the model directory or HuggingFace model ID
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """
    Generate a response from the model.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use for encoding/decoding
        prompt (str): The prompt to generate from
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The generated response
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def main():
    # Check if model path is provided
    if len(sys.argv) < 2:
        print("Usage: python inference_example.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Example financial context and question
    financial_context = """
    Here's a table showing financial data:
    
    Item | 2007 | 2008 | 2009
    -----|------|------|------
    Revenue | 100.5 | 120.3 | 95.7
    Expenses | 80.2 | 85.6 | 70.3
    Net Income | 20.3 | 34.7 | 25.4
    Assets | 500.0 | 550.0 | 580.0
    Liabilities | 300.0 | 320.0 | 330.0
    Equity | 200.0 | 230.0 | 250.0
    """
    
    # Example questions
    questions = [
        "What was the revenue in 2008?",
        "What was the difference between revenue and expenses in 2008?",
        "What was the percentage increase in revenue from 2007 to 2008?"
    ]
    
    # System prompt
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
    
    # Process each question
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        
        # Format the prompt
        if i == 0:
            # First question includes the financial context
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nI'm looking at some financial data. Here's the context:\n\n{financial_context}\n\n{question} [/INST]"
        else:
            # Subsequent questions don't need the context again
            prompt = f"<s>[INST] {question} [/INST]"
        
        # Generate raw response
        raw_response = generate_response(model, tokenizer, prompt)
        print("\nRaw model output:")
        print(raw_response)
        
        # Clean the response
        cleaned_response = clean_model_output(raw_response)
        print("\nCleaned output:")
        print(cleaned_response)
        
        # Extract program and answer
        extracted = extract_program_and_answer(raw_response)
        print("\nExtracted program:", extracted["program"])
        print("Extracted answer:", extracted["answer"])
        
        print("-" * 80)

if __name__ == "__main__":
    main() 