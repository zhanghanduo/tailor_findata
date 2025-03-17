import os
import json
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Extract program tokens from model output
def extract_program_tokens(text):
    # Look for program tags
    program_pattern = r'<begin_of_program>\s*(.*?)\s*<end_of_program>'
    match = re.search(program_pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return ""

# Extract answer from model output
def extract_answer(text):
    # Look for answer tags
    answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
    match = re.search(answer_pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no answer tags found, try to find numerical answers in the text
    alt_patterns = [
        r'(?:answer|result|value)(?:\s+is|\s*[:=])\s*([-+]?\d*\.?\d*%?)',
        r'(?:=|equals)\s*([-+]?\d*\.?\d*%?)',
        r'(?:[\$£€])\s*([-+]?\d*\.?\d*%?)',
        r'(?:final answer|final result)(?:\s+is|\s*[:=])\s*([-+]?\d*\.?\d*%?)'
    ]
    
    for pattern in alt_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()
    
    # If all else fails, extract all numbers and return the last one
    numbers = re.findall(r'[-+]?\d*\.?\d+%?', text)
    if numbers:
        return numbers[-1]
    
    # If no answer found, return N/A
    return "N/A"

# Function to answer questions using local model
def answer_question(model, tokenizer, device, document, question):
    # Create context from document
    context = ""
    
    # Add pre-text
    if "pre_text" in document:
        context += "Document Text:\n" + "\n".join(document["pre_text"]) + "\n\n"
    
    # Add table
    if "table_ori" in document:
        context += "Financial Table:\n"
        for row in document["table_ori"]:
            context += " | ".join(str(cell) for cell in row) + "\n"
        context += "\n"
    
    # Add post-text
    if "post_text" in document:
        context += "\n" + "\n".join(document["post_text"])
    
    # Create system prompt
    system_prompt = """You are a financial analyst assistant that answers questions about financial documents.
    
For each question, you need to:
1. Analyze the financial data provided
2. Generate the calculation program that represents the steps to solve the problem
3. Calculate the final answer

Your response should include:
- The program tokens that represent the calculation using <begin_of_program> and <end_of_program> tags
- The final answer using <begin_of_answer> and <end_of_answer> tags

The program tokens should follow this format:
<begin_of_program>
operation_name(number1, number2), operation_name(#0, number3)
<end_of_program>

Where:
- operation_name can be: add, subtract, multiply, divide
- #0, #1, etc. refer to the results of previous operations
- All numbers should be extracted directly from the document

Example:
<begin_of_program>
subtract(206588, 181001), divide(#0, 181001)
<end_of_program>

<begin_of_answer>
14.1%
<end_of_answer>

Be precise with your calculations and format the answer appropriately (include % for percentages, $ for dollar amounts, etc.).
"""
    
    # Create user content
    user_content = f"Context: {context}\n\nQuestion: {question}"
    
    # Format messages for the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        pad_to_multiple_of=8
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.1
        )
    
    # Decode
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    assistant_response = decoded_output
    
    # Try to extract just the generated part (after the prompt)
    if assistant_response.startswith(prompt):
        assistant_response = assistant_response[len(prompt):].strip()
    
    # Extract program and answer
    program = extract_program_tokens(assistant_response)
    answer = extract_answer(assistant_response)
    
    return {
        "program": program,
        "answer": answer,
        "raw_output": assistant_response
    }

def load_model(model_path, load_in_4bit=True):
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load in 4-bit to save memory if requested
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            
            print("Loading model in 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=bnb_config
            )
        except Exception as e:
            print(f"Error loading in 4-bit: {e}")
            print("Falling back to regular loading...")
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    else:
        # Regular loading
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

def generate_final_json(dataset_path, model_path, output_path, load_in_4bit=True):
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} examples")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, tokenizer, device = load_model(model_path, load_in_4bit)
    
    # Process each example
    results = []
    
    print("Processing examples...")
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        if "qa" in example and "question" in example["qa"]:
            question = example["qa"]["question"]
            result = answer_question(model, tokenizer, device, example, question)
            
            results.append({
                "qa": {
                    "question": question,
                    "answer": result["answer"]
                }
            })
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} examples")
    
    # Save to file
    print(f"Saving results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Done! Results saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate final.json for ConvFinQA dataset")
    parser.add_argument("--dataset", type=str, default="data/sample_train.json", help="Path to dataset file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model path or name")
    parser.add_argument("--output", type=str, default="app/final.json", help="Output file path")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    generate_final_json(
        dataset_path=args.dataset,
        model_path=args.model,
        output_path=args.output,
        load_in_4bit=not args.no_4bit
    )

if __name__ == "__main__":
    main() 