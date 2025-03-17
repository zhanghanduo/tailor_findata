import os
import json
import re
import streamlit as st
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set page configuration
st.set_page_config(
    page_title="Financial Document QA (Local Model)",
    page_icon="💰",
    layout="wide"
)

# Load the ConvFinQA dataset
@st.cache_data
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Format table for display
def format_table(table_data):
    if not table_data:
        return "No table data available"
    
    # Convert to markdown table
    table_md = ""
    for i, row in enumerate(table_data):
        if i == 0:
            table_md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
            table_md += "| " + " | ".join(["---"] * len(row)) + " |\n"
        else:
            table_md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
    
    return table_md

# Load model and tokenizer
@st.cache_resource
def load_model(model_path):
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load in 4-bit to save memory
    try:
        from transformers import BitsAndBytesConfig
        
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
    except:
        # Fallback to regular loading
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

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
        "explanation": "",  # Local models typically don't provide explanations
        "raw_output": assistant_response  # Include raw output for debugging
    }

# Create a function to generate the final.json file
def generate_final_json(dataset, output_path="app/final.json"):
    results = []
    
    # Load model
    model_path = st.session_state.get("model_path", "meta-llama/Llama-2-7b-chat-hf")
    model, tokenizer, device = load_model(model_path)
    
    # Process each example
    for i, example in enumerate(dataset):
        if "qa" in example and "question" in example["qa"]:
            question = example["qa"]["question"]
            result = answer_question(model, tokenizer, device, example, question)
            
            results.append({
                "qa": {
                    "question": question,
                    "answer": result["answer"]
                }
            })
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return output_path

# Main application
def main():
    st.title("Financial Document QA System (Local Model)")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path/Name",
        value=st.session_state.get("model_path", "meta-llama/Llama-2-7b-chat-hf")
    )
    
    # Save model path to session state
    if model_path != st.session_state.get("model_path"):
        st.session_state["model_path"] = model_path
    
    # Dataset selection
    dataset_path = st.sidebar.selectbox(
        "Select Dataset",
        ["data/train.json", "data/dev.json", "data/sample_train.json"],
        index=2  # Default to sample_train.json
    )
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
        st.sidebar.success(f"Loaded {len(dataset)} examples from {dataset_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")
        return
    
    # Generate final.json button
    if st.sidebar.button("Generate final.json"):
        with st.spinner("Generating final.json..."):
            output_path = generate_final_json(dataset)
            st.sidebar.success(f"Generated final.json at {output_path}")
    
    # Example selection
    example_idx = st.sidebar.number_input(
        "Select Example Index",
        min_value=0,
        max_value=len(dataset)-1,
        value=0
    )
    
    # Load model button
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            try:
                model, tokenizer, device = load_model(model_path)
                st.session_state["model_loaded"] = True
                st.sidebar.success(f"Model loaded successfully on {device}")
            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")
                st.session_state["model_loaded"] = False
    
    # Display selected example
    if example_idx < len(dataset):
        example = dataset[example_idx]
        
        # Display document content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Document Text")
            
            # Pre-text
            if "pre_text" in example:
                st.markdown("**Pre Text:**")
                st.markdown("\n".join(example["pre_text"]))
            
            # Post-text
            if "post_text" in example:
                st.markdown("**Post Text:**")
                st.markdown("\n".join(example["post_text"]))
        
        with col2:
            st.subheader("Financial Table")
            if "table_ori" in example:
                st.markdown(format_table(example["table_ori"]))
        
        # Question answering section
        st.subheader("Question Answering")
        
        # Get question from example or let user input
        default_question = example.get("qa", {}).get("question", "") if "qa" in example else ""
        question = st.text_input("Question:", value=default_question)
        
        if st.button("Answer Question"):
            if question and st.session_state.get("model_loaded", False):
                with st.spinner("Generating answer..."):
                    # Get model from session state
                    model, tokenizer, device = load_model(model_path)
                    
                    # Generate answer
                    answer = answer_question(model, tokenizer, device, example, question)
                    
                    # Display answer
                    st.success(f"Answer: {answer['answer']}")
                    
                    # Display calculation
                    with st.expander("See calculation"):
                        st.code(answer["program"])
                    
                    # Display raw output for debugging
                    with st.expander("See raw model output"):
                        st.text(answer["raw_output"])
                    
                    # If ground truth is available, show it
                    if "qa" in example and "answer" in example["qa"]:
                        st.info(f"Ground Truth Answer: {example['qa']['answer']}")
            elif not st.session_state.get("model_loaded", False):
                st.warning("Please load the model first")
            else:
                st.warning("Please enter a question")

if __name__ == "__main__":
    main() 