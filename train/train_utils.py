import re
import torch
import numpy as np
import wandb

from datasets import Dataset
from unsloth.chat_templates import get_chat_template, standardize_sharegpt


def apply_custom_chat_template(tokenizer, chat_template_name):
    """
    Apply custom chat templates for models that might not be directly supported by unsloth.
    
    Args:
        tokenizer: The tokenizer to modify
        chat_template_name: The name of the chat template to apply
        
    Returns:
        The modified tokenizer with the custom chat template
    """
    # Qwen2.5 chat template
    if chat_template_name.lower() == "qwen":
        # Qwen2.5 uses a specific chat template format
        chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
        tokenizer.chat_template = chat_template
        print(f"Applied custom Qwen chat template")
        return tokenizer
    
    # Yi chat template
    elif chat_template_name.lower() == "yi":
        chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
        tokenizer.chat_template = chat_template
        print(f"Applied custom Yi chat template")
        return tokenizer
    
    # Claude chat template
    elif chat_template_name.lower() == "claude":
        chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}
{% elif message['role'] == 'user' %}Human: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"""
        tokenizer.chat_template = chat_template
        print(f"Applied custom Claude chat template")
        return tokenizer
    
    # Return None if no custom template was applied
    return None


def validate_format(text):
    """Validate that the text follows the expected format."""
    # Check for program tokens format
    program_pattern = r'<begin_of_program>\s*(.*?)\s*<end_of_program>'
    program_match = re.search(program_pattern, text, re.DOTALL)
    
    # Check for answer format
    answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    # Basic validation
    is_valid = program_match is not None and answer_match is not None
    
    # Check for common formatting issues in program tokens
    if program_match:
        program_text = program_match.group(1).strip()
        tokens = program_text.split()
        
        # Check for reference numbers without #
        for i, token in enumerate(tokens):
            if i > 0 and token.isdigit() and tokens[i-1].endswith("("):
                # This might be a reference number without #
                is_valid = False
                break
    
    # Check for common formatting issues in answers
    if answer_match and is_valid:
        answer_text = answer_match.group(1).strip()
        
        # Check for line breaks in the answer
        if '\n' in answer_text:
            is_valid = False
        
        # Check for multiple spaces in the answer
        if '  ' in answer_text:
            is_valid = False
        
        # Check for invalid numerical format (should be a clean number)
        # Valid formats: 123, 123.45, -123, -123.45
        if not re.match(r'^-?\d+\.?\d*$', answer_text):
            # Allow for some common formatting that can be normalized
            # e.g., with currency symbols or commas
            normalized = answer_text.replace('$', '').replace(',', '').replace('%', '')
            if not re.match(r'^-?\d+\.?\d*$', normalized):
                is_valid = False
    
    return is_valid


def debug_tokenized_dataset(tokenizer, dataset, num_samples=2):
    """Debug function to examine the tokenized dataset and identify issues."""
    print("\n===== DEBUGGING TOKENIZED DATASET =====")
    print(f"Examining {num_samples} samples from the dataset")
    
    # Get a few samples from the dataset
    samples = dataset.select(range(min(num_samples, len(dataset))))
    
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        text = sample["text"]
        print(f"Raw text (first 100 chars): {text[:100]}...")
        
        # Tokenize the text
        tokenized = tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"][0]
        
        # Decode back to see what the tokenizer understood
        decoded = tokenizer.decode(input_ids)
        print(f"Decoded (first 100 chars): {decoded[:100]}...")
        
        # Check for key markers
        user_marker = "<|im_start|>user"
        assistant_marker = "<|im_start|>assistant"
        
        user_pos = text.find(user_marker)
        assistant_pos = text.find(assistant_marker)
        
        print(f"User marker position: {user_pos}")
        print(f"Assistant marker position: {assistant_pos}")
        
        # Check if markers are present in the expected order
        if user_pos >= 0 and assistant_pos >= 0 and user_pos < assistant_pos:
            print("✅ Markers found in correct order")
        else:
            print("❌ Markers not found in correct order")
        
        # Try to manually apply train_on_responses_only logic
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
        
        instruction_pos = text.find(instruction_part)
        response_pos = text.find(response_part)
        
        if instruction_pos >= 0 and response_pos >= 0:
            print(f"Instruction part found at: {instruction_pos}")
            print(f"Response part found at: {response_pos}")
            
            # Get the text between markers
            instruction_text = text[instruction_pos:response_pos]
            response_text = text[response_pos:]
            
            print(f"Instruction text (first 50 chars): {instruction_text[:50]}...")
            print(f"Response text (first 50 chars): {response_text[:50]}...")
        else:
            print("❌ Could not find instruction or response parts with exact markers")
    
    print("\n===== END DEBUGGING =====")


def custom_train_on_responses_only(trainer, tokenizer, instruction_part, response_part, end_token=None):
    """
    Custom implementation of train_on_responses_only that's more robust for Qwen models.
    
    Args:
        trainer: The SFTTrainer instance
        tokenizer: The tokenizer
        instruction_part: The marker for user/instruction part
        response_part: The marker for assistant/response part
        end_token: Optional end token to mark the end of responses
        
    Returns:
        The modified trainer
    """
    print(f"\n===== APPLYING CUSTOM TRAIN_ON_RESPONSES_ONLY =====")
    print(f"Instruction part: '{instruction_part}'")
    print(f"Response part: '{response_part}'")
    if end_token:
        print(f"End token: '{end_token}'")
    
    # Get the original dataset
    dataset = trainer.train_dataset
    
    # Function to process a single example
    def process_example(example):
        text = example["text"]
        
        # Find the positions of markers
        instruction_pos = text.find(instruction_part)
        response_pos = text.find(response_part, instruction_pos + len(instruction_part) if instruction_pos >= 0 else 0)
        
        # If we can't find the markers, return the original
        if instruction_pos < 0 or response_pos < 0:
            print(f"Warning: Could not find markers in text: {text[:100]}...")
            return example
        
        # Tokenize the full text
        tokenized = tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"][0]
        
        # Find token indices for the markers
        full_text_tokens = tokenizer.encode(text)
        instruction_tokens = tokenizer.encode(instruction_part)
        response_tokens = tokenizer.encode(response_part)
        
        # Find the position of instruction and response in the tokenized text
        instruction_token_pos = -1
        response_token_pos = -1
        
        # Simple search for the instruction tokens
        for i in range(len(full_text_tokens) - len(instruction_tokens) + 1):
            if full_text_tokens[i:i+len(instruction_tokens)] == instruction_tokens:
                instruction_token_pos = i
                break
        
        # Search for response tokens after the instruction
        if instruction_token_pos >= 0:
            for i in range(instruction_token_pos + len(instruction_tokens), len(full_text_tokens) - len(response_tokens) + 1):
                if full_text_tokens[i:i+len(response_tokens)] == response_tokens:
                    response_token_pos = i
                    break
        
        # If we couldn't find the token positions, use a different approach
        if instruction_token_pos < 0 or response_token_pos < 0:
            print(f"Warning: Could not find token positions for markers. Using text positions instead.")
            # Tokenize the text up to the response part
            prefix_text = text[:response_pos]
            prefix_tokens = tokenizer.encode(prefix_text)
            response_token_pos = len(prefix_tokens)
        
        # Create labels: -100 for non-response tokens, actual token IDs for response tokens
        labels = [-100] * len(input_ids)
        
        # Set labels for the response part
        if response_token_pos >= 0:
            for i in range(response_token_pos, len(input_ids)):
                labels[i] = input_ids[i].item()
        
        # Check if we have any non-masked labels
        if all(label == -100 for label in labels):
            print(f"Warning: All labels are masked for this example!")
            # Print some debug info
            print(f"Text: {text[:100]}...")
            print(f"Instruction pos: {instruction_pos}, Response pos: {response_pos}")
            print(f"Instruction token pos: {instruction_token_pos}, Response token pos: {response_token_pos}")
        
        return {"input_ids": input_ids, "labels": labels, "attention_mask": tokenized["attention_mask"][0]}
    
    # Process a few examples for debugging
    print("\nProcessing a few examples for debugging:")
    for i in range(min(2, len(dataset))):
        example = dataset[i]
        processed = process_example(example)
        
        # Check if we have any non-masked labels
        if "labels" in processed and all(label == -100 for label in processed["labels"]):
            print(f"Example {i}: All labels are masked!")
        else:
            print(f"Example {i}: Labels contain non-masked values")
    
    # Create a new dataset with the processed examples
    # Process all examples
    processed_examples = []
    for i, example in enumerate(dataset):
        processed = process_example(example)
        processed_examples.append(processed)
    
    # Create a new dataset
    processed_dataset = Dataset.from_dict({
        "input_ids": [ex["input_ids"] for ex in processed_examples],
        "labels": [ex["labels"] for ex in processed_examples],
        "attention_mask": [ex["attention_mask"] for ex in processed_examples]
    })
    
    # Replace the trainer's dataset
    trainer.train_dataset = processed_dataset
    
    print(f"===== CUSTOM TRAIN_ON_RESPONSES_ONLY APPLIED =====\n")
    return trainer


def analyze_predictions(pred_str, label_str, pred_answers, label_answers, step=None):
    """Analyze prediction patterns to understand model behavior."""
    analysis = {
        "total_samples": len(pred_str),
        "program_token_matches": 0,
        "numerical_answers": 0,
        "string_answers": 0,
        "answer_type_mismatches": 0,
        "empty_predictions": 0,
        "exact_matches": 0,
        "format_valid": 0,
    }
    
    # Initialize wandb if not already done
    if not wandb.run:
        try:
            wandb.init(project="finqa_analysis", mode="offline")
        except:
            pass
    
    for i, (pred, label, pred_ans, label_ans) in enumerate(zip(pred_str, label_str, pred_answers, label_answers)):
        # Check for empty predictions
        if not pred or pred.isspace():
            analysis["empty_predictions"] += 1
            continue
            
        # Check for exact matches
        if pred == label:
            analysis["exact_matches"] += 1

        # Check format validity
        if validate_format(pred):
            analysis["format_valid"] += 1
            
        # Count answer types
        if isinstance(pred_ans, float):
            analysis["numerical_answers"] += 1
        elif isinstance(pred_ans, str) and pred_ans:
            analysis["string_answers"] += 1
            
        # Check for type mismatches
        if (isinstance(pred_ans, float) and isinstance(label_ans, str)) or \
           (isinstance(pred_ans, str) and isinstance(label_ans, float)):
            analysis["answer_type_mismatches"] += 1
    
    # Calculate percentages
    total = max(1, len(pred_str))
    analysis_keys = list(analysis.keys())  # Create a copy of keys to prevent iteration issues
    for key in analysis_keys:
        analysis[f"{key}_pct"] = analysis[key] / total * 100
        
    # Log to wandb
    if wandb.run:
        metrics_to_log = {f"analysis/{k}": v for k, v in analysis.items()}
        if step is not None:
            wandb.log(metrics_to_log, step=step)
        else:
            wandb.log(metrics_to_log)
            
    # Print summary
    print("\nPrediction Analysis:")
    for key, value in analysis.items():
        if not key.endswith('_pct'):
            print(f"  - {key}: {value} ({analysis[f'{key}_pct']:.1f}%)")
            
    return analysis


def extract_program_tokens(text):
    """Extract program tokens from the text."""
    try:
        program_pattern = r'<begin_of_program>\s*(.*?)\s*<end_of_program>'
        match = re.search(program_pattern, text, re.DOTALL)
        if match:
            program_text = match.group(1).strip()
            return program_text.split()
        return ["EOF"]
    except Exception as e:
        print(f"Error extracting program tokens: {e}")
        return ["ERROR"]


def extract_answer(text):
    """Extract the answer from the text."""
    try:
        # First try to extract from the standard format
        answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
        match = re.search(answer_pattern, text, re.DOTALL)
        if match:
            solution = match.group(1).strip()
            # Normalize numerical values
            solution = solution.replace('%', '')
            solution = solution.replace('$', '')
            solution = solution.replace(',', '')
            
            # Clean up line breaks and extra spaces
            solution = re.sub(r'\s+', ' ', solution).strip()
            
            # Try to convert to float for numerical comparison
            try:
                # Check if the solution contains only digits, decimal point, and minus sign
                # More robust pattern that handles potential line breaks and spaces
                if re.match(r'^-?\d+\.?\d*$', solution):
                    return float(solution)
                else:
                    # If it contains other characters, it's likely a string answer
                    return solution.lower()  # Normalize case for string comparison
            except ValueError:
                # If conversion fails, try to extract just the numerical part
                numerical_match = re.search(r'(-?\d+\.?\d*)', solution)
                if numerical_match:
                    try:
                        return float(numerical_match.group(1))
                    except ValueError:
                        pass
                # Return as string if all else fails
                return solution.lower()
        
        # If standard format not found, try to find numerical answers in the text
        # Look for patterns like "The answer is X" or "= X"
        alt_patterns = [
            r'(?:answer|result|value)(?:\s+is|\s*[:=])\s*([-\d\.\,]+)',
            r'(?:=|equals)\s*([-\d\.\,]+)',
            r'(?:[\$£€])\s*([-\d\.\,]+)'
        ]
        
        for pattern in alt_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                solution = match.group(1).strip()
                solution = solution.replace(',', '')
                try:
                    return float(solution)
                except ValueError:
                    pass
        
        # If no patterns matched, return the whole text as a fallback
        return text.strip().lower()
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return "ERROR"


def setup_wandb(args):
    """Initialize Weights & Biases for experiment tracking."""
    config = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_eval_samples": args.max_eval_samples,
        "force_dataset_download": args.force_dataset_download,
        "chat_template": args.chat_template,
    }
    
    run_name = args.wandb_run_name or f"{args.model_name.split('/')[-1]}-lora-r{args.lora_r}-bs{args.batch_size*args.gradient_accumulation_steps}"
    wandb.init(project=args.wandb_project, name=run_name, config=config)
    return wandb.run


def format_qwen_dataset(convo, sys_prompt):
    """Format a conversation for Qwen models."""
    formatted_text = "<|im_start|>system\n" + sys_prompt + "<|im_end|>\n"
    
    for message in convo:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            formatted_text += "<|im_start|>user\n" + content + "<|im_end|>\n"
        elif role == "assistant":
            formatted_text += "<|im_start|>assistant\n" + content + "<|im_end|>\n"
    
    return formatted_text


def get_chat_template_name(model_name):
    """Determine the chat template name based on the model name."""
    model_name_lower = model_name.lower()
    
    if "phi-4" in model_name_lower or "phi4" in model_name_lower:
        return "phi-4"
    elif "qwen2" in model_name_lower or "qwen-2" in model_name_lower:
        return "qwen"
    elif "llama" in model_name_lower or "mistral" in model_name_lower:
        return "llama-2"
    elif "gemma" in model_name_lower:
        return "gemma"
    elif "phi-2" in model_name_lower or "phi2" in model_name_lower:
        return "phi-2"
    elif "falcon" in model_name_lower:
        return "falcon"
    elif "yi" in model_name_lower:
        return "yi"
    elif "claude" in model_name_lower:
        return "claude"
    else:
        # Default to chatml format which works for many models
        return "chatml"


def safe_int_conversion(token, vocab_size):
    """Safely convert tokens to integers within the vocabulary range."""
    try:
        # Check if it's a valid integer within Python's int range
        if isinstance(token, (int, np.integer)):
            # For numpy integers, convert to Python int
            int_val = int(token)
            # Clip to valid vocabulary range if needed
            if int_val >= vocab_size:
                print(f"  - Clipping token {int_val} to vocab size {vocab_size-1}")
                return vocab_size - 1
            elif int_val < 0:
                # Skip negative tokens (they're masked)
                return None
            return int_val
        elif isinstance(token, float) and token.is_integer():
            # For floats that are integers
            int_val = int(token)
            # Clip to valid vocabulary range if needed
            if int_val >= vocab_size:
                print(f"  - Clipping token {int_val} to vocab size {vocab_size-1}")
                return vocab_size - 1
            elif int_val < 0:
                # Skip negative tokens (they're masked)
                return None
            return int_val
        return None
    except (OverflowError, ValueError) as e:
        # If conversion fails due to value being too large
        print(f"  - Conversion error for token {token} (type: {type(token)}): {e}")
        return None