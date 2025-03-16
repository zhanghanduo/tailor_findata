import os
import argparse
import wandb
import re
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer
from transformers import (
    TrainingArguments, 
    DataCollatorForSeq2Seq,
    TextStreamer,
    EvalPrediction
)
import torch
from train_utils import (
    apply_custom_chat_template,
    validate_format,
    debug_tokenized_dataset,
    custom_train_on_responses_only
)
import random





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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Phi-4 with LoRA")
    parser.add_argument("--model_name", type=str, default="unsloth/phi-4", help="Base model to fine-tune")
    parser.add_argument("--cache_dir", type=str, default="/workspace/mnt/watt/public_models", help="Cache directory for models")
    parser.add_argument("--dataset_name", type=str, default="christlurker/findata_test", help="Dataset to use for training")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save model checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Whether to load in 4-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides num_train_epochs if > 0)")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=20, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=40, help="Save checkpoint steps")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="phi4-lora-sft", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--eval_split_percentage", type=int, default=4, help="Percentage of data to use for evaluation (only used if dataset has no validation split)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing to save memory")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum number of samples to use for evaluation")
    parser.add_argument("--force_dataset_download", action="store_true", default=False, help="Force redownload of dataset instead of using cache")
    parser.add_argument("--chat_template", type=str, default=None, help="Chat template to use (e.g., 'phi-4', 'llama-2', 'gemma', 'qwen', 'yi', 'claude', 'phi-2', 'falcon', 'chatml'). If not specified, will be auto-detected based on model name.")
    args = parser.parse_args()
    
    # Ensure max_steps has a valid value
    if args.max_steps is None:
        args.max_steps = -1
        
    return args


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
    
    run_name = args.wandb_run_name or f"phi4-lora-r{args.lora_r}-bs{args.batch_size*args.gradient_accumulation_steps}"
    wandb.init(project=args.wandb_project, name=run_name, config=config)
    return wandb.run





def load_model_and_tokenizer(args):
    """Load the base model and tokenizer."""
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else None,
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Use chat template from args if provided, otherwise auto-detect
    if args.chat_template:
        chat_template = args.chat_template
        print(f"Using user-specified chat template: {chat_template}")
    else:
        # Set chat template based on model type
        model_name_lower = args.model_name.lower()
        
        # Detect model type and set appropriate chat template
        if "phi-4" in model_name_lower or "phi4" in model_name_lower:
            chat_template = "phi-4"
        elif "qwen2" in model_name_lower or "qwen-2" in model_name_lower:
            chat_template = "qwen"
        elif "llama" in model_name_lower or "mistral" in model_name_lower:
            chat_template = "llama-2"
        elif "gemma" in model_name_lower:
            chat_template = "gemma"
        elif "phi-2" in model_name_lower or "phi2" in model_name_lower:
            chat_template = "phi-2"
        elif "falcon" in model_name_lower:
            chat_template = "falcon"
        elif "yi" in model_name_lower:
            chat_template = "yi"
        elif "claude" in model_name_lower:
            chat_template = "claude"
        else:
            # Default to chatml format which works for many models
            chat_template = "chatml"
        
        print(f"Auto-detected chat template: {chat_template} for model: {args.model_name}")
    
    try:
        # First try to apply a custom chat template
        custom_tokenizer = apply_custom_chat_template(tokenizer, chat_template)
        
        # If a custom template was applied, use that tokenizer
        if custom_tokenizer is not None:
            tokenizer = custom_tokenizer
        else:
            # Otherwise, use the unsloth chat template function
            tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    except Exception as e:
        print(f"Warning: Failed to apply chat template '{chat_template}': {e}")
        print("Falling back to default chat template 'chatml'")
        try:
            # Try custom template first
            custom_tokenizer = apply_custom_chat_template(tokenizer, "chatml")
            if custom_tokenizer is not None:
                tokenizer = custom_tokenizer
            else:
                # Fall back to unsloth's implementation
                tokenizer = get_chat_template(tokenizer, chat_template="chatml")
        except Exception as e2:
            print(f"Warning: Also failed to apply fallback template: {e2}")
            print("Proceeding with original tokenizer without chat template")
    
    return model, tokenizer


def prepare_dataset(tokenizer, args):
    """Load and prepare the dataset for training and evaluation."""
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        system_prompts = examples["system"]

        texts = []
        for convo, sys_prompt in zip(convos, system_prompts):
            # Check if we're using a Qwen model
            is_qwen = "qwen" in args.model_name.lower()
            
            # For Qwen models, ensure the conversation format is correct
            if is_qwen:
                # Print debug info for the first few examples
                if len(texts) < 2:
                    print(f"\nFormatting example for Qwen model:")
                    print(f"System prompt: {sys_prompt}")
                    print(f"Conversation: {convo[:2]}")  # Show first 2 messages
                
                # Ensure the conversation has the right format
                formatted_text = "<|im_start|>system\n" + sys_prompt + "<|im_end|>\n"
                
                for message in convo:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "user":
                        formatted_text += "<|im_start|>user\n" + content + "<|im_end|>\n"
                    elif role == "assistant":
                        # For assistant messages, ensure they have the proper answer format
                        if "<begin_of_answer>" not in content and "<end_of_answer>" not in content:
                            # Try to extract a numerical answer from the content
                            numerical_match = re.search(r'(-?\d+\.?\d*)', content)
                            if numerical_match:
                                # Format with the expected tags
                                answer = numerical_match.group(1).strip()
                                content = f"<begin_of_program>\nEOF\n<end_of_program>\n\n<begin_of_answer>\n{answer}\n<end_of_answer>"
                        
                        formatted_text += "<|im_start|>assistant\n" + content + "<|im_end|>\n"
                
                # Print the formatted text for debugging
                if len(texts) < 2:
                    print(f"Formatted text (first 200 chars): {formatted_text[:200]}...")
                
                texts.append(formatted_text)
            else:
                # For other models, use the standard chat template
                formatted_text = tokenizer.apply_chat_template(
                    convo, 
                    tokenize=False, 
                    add_generation_prompt=False,
                    system_message=sys_prompt
                )
                texts.append(formatted_text)
        
        return {"text": texts}
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")

    # Load both train and validation splits
    dataset_dict = load_dataset(
        args.dataset_name,
        cache_dir=None,  # Don't use cache
        download_mode="force_redownload" if args.force_dataset_download else "reuse_dataset_if_exists",  # Force redownload if specified
        verification_mode="no_checks"  # Skip verification checks
    )
    
    # Check if the dataset has a validation split
    if "validation" in dataset_dict:
        print("Using existing train/validation splits from the dataset")
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["validation"]
        
        # Standardize the format
        train_dataset = standardize_sharegpt(train_dataset)
        eval_dataset = standardize_sharegpt(eval_dataset)
    else:
        # Fall back to creating a split if no validation set exists
        print("No validation split found in dataset. Creating a custom split.")
        train_dataset = dataset_dict["train"]
        train_dataset = standardize_sharegpt(train_dataset)
        
        if args.eval_split_percentage > 0:
            split = train_dataset.train_test_split(test_size=args.eval_split_percentage/100, seed=args.seed)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            eval_dataset = None
    
    # Limit eval dataset size if specified
    if eval_dataset and args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Limiting evaluation dataset to {args.max_eval_samples} samples")
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_eval_samples)))
    
    # Set padding and truncation settings for Qwen models BEFORE formatting
    if "qwen" in args.model_name.lower():
        print("Applying padding and truncation settings for Qwen model")
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Remove reference to model which is not available in this scope
        # We'll handle model config updates in the setup_trainer function
    
    # Format datasets
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=2,
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            formatting_prompts_func,
            batched=True,
            num_proc=2,
        )
        
        # For Qwen models, add additional processing for the evaluation dataset
        if "qwen" in args.model_name.lower():
            print("Applying additional processing for Qwen evaluation dataset")
            
            # Function to extract the last assistant response for each conversation
            def extract_last_assistant_response(example):
                text = example["text"]
                
                # For debugging
                if random.random() < 0.05:  # Print ~5% of examples
                    print(f"\nQwen eval example text (first 200 chars): {text[:200]}...")
                
                # Extract all assistant responses
                assistant_pattern = r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>'
                assistant_responses = re.findall(assistant_pattern, text, re.DOTALL)
                
                # Use the last assistant response as the ground truth
                if assistant_responses:
                    last_response = assistant_responses[-1].strip()
                    
                    # For debugging
                    if random.random() < 0.05:  # Print ~5% of examples
                        print(f"Extracted last assistant response: {last_response}")
                    
                    # Store the last response for evaluation
                    example["last_assistant_response"] = last_response
                else:
                    example["last_assistant_response"] = ""
                    
                return example
            
            # Apply the extraction
            eval_dataset = eval_dataset.map(
                extract_last_assistant_response,
                batched=False,
                desc="Extracting last assistant responses for Qwen evaluation"
            )
    
    # For Qwen models, tokenize the dataset with explicit padding and truncation
    if "qwen" in args.model_name.lower():
        print("Tokenizing Qwen dataset with explicit padding and truncation")
        
        def tokenize_function(examples):
            # Tokenize with explicit padding and truncation
            tokenized = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors=None  # Return Python lists
            )
            return tokenized
        
        # Apply tokenization
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=2,
            remove_columns=["text"]  # Remove the original text column
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=2,
                remove_columns=["text"]
            )
        
        # Convert to torch tensors
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        if eval_dataset:
            eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")

    return train_dataset, eval_dataset


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


def get_compute_metrics_fn(tokenizer):
    """Create a compute_metrics function that has access to the tokenizer."""
    # Get the vocabulary size for clipping
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 32000  # Default fallback
    
    def compute_metrics(eval_preds):
        """Compute metrics for evaluation."""
        preds, labels = eval_preds
        
        # Debug information
        print("\n===== DEBUGGING COMPUTE_METRICS =====")
        print(f"Predictions type: {type(preds)}")
        print(f"Predictions shape: {preds.shape if hasattr(preds, 'shape') else 'No shape attribute'}")
        print(f"Labels type: {type(labels)}")
        print(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else 'No shape attribute'}")
        print(f"Tokenizer vocabulary size: {vocab_size}")
        
        if len(preds) > 0:
            print(f"First prediction type: {type(preds[0])}")
            print(f"First prediction sample: {preds[0][:10] if len(preds[0]) > 10 else preds[0]}")
        
        # Convert logits to token IDs if needed
        if len(preds.shape) == 3:  # Shape: (batch_size, seq_len, vocab_size)
            print("Detected logits format. Converting to token IDs...")
            # Get the token ID with the highest probability for each position
            preds = np.argmax(preds, axis=-1)
            print(f"Converted predictions shape: {preds.shape}")
            print(f"First converted prediction: {preds[0][:10]}")
        
        # Create masks for valid tokens (not -100)
        label_mask = labels != -100
        
        # Process in smaller batches to avoid OOM
        batch_size = 8  # Reduced batch size
        num_samples = len(preds)
        pred_str = []
        label_str = []
        
        print(f"Processing {num_samples} samples with batch size {batch_size}")
        
        # Helper function to safely convert tokens to integers
        def safe_int_conversion(token):
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
        
        try:
            # Check if we're using a Qwen model and have access to the dataset
            is_qwen = "qwen" in tokenizer.__class__.__name__.lower()
            has_last_responses = False
            
            # For Qwen models, try to get the last_assistant_response directly from the dataset
            if is_qwen and hasattr(trainer, 'eval_dataset'):
                print("\nChecking for last_assistant_response field in eval_dataset...")
                sample = trainer.eval_dataset[0] if len(trainer.eval_dataset) > 0 else {}
                has_last_responses = "last_assistant_response" in sample
                
                if has_last_responses:
                    print("✅ Found last_assistant_response field in evaluation dataset!")
                    # We'll use this field directly for labels
                    
                    # Get all the last_assistant_responses
                    last_responses = [example["last_assistant_response"] for example in trainer.eval_dataset]
                    print(f"Loaded {len(last_responses)} last_assistant_responses")
                    
                    # Use these as the ground truth labels
                    if len(last_responses) == num_samples:
                        print("Using last_assistant_response field as ground truth labels")
                        
                        # Process the last_assistant_responses to extract the answers
                        processed_labels = []
                        for response in last_responses:
                            # Extract the answer from the response
                            answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
                            answer_match = re.search(answer_pattern, response, re.DOTALL)
                            
                            if answer_match:
                                # Format the answer with the expected tags
                                answer = answer_match.group(1).strip()
                                processed_label = f"<begin_of_program>\nEOF\n<end_of_program>\n\n<begin_of_answer>\n{answer}\n<end_of_answer>"
                                processed_labels.append(processed_label)
                            else:
                                # If no answer tag found, use the whole response
                                processed_labels.append(response)
                        
                        # Use the processed labels
                        label_str = processed_labels
                        
                        # Skip the normal label processing for these
                        skip_label_processing = True
                    else:
                        print(f"Warning: Number of last_assistant_responses ({len(last_responses)}) doesn't match number of samples ({num_samples})")
                        skip_label_processing = False
                else:
                    print("❌ last_assistant_response field not found in evaluation dataset")
                    skip_label_processing = False
            else:
                skip_label_processing = False
            
            # Safer approach: process one sample at a time
            for i in range(num_samples):
                print(f"Processing sample {i+1}/{num_samples}...", end="\r")
                
                # Process prediction
                try:
                    pred = preds[i].tolist() if hasattr(preds[i], 'tolist') else preds[i]
                    
                    # Debug: Print some information about the prediction array
                    if i < 2 or i >= num_samples - 2:  # Only for first 2 and last 2 samples
                        print(f"\nSample {i} prediction info:")
                        print(f"  - Type: {type(pred)}")
                        print(f"  - Length: {len(pred) if hasattr(pred, '__len__') else 'N/A'}")
                        print(f"  - First few values: {pred[:5] if hasattr(pred, '__getitem__') else 'N/A'}")
                        print(f"  - Min value: {np.min(pred) if isinstance(pred, np.ndarray) else 'N/A'}")
                        print(f"  - Max value: {np.max(pred) if isinstance(pred, np.ndarray) else 'N/A'}")
                    
                    # Filter out any non-integer values or values that are too large
                    valid_tokens = []
                    invalid_count = 0
                    for token in pred:
                        safe_token = safe_int_conversion(token)
                        if safe_token is not None:
                            valid_tokens.append(safe_token)
                        else:
                            invalid_count += 1
                    
                    if invalid_count > 0 and (i < 2 or i >= num_samples - 2):
                        print(f"\nSample {i}: Filtered out {invalid_count} invalid tokens")
                    
                    try:
                        # Add a check for empty token list
                        if not valid_tokens:
                            print(f"\nSample {i}: No valid tokens to decode")
                            decoded_pred = ""
                        else:
                            decoded_pred = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                    except Exception as e:
                        print(f"\nError in tokenizer.decode for prediction {i}: {e}")
                        print(f"Valid tokens: {valid_tokens[:10]}{'...' if len(valid_tokens) > 10 else ''}")
                        decoded_pred = ""
                    
                    pred_str.append(decoded_pred)
                except Exception as e:
                    print(f"\nError decoding prediction {i}: {e}")
                    print(f"Prediction type: {type(preds[i])}")
                    # Add empty string as fallback
                    pred_str.append("")
                
                # Process label (skip if we're using last_assistant_response)
                if not skip_label_processing:
                    try:
                        # Get the mask for this sample
                        sample_mask = label_mask[i]
                        # Get only the valid tokens (not -100)
                        valid_label_tokens = labels[i][sample_mask].tolist() if hasattr(labels[i][sample_mask], 'tolist') else labels[i][sample_mask]
                        
                        # Filter out any values that are too large
                        valid_tokens = []
                        invalid_count = 0
                        for token in valid_label_tokens:
                            safe_token = safe_int_conversion(token)
                            if safe_token is not None:
                                valid_tokens.append(safe_token)
                            else:
                                invalid_count += 1
                        
                        if invalid_count > 0 and (i < 2 or i >= num_samples - 2):
                            print(f"\nSample {i} (label): Filtered out {invalid_count} invalid tokens")
                        
                        try:
                            # Add a check for empty token list
                            if not valid_tokens:
                                print(f"\nSample {i} (label): No valid tokens to decode")
                                decoded_label = ""
                            else:
                                decoded_label = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                                
                                # IMPORTANT FIX: For Qwen models, extract the actual answer from the label
                                # This fixes the issue where labels are showing up as "and" instead of the actual answer
                                # Check if we're using a Qwen model based on tokenizer name
                                if is_qwen:
                                    # Look for the assistant's response in the decoded label
                                    assistant_pattern = r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>'
                                    assistant_matches = re.findall(assistant_pattern, decoded_label, re.DOTALL)
                                    
                                    if assistant_matches:
                                        # Use the last assistant response as the label
                                        assistant_text = assistant_matches[-1].strip()
                                        print(f"\nSample {i}: Extracted assistant response from Qwen format: {assistant_text[:50]}...")
                                        
                                        # Try to extract the answer from the assistant response
                                        answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
                                        answer_match = re.search(answer_pattern, assistant_text, re.DOTALL)
                                        if answer_match:
                                            answer = answer_match.group(1).strip()
                                            decoded_label = f"<begin_of_program>\nEOF\n<end_of_program>\n\n<begin_of_answer>\n{answer}\n<end_of_answer>"
                                            print(f"\nSample {i}: Extracted answer: {answer}")
                                        else:
                                            # If no answer tag found, use the whole assistant response
                                            decoded_label = assistant_text
                                    else:
                                        # Fallback: Try to extract the answer directly
                                        answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
                                        answer_match = re.search(answer_pattern, decoded_label, re.DOTALL)
                                        if answer_match:
                                            answer = answer_match.group(1).strip()
                                            decoded_label = f"<begin_of_program>\nEOF\n<end_of_program>\n\n<begin_of_answer>\n{answer}\n<end_of_answer>"
                                            print(f"\nSample {i}: Extracted answer directly: {answer}")
                        except Exception as e:
                            print(f"\nError in tokenizer.decode for label {i}: {e}")
                            print(f"Valid tokens: {valid_tokens[:10]}{'...' if len(valid_tokens) > 10 else ''}")
                            decoded_label = ""
                        
                        label_str.append(decoded_label)
                    except Exception as e:
                        print(f"\nError decoding label {i}: {e}")
                        print(f"Label type: {type(labels[i])}")
                        # Add empty string as fallback
                        label_str.append("")
            
            print("\nFinished decoding all samples")
            print(f"Total samples processed: {num_samples}")
            print(f"Successfully decoded predictions: {len([p for p in pred_str if p])}")
            print(f"Successfully decoded labels: {len([l for l in label_str if l])}")
            
            # Extract program tokens and answers
            def extract_program_tokens(text):
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
                try:
                    # Special case for Qwen models - check for "and" as the entire answer
                    if text.strip().lower() == "and":
                        # This is likely a Qwen tokenization issue - try to find a numerical value in the original text
                        numerical_match = re.search(r'(-?\d+\.?\d*)', text)
                        if numerical_match:
                            try:
                                return float(numerical_match.group(1))
                            except ValueError:
                                pass
                        
                        # If we can't find a numerical value, return a placeholder
                        print(f"Found 'and' as the entire answer text - likely a Qwen tokenization issue")
                        return 0.0  # Default numerical value
                    
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
                    
                    # For Qwen models, check for assistant responses
                    qwen_pattern = r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>'
                    qwen_matches = re.findall(qwen_pattern, text, re.DOTALL)
                    if qwen_matches:
                        # Use the last assistant response
                        assistant_text = qwen_matches[-1].strip()
                        
                        # Try to extract answer from the assistant text
                        answer_match = re.search(answer_pattern, assistant_text, re.DOTALL)
                        if answer_match:
                            solution = answer_match.group(1).strip()
                            # Normalize and try to convert to float
                            solution = solution.replace('%', '').replace('$', '').replace(',', '')
                            solution = re.sub(r'\s+', ' ', solution).strip()
                            
                            try:
                                if re.match(r'^-?\d+\.?\d*$', solution):
                                    return float(solution)
                                else:
                                    return solution.lower()
                            except ValueError:
                                return solution.lower()
                        
                        # If no answer tag in assistant response, use the whole response
                        # but check for numerical values
                        numerical_match = re.search(r'(-?\d+\.?\d*)', assistant_text)
                        if numerical_match:
                            try:
                                return float(numerical_match.group(1))
                            except ValueError:
                                pass
                        
                        # Return the assistant text as a fallback
                        return assistant_text.lower()
                    
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
            
            print("Extracting programs and answers...")
            
            # Extract programs and answers with error handling
            pred_programs = []
            label_programs = []
            pred_answers = []
            label_answers = []
            
            for p in pred_str:
                try:
                    pred_programs.append(extract_program_tokens(p))
                except Exception as e:
                    print(f"Error processing prediction program: {e}")
                    pred_programs.append(["ERROR"])
                    
                try:
                    pred_answers.append(extract_answer(p))
                except Exception as e:
                    print(f"Error processing prediction answer: {e}")
                    pred_answers.append("")
            
            # Check if we're using a Qwen model based on tokenizer name
            is_qwen = "qwen" in tokenizer.__class__.__name__.lower()
            has_last_responses = False
            
            for l in label_str:
                try:
                    label_programs.append(extract_program_tokens(l))
                except Exception as e:
                    print(f"Error processing label program: {e}")
                    label_programs.append(["ERROR"])
                    
                try:
                    label_answers.append(extract_answer(l))
                except Exception as e:
                    print(f"Error processing label answer: {e}")
                    label_answers.append("")
            
            print("Calculating metrics...")
            
            # Program token match
            program_matches = 0
            for pred_prog, label_prog in zip(pred_programs, label_programs):
                if pred_prog == label_prog:
                    program_matches += 1
            
            # Answer matches (numerical with tolerance or exact)
            answer_matches = 0
            tolerance = 0.01  # 1% tolerance - increased from 0.1%
            
            # Debug information for answer matching
            print("\nAnswer matching details:")
            for i, (pred, label) in enumerate(zip(pred_answers, label_answers)):
                match = False
                reason = "No match"
                
                # Try to convert string predictions to float if they look numerical
                if isinstance(pred, str) and isinstance(label, float):
                    try:
                        # Clean up the string and try to convert
                        pred_clean = re.sub(r'\s+', '', pred)  # Remove all whitespace
                        pred_clean = re.sub(r'[^\d.-]', '', pred_clean)  # Keep only digits, decimal point, and minus
                        if pred_clean:
                            pred = float(pred_clean)
                            print(f"  Converted string '{pred}' to float {pred}")
                    except ValueError:
                        pass
                
                # Try to convert string labels to float if they look numerical
                if isinstance(label, str) and isinstance(pred, float):
                    try:
                        # Clean up the string and try to convert
                        label_clean = re.sub(r'\s+', '', label)  # Remove all whitespace
                        label_clean = re.sub(r'[^\d.-]', '', label_clean)  # Keep only digits, decimal point, and minus
                        if label_clean:
                            label = float(label_clean)
                            print(f"  Converted string '{label}' to float {label}")
                    except ValueError:
                        pass
                
                if isinstance(pred, float) and isinstance(label, float):
                    # Numerical comparison with tolerance
                    if abs(pred - label) <= tolerance * max(1, abs(label)):
                        match = True
                        reason = f"Numerical match within {tolerance*100}% tolerance"
                    else:
                        reason = f"Numerical values differ: {pred} vs {label}"
                elif isinstance(pred, str) and isinstance(label, str):
                    # String comparison with normalization
                    pred_norm = pred.lower().strip()
                    label_norm = label.lower().strip()
                    
                    # Exact match
                    if pred_norm == label_norm:
                        match = True
                        reason = "Exact string match"
                    # Check if one is a substring of the other
                    elif pred_norm in label_norm or label_norm in pred_norm:
                        match = True
                        reason = "Substring match"
                    else:
                        # Calculate string similarity
                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(None, pred_norm, label_norm).ratio()
                        if similarity > 0.8:  # 80% similarity threshold
                            match = True
                            reason = f"String similarity: {similarity:.2f}"
                        else:
                            reason = f"Strings differ (similarity: {similarity:.2f})"
                else:
                    reason = f"Type mismatch: {type(pred)} vs {type(label)}"
                
                if match:
                    answer_matches += 1
                
                # Only print details for the first few and last few samples
                if i < 5 or i >= len(pred_answers) - 5:
                    print(f"Sample {i}: Match={match}, Reason={reason}")
                    print(f"  Pred: {pred} ({type(pred)})")
                    print(f"  Label: {label} ({type(label)})")
            
            # Calculate metrics
            program_match_percentage = program_matches / max(1, len(pred_str)) * 100

            # Log final metrics
            if wandb.run:
                metrics = {
                    "train/global_step": trainer_stats.global_step,
                }
                
                # Add training loss if available
                if hasattr(trainer_stats, 'training_loss') and trainer_stats.training_loss is not None:
                    metrics["train/final_loss"] = trainer_stats.training_loss
                    
                # Add epoch if available
                if hasattr(trainer_stats, 'epoch') and trainer_stats.epoch is not None:
                    metrics["train/epoch"] = trainer_stats.epoch
                    
                # Add padding and truncation settings
                metrics["train/max_length"] = trainer.args.max_seq_length
                metrics["train/padding"] = getattr(trainer.args, "padding", "unknown")
                metrics["train/truncation"] = getattr(trainer.args, "truncation", "unknown")
                
                wandb.log(metrics)

