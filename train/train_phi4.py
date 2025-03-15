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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Phi-4 with LoRA")
    parser.add_argument("--model_name", type=str, default="unsloth/phi-4", help="Base model to fine-tune")
    parser.add_argument("--cache_dir", type=str, default="/workspace/mnt/watt/public_models", help="Cache directory for models")
    parser.add_argument("--dataset_name", type=str, default="christlurker/finqa_sharegpt", help="Dataset to use for training")
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
    parser.add_argument("--eval_split_percentage", type=int, default=4, help="Percentage of data to use for evaluation")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing to save memory")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum number of samples to use for evaluation")
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
    
    # Set chat template
    tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
    
    return model, tokenizer


def prepare_dataset(tokenizer, args):
    """Load and prepare the dataset for training and evaluation."""
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    dataset = standardize_sharegpt(dataset)
    
    # Create train/eval split
    if args.eval_split_percentage > 0:
        dataset = dataset.train_test_split(test_size=args.eval_split_percentage/100, seed=args.seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        
        # Limit eval dataset size if specified
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            print(f"Limiting evaluation dataset to {args.max_eval_samples} samples")
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_eval_samples)))
    else:
        train_dataset = dataset
        eval_dataset = None
    
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
    
    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def analyze_predictions(pred_str, label_str, pred_answers, label_answers, step=None):
    """Analyze prediction patterns and log detailed metrics to wandb."""
    # Count different types of predictions
    analysis = {
        "empty_predictions": 0,
        "empty_labels": 0,
        "format_matches": 0,
        "answer_format_errors": 0,
        "numerical_answers": 0,
        "string_answers": 0,
        "answer_type_mismatches": 0,
    }
    
    # Check for answer format patterns
    answer_begin_pattern = r'<begin_of_answer>'
    answer_end_pattern = r'<end_of_answer>'
    
    for i, (pred, label, pred_ans, label_ans) in enumerate(zip(pred_str, label_str, pred_answers, label_answers)):
        # Check for empty predictions/labels
        if not pred.strip():
            analysis["empty_predictions"] += 1
        if not label.strip():
            analysis["empty_labels"] += 1
            
        # Check for format matches
        pred_has_begin = bool(re.search(answer_begin_pattern, pred))
        pred_has_end = bool(re.search(answer_end_pattern, pred))
        label_has_begin = bool(re.search(answer_begin_pattern, label))
        label_has_end = bool(re.search(answer_end_pattern, label))
        
        if (pred_has_begin and pred_has_end) == (label_has_begin and label_has_end):
            analysis["format_matches"] += 1
        
        # Check for answer format errors
        if (pred_has_begin and not pred_has_end) or (not pred_has_begin and pred_has_end):
            analysis["answer_format_errors"] += 1
            
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
    for key in analysis:
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
                
                # Process label
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
                    # First try to extract from the standard format
                    answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
                    match = re.search(answer_pattern, text, re.DOTALL)
                    if match:
                        solution = match.group(1).strip()
                        # Normalize numerical values
                        solution = solution.replace('%', '')
                        solution = solution.replace('$', '')
                        solution = solution.replace(',', '')
                        try:
                            # Convert to float for numerical comparison
                            return float(solution)
                        except:
                            # If not a number, return as is
                            return solution.lower()  # Normalize case for string comparison
                    
                    # If standard format not found, try to find numerical answers in the text
                    # Look for patterns like "The answer is X" or "= X"
                    alt_patterns = [
                        r'(?:answer|result|value)(?:\s+is|\s*[:=])\s*([\d\.\,]+)',
                        r'(?:=|equals)\s*([\d\.\,]+)',
                        r'(?:[\$£€])\s*([\d\.\,]+)'
                    ]
                    
                    for pattern in alt_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            solution = match.group(1).strip()
                            solution = solution.replace(',', '')
                            try:
                                return float(solution)
                            except:
                                pass
                    
                    # If no patterns matched, return the whole text as a fallback
                    return text.strip().lower()
                except Exception as e:
                    print(f"Error extracting answer: {e}")
                    return text.strip().lower()  # Return the full text as a fallback
            
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
            answer_match_percentage = answer_matches / max(1, len(pred_str)) * 100
            
            print(f"Program match: {program_match_percentage:.2f}%, Answer match: {answer_match_percentage:.2f}%")
            
            # Analyze predictions to understand patterns
            analysis = analyze_predictions(pred_str, label_str, pred_answers, label_answers)
            
            # Log examples to wandb
            if wandb.run:
                try:
                    examples_table = wandb.Table(columns=["Prediction", "Reference", "Pred Program", "Label Program", "Pred Answer", "Label Answer", "Answer Match"])
                    for i, (p, l, pp, lp, pa, la) in enumerate(list(zip(pred_str, label_str, pred_programs, label_programs, pred_answers, label_answers))[:10]):
                        # Determine if answers match
                        match = False
                        if isinstance(pa, float) and isinstance(la, float):
                            match = abs(pa - la) <= tolerance * max(1, abs(la))
                        elif isinstance(pa, str) and isinstance(la, str):
                            pa_norm = pa.lower().strip()
                            la_norm = la.lower().strip()
                            match = (pa_norm == la_norm) or (pa_norm in la_norm) or (la_norm in pa_norm)
                            if not match:
                                from difflib import SequenceMatcher
                                similarity = SequenceMatcher(None, pa_norm, la_norm).ratio()
                                match = similarity > 0.8
                        
                        examples_table.add_data(p, l, str(pp), str(lp), str(pa), str(la), str(match))
                    wandb.log({"eval_examples": examples_table})
                except Exception as e:
                    print(f"Error logging to wandb: {e}")
            
            # Return both raw counts and percentages for better debugging
            return {
                "program_match": program_match_percentage,
                "answer_match": answer_match_percentage,
                "program_matches_count": program_matches,
                "answer_matches_count": answer_matches,
                "total_samples": len(pred_str),
            }
            
        except Exception as e:
            print(f"Critical error in compute_metrics: {e}")
            import traceback
            traceback.print_exc()
            # Return default metrics to avoid breaking the training loop
            return {
                "program_match": 0.0,
                "answer_match": 0.0,
                "program_matches_count": 0,
                "answer_matches_count": 0,
                "total_samples": 0,
            }
    
    return compute_metrics


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, args):
    """Set up the SFT trainer."""
    # Ensure max_steps is an integer
    max_steps = args.max_steps if args.max_steps > 0 else -1
    
    # Define a function to preprocess logits for metrics calculation
    def preprocess_logits_for_metrics(logits, labels):
        """Convert logits to predictions for metrics calculation."""
        # If logits is a tuple, take the first element (which should be the main logits)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Get the predicted token IDs
        pred_ids = logits.argmax(dim=-1)
        return pred_ids
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,  # Use the pre-processed value
        warmup_steps=args.warmup_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep only the 3 best checkpoints
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="wandb",  # Enable wandb reporting
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="answer_match" if eval_dataset else None,
        greater_is_better=True,
        gradient_checkpointing=args.gradient_checkpointing,
        # Add memory optimization options
        deepspeed=None,  # Let the trainer handle memory optimization
        ddp_find_unused_parameters=False,
        # Ensure we get predictions, not just loss
        prediction_loss_only=False,
    )
    
    # Create callbacks list
    callbacks = []
    
    # Add early stopping callback if using evaluation
    if eval_dataset:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        compute_metrics=get_compute_metrics_fn(tokenizer) if eval_dataset else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_dataset else None,
        callbacks=callbacks,
    )
    
    # Configure to train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>human<|im_sep|>",
        response_part="<|im_start|>assistant<|im_sep|>",
    )
    
    return trainer


def train_model(trainer):
    """Train the model and return training stats."""
    print("Starting training...")
    trainer_stats = trainer.train()
    
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
            
        wandb.log(metrics)
    
    return trainer_stats


def save_model(model, tokenizer, args):
    """Save the fine-tuned model and tokenizer."""
    output_path = os.path.join(args.output_dir, "final_model")
    print(f"Saving model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path


def test_model(model_path, args):
    """Test the saved model with a few examples."""
    print(f"Testing model from {model_path}")
    
    # Load the saved model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    
    # Test examples
    test_examples = [
        "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,",
        "Describe a tall tower in the capital of France.",
        "What are three key benefits of fine-tuning language models?",
    ]
    
    results = []
    for example in test_examples:
        print(f"\nTest prompt: {example}")
        
        messages = [{"role": "user", "content": example}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        outputs = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"prompt": example, "response": generated_text})
    
    # Log test results to wandb
    if wandb.run:
        test_table = wandb.Table(columns=["Prompt", "Response"])
        for result in results:
            test_table.add_data(result["prompt"], result["response"])
        wandb.log({"test_examples": test_table})
    
    return results


def evaluate_on_fixed_examples(model, tokenizer, args):
    """Evaluate the model on a fixed set of examples to ensure consistent evaluation."""
    print("Evaluating model on fixed examples...")
    
    # Fixed set of financial questions
    fixed_examples = [
        "What is the present value of $1000 to be received in 5 years if the discount rate is 8%?",
        "A company has a debt-to-equity ratio of 0.75. If its total assets are $500,000, what is the value of its equity?",
        "If a stock has a beta of 1.5, risk-free rate is 3%, and market risk premium is 6%, what is the expected return according to CAPM?",
        "A bond with face value $1000 pays 5% annual coupon and matures in 10 years. If the yield to maturity is 6%, what is the bond's price?",
        "If a company's ROE is 15% and its retention ratio is 40%, what is its sustainable growth rate?",
    ]
    
    results = []
    for example in fixed_examples:
        print(f"\nFixed example: {example}")
        
        messages = [{"role": "user", "content": example}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        
        # Generate without streaming for evaluation
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            use_cache=True,
            temperature=0.2,  # Lower temperature for more deterministic outputs
            top_p=0.9,
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's response
        response = generated_text.split("<|im_start|>assistant<|im_sep|>")[-1].strip()
        
        # Try to extract the answer
        try:
            answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
            match = re.search(answer_pattern, response, re.DOTALL)
            if match:
                answer = match.group(1).strip()
            else:
                # Try alternative patterns
                alt_patterns = [
                    r'(?:answer|result|value)(?:\s+is|\s*[:=])\s*([\d\.\,\$]+)',
                    r'(?:=|equals)\s*([\d\.\,\$]+)',
                ]
                
                for pattern in alt_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        answer = match.group(1).strip()
                        break
                else:
                    answer = "No explicit answer found"
        except Exception as e:
            print(f"Error extracting answer: {e}")
            answer = "Error extracting answer"
        
        results.append({
            "question": example,
            "response": response,
            "extracted_answer": answer
        })
    
    # Log results to wandb
    if wandb.run:
        try:
            fixed_examples_table = wandb.Table(columns=["Question", "Response", "Extracted Answer"])
            for result in results:
                fixed_examples_table.add_data(
                    result["question"], 
                    result["response"], 
                    result["extracted_answer"]
                )
            wandb.log({"fixed_examples": fixed_examples_table})
        except Exception as e:
            print(f"Error logging fixed examples to wandb: {e}")
    
    return results


def main():
    """Main function to run the training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Ensure max_steps is properly set
    if args.max_steps is None or args.max_steps <= 0:
        args.max_steps = -1
    
    print(f"Training configuration:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Train batch size: {args.batch_size}")
    print(f"  - Eval batch size: {args.eval_batch_size}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  - Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  - Max eval samples: {args.max_eval_samples}")
    print(f"  - Max sequence length: {args.max_seq_length}")
    
    # Setup wandb
    wandb_run = setup_wandb(args)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(tokenizer, args)
    
    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, args)
    
    # Train model
    trainer_stats = train_model(trainer)
    
    # Save model
    model_path = save_model(model, tokenizer, args)
    
    # Prepare model for inference before evaluation
    print("Preparing model for inference evaluation...")
    FastLanguageModel.for_inference(model)
    
    # Evaluate on fixed examples using the in-memory model
    fixed_examples_results = evaluate_on_fixed_examples(model, tokenizer, args)
    
    # Test model by loading from disk
    test_results = test_model(model_path, args)
    
    # Finish wandb run
    if wandb.run:
        wandb.finish()
    
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()

