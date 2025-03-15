import os
import argparse
import wandb
import re
import numpy as np
import torch
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
    parser.add_argument("--train_device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--eval_device", type=str, default="cuda:1", help="Device to use for evaluation")
    parser.add_argument("--separate_eval_gpu", action="store_true", default=False, help="Whether to use a separate GPU for evaluation")
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
        "train_device": args.train_device,
        "eval_device": args.eval_device if args.separate_eval_gpu else None,
        "separate_eval_gpu": args.separate_eval_gpu,
    }
    
    run_name = args.wandb_run_name or f"phi4-lora-r{args.lora_r}-bs{args.batch_size*args.gradient_accumulation_steps}"
    wandb.init(project=args.wandb_project, name=run_name, config=config)
    return wandb.run


def load_model_and_tokenizer(args, device=None):
    """Load the base model and tokenizer."""
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        device_map=device if device else "auto",  # Specify device if provided
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
        
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
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
                        print(f"  - Clipping negative token {int_val} to 0")
                        return 0
                    return int_val
                elif isinstance(token, float) and token.is_integer():
                    # For floats that are integers
                    int_val = int(token)
                    # Clip to valid vocabulary range if needed
                    if int_val >= vocab_size:
                        print(f"  - Clipping token {int_val} to vocab size {vocab_size-1}")
                        return vocab_size - 1
                    elif int_val < 0:
                        print(f"  - Clipping negative token {int_val} to 0")
                        return 0
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
                    
                    if invalid_count > 0:
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
                    label = labels[i].tolist() if hasattr(labels[i], 'tolist') else labels[i]
                    # Filter out any non-integer values or values that are too large
                    valid_tokens = []
                    invalid_count = 0
                    for token in label:
                        safe_token = safe_int_conversion(token)
                        if safe_token is not None:
                            valid_tokens.append(safe_token)
                        else:
                            invalid_count += 1
                    
                    if invalid_count > 0:
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
                    answer_pattern = r'<begin_of_answer>\s*(.*?)\s*<end_of_answer>'
                    match = re.search(answer_pattern, text, re.DOTALL)
                    if match:
                        solution = match.group(1).strip()
                        # Normalize numerical values
                        solution = solution.replace('%', '')
                        try:
                            # Convert to float for numerical comparison
                            return float(solution)
                        except:
                            # If not a number, return as is
                            return solution
                    return text.strip()
                except Exception as e:
                    print(f"Error extracting answer: {e}")
                    return ""
            
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
            tolerance = 0.001  # 0.1% tolerance
            
            for pred, label in zip(pred_answers, label_answers):
                if isinstance(pred, float) and isinstance(label, float):
                    # Numerical comparison with tolerance
                    if abs(pred - label) <= tolerance * max(1, abs(label)):
                        answer_matches += 1
                elif pred == label:
                    answer_matches += 1
            
            # Calculate metrics
            program_match_percentage = program_matches / max(1, len(pred_str)) * 100
            answer_match_percentage = answer_matches / max(1, len(pred_str)) * 100
            
            print(f"Program match: {program_match_percentage:.2f}%, Answer match: {answer_match_percentage:.2f}%")
            
            # Log examples to wandb
            if wandb.run:
                try:
                    examples_table = wandb.Table(columns=["Prediction", "Reference", "Pred Program", "Label Program", "Pred Answer", "Label Answer"])
                    for p, l, pp, lp, pa, la in list(zip(pred_str, label_str, pred_programs, label_programs, pred_answers, label_answers))[:5]:
                        examples_table.add_data(p, l, str(pp), str(lp), str(pa), str(la))
                    wandb.log({"eval_examples": examples_table})
                except Exception as e:
                    print(f"Error logging to wandb: {e}")
            
            return {
                "program_match": program_match_percentage,
                "answer_match": answer_match_percentage,
            }
            
        except Exception as e:
            print(f"Critical error in compute_metrics: {e}")
            import traceback
            traceback.print_exc()
            # Return default metrics to avoid breaking the training loop
            return {
                "program_match": 0.0,
                "answer_match": 0.0,
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
    )
    
    # Configure to train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>human<|im_sep|>",
        response_part="<|im_start|>assistant<|im_sep|>",
    )
    
    return trainer


def train_model(trainer, model, args):
    """Train the model and return training stats."""
    print("Starting training...")
    
    # If using separate GPU for evaluation, modify the trainer's evaluation strategy
    if args.separate_eval_gpu:
        # Disable automatic evaluation during training
        trainer.args.evaluation_strategy = "no"
        
        # Custom training loop with manual evaluation on separate GPU
        total_steps = trainer.args.max_steps if trainer.args.max_steps > 0 else \
                     (len(trainer.train_dataset) // (trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps)) * trainer.args.num_train_epochs
        
        eval_steps = args.eval_steps
        current_step = 0
        
        # Start training
        while current_step < total_steps:
            # Train for eval_steps
            next_step = min(current_step + eval_steps, total_steps)
            trainer.args.max_steps = next_step
            partial_train_stats = trainer.train(resume_from_checkpoint=current_step > 0)
            current_step = next_step
            
            # Evaluate on separate GPU
            if trainer.is_world_process_zero() and args.eval_split_percentage > 0:
                print(f"Step {current_step}/{total_steps}: Running evaluation on separate GPU")
                evaluate_on_separate_gpu(trainer, model, args)
        
        trainer_stats = partial_train_stats
    else:
        # Standard training with automatic evaluation
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
        dtype="bfloat16",
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


def evaluate_on_separate_gpu(trainer, model, args):
    """Evaluate the model on a separate GPU to avoid CUDA OOM issues."""
    print(f"Moving model to evaluation device: {args.eval_device}")
    
    # Store the original device
    original_device = next(model.parameters()).device
    
    # Move model to evaluation device
    with torch.no_grad():  # Prevent gradient storage to save memory
        model.to(args.eval_device)
        
        # Run evaluation
        print("Starting evaluation on separate GPU...")
        evaluation_results = trainer.evaluate()
        
        # Log evaluation results
        if wandb.run:
            for key, value in evaluation_results.items():
                wandb.log({f"eval/{key}": value})
        
        # Move model back to original device
        print(f"Moving model back to training device: {original_device}")
        model.to(original_device)
    
    return evaluation_results


def main():
    """Main function to run the training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Ensure max_steps is properly set
    if args.max_steps is None or args.max_steps <= 0:
        args.max_steps = -1
    
    # Check if separate GPU evaluation is possible
    if args.separate_eval_gpu:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Disabling separate GPU evaluation.")
            args.separate_eval_gpu = False
        elif torch.cuda.device_count() < 2:
            print(f"Warning: Only {torch.cuda.device_count()} GPU available. Disabling separate GPU evaluation.")
            args.separate_eval_gpu = False
    
    print(f"Training configuration:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Train batch size: {args.batch_size}")
    print(f"  - Eval batch size: {args.eval_batch_size}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  - Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  - Max eval samples: {args.max_eval_samples}")
    print(f"  - Max sequence length: {args.max_seq_length}")
    print(f"  - Training device: {args.train_device}")
    if args.separate_eval_gpu:
        print(f"  - Evaluation device: {args.eval_device}")
    
    # Setup wandb
    wandb_run = setup_wandb(args)
    
    # Load model and tokenizer on the training device
    model, tokenizer = load_model_and_tokenizer(args, device=args.train_device if args.separate_eval_gpu else None)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(tokenizer, args)
    
    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, args)
    
    # Train model
    trainer_stats = train_model(trainer, model, args)
    
    # Save model
    model_path = save_model(model, tokenizer, args)
    
    # Test model
    test_results = test_model(model_path, args)
    
    # Finish wandb run
    if wandb.run:
        wandb.finish()
    
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()

