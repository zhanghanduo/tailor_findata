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


def compute_metrics(eval_preds):
    """Compute metrics for evaluation."""
    preds, labels = eval_preds
    
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Process in smaller batches to avoid OOM
    batch_size = 16
    num_samples = len(preds)
    pred_str = []
    label_str = []
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        pred_str.extend(tokenizer.batch_decode(preds[i:end_idx], skip_special_tokens=True))
        label_str.extend(tokenizer.batch_decode(labels[i:end_idx], skip_special_tokens=True))
    
    # Extract program tokens and answers
    def extract_program_tokens(text):
        program_pattern = r'<begin_of_program>\s*(.*?)\s*<end_of_program>'
        match = re.search(program_pattern, text, re.DOTALL)
        if match:
            program_text = match.group(1).strip()
            return program_text.split()
        return ["EOF"]
    
    def extract_answer(text):
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
    
    # Extract programs and answers
    pred_programs = [extract_program_tokens(p) for p in pred_str]
    label_programs = [extract_program_tokens(l) for l in label_str]
    
    pred_answers = [extract_answer(p) for p in pred_str]
    label_answers = [extract_answer(l) for l in label_str]
    
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
    program_match_percentage = program_matches / len(pred_str) * 100
    answer_match_percentage = answer_matches / len(pred_str) * 100
    
    # Log examples to wandb
    if wandb.run:
        examples_table = wandb.Table(columns=["Prediction", "Reference", "Pred Program", "Label Program", "Pred Answer", "Label Answer"])
        for p, l, pp, lp, pa, la in list(zip(pred_str, label_str, pred_programs, label_programs, pred_answers, label_answers))[:5]:
            examples_table.add_data(p, l, str(pp), str(lp), str(pa), str(la))
        wandb.log({"eval_examples": examples_table})
    
    return {
        "program_match": program_match_percentage,
        "answer_match": answer_match_percentage,
    }


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, args):
    """Set up the SFT trainer."""
    # Ensure max_steps is an integer
    max_steps = args.max_steps if args.max_steps > 0 else -1
    
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
        compute_metrics=compute_metrics if eval_dataset else None,
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
    
    # Test model
    test_results = test_model(model_path, args)
    
    # Finish wandb run
    if wandb.run:
        wandb.finish()
    
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    # Make tokenizer globally accessible for compute_metrics
    tokenizer = None
    main()

