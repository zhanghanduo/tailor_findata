#!/bin/bash

# Run training with memory optimization settings
python train/train_phi4.py \
  --model_name "unsloth/phi-4" \
  --dataset_name "christlurker/finqa_sharegpt" \
  --output_dir "outputs/phi4-finqa" \
  --batch_size 2 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --max_eval_samples 50 \
  --eval_steps 20 \
  --save_steps 40 \
  --max_seq_length 4096 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --wandb_project "phi4-finqa-sft" \
  --wandb_run_name "phi4-finqa-memory-optimized" 