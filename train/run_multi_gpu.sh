#!/bin/bash

# Run training with separate GPU for evaluation
python train/train_phi4.py \
    --model_name="unsloth/phi-4" \
    --dataset_name="christlurker/finqa_sharegpt" \
    --output_dir="outputs/phi4_multi_gpu" \
    --batch_size=2 \
    --eval_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-4 \
    --num_train_epochs=3 \
    --eval_steps=20 \
    --save_steps=40 \
    --wandb_project="phi4-multi-gpu" \
    --wandb_run_name="phi4-separate-eval-gpu" \
    --train_device="cuda:0" \
    --eval_device="cuda:1" \
    --separate_eval_gpu \
    --max_eval_samples=100 