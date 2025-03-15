#!/bin/bash

# Run distributed training with DeepSpeed on 8 GPUs
deepspeed --num_gpus=8 \
    train/train_phi4.py \
    --batch_size=2 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-4 \
    --num_train_epochs=1 \
    --output_dir=outputs/phi4_deepspeed \
    --wandb_project=phi4-deepspeed \
    --wandb_run_name=phi4-8gpu-deepspeed \
    --deepspeed=train/ds_config.json 