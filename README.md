# Fine-Tuning phi-4 for Financial Question Answering

This repository is dedicated to enhancing the performance of the [phi-4 model](https://huggingface.co/microsoft/phi-4) through supervised fine-tuning. The goal is to improve the model's ability to retrieve and generate answers for questions in the ConvFinQA test set, a dataset focused on conversational financial question answering.

## Strategy Overview

The chosen strategy involves fine-tuning the phi-4 model using the train set of the ConvFinQA dataset. This approach is selected for its potential to deliver the most accurate results, particularly in generating programs for numerical reasoning tasks. Fine-tuning allows the model to learn specific patterns and nuances of the dataset, ensuring precision and reliability in its outputs.

## Why Fine-Tuning?

- **Accuracy**: Fine-tuning on the train set enables the model to learn the exact patterns required for generating correct answers, especially for tasks involving numerical reasoning.
- **Customization**: Tailors the model to the specific requirements of the ConvFinQA dataset, improving its performance in this niche domain.
- **Reliability**: While resource-intensive, this method is expected to yield the best results in terms of accuracy and consistency.

## Getting Started

To get started with fine-tuning the phi-4 model, follow the instructions provided in the repository. Ensure you have access to the ConvFinQA dataset and the necessary computational resources for training.

For more details on the strategy and analysis, refer to the `docs/strategy_selection.md` file.

# ConvFinQA Evaluation

This repository contains scripts for training and evaluating models on the ConvFinQA dataset, a conversational financial QA dataset that requires generating program tokens for numerical reasoning.

## Dataset Processing

The dataset processing script converts the ConvFinQA dataset into a format suitable for training models to generate both reasoning steps and program tokens.

```bash
python data_process/prepare_dataset_convfinqa.py
```

This script:
1. Processes the ConvFinQA dataset (train.json, dev.json, test.json)
2. Formats the data to include:
   - Reasoning steps with `<begin_of_thought>` and `<end_of_thought>` tags (Not implemented yet, need a powerful LLM to provide detailed chain of thoughts with brainstorming, traceback and reflection process.)
   - Program tokens with `<begin_of_program>` and `<end_of_program>` tags
   - Final answers with `<begin_of_answer>` and `<end_of_answer>` tags
3. Saves the processed datasets to the specified output directory

## Training

To train a model on the processed dataset:

```bash
python train/train_phi4.py \
  --model_name "unsloth/phi-4" \
  --dataset_name "christlurker/finqa_sharegpt" \
  --output_dir "outputs/phi4-convfinqa" \
  --max_seq_length 8192 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --max_steps 1000 \
  --eval_split_percentage 5
```

## Inference and Evaluation

To run inference on the test set and format the predictions for evaluation:

```bash
python evaluate/run_inference.py \
  --model_path "outputs/phi4-convfinqa/final_model" \
  --test_data "data/test.json" \
  --output_dir "predictions" \
  --batch_size 1 \
  --max_length 8192 \
  --max_new_tokens 512 \
  --load_in_4bit
```

## Post-Processing

```bash
python evaluate/post_process.py \
  --model_outputs "path/to/model_outputs.json" \
  --output_file "formatted_predictions.json" \
  --dataset_file "data/test.json"
```

This script:
1. Extracts program tokens from model outputs
2. Formats them according to the ConvFinQA evaluation requirements
3. Saves the formatted predictions to the specified output file

## Evaluation Format

The ConvFinQA evaluation expects predictions in the following format:

```json
[
  {
    "id": "ETR/2016/page_23.pdf-2",
    "predicted": ["subtract(", "5829", "5735", ")", "EOF"]
  },
  {
    "id": "INTC/2015/page_41.pdf-4",
    "predicted": ["divide(", "8.1", "56.0", ")", "EOF"]
  }
]
```

Each prediction is a dictionary with:
- `id`: The example ID
- `predicted`: A list of program tokens ending with "EOF"

## Evaluation Metrics

The evaluation script calculates two main metrics:
1. **Execution Accuracy**: Whether the program executes to the correct answer
2. **Program Accuracy**: Whether the predicted program is semantically equivalent to the gold program

To run the evaluation:

```bash
python evaluate/general_utils.py \
  --json_in "formatted_predictions.json" \
  --json_ori "data/test.json" \
  --all_res_file "all_results.json" \
  --error_file "error_cases.json" \
  --program_mode "flat"
```

## Example Workflow

1. Process the dataset:
   ```bash
   python data_process/prepare_dataset_convfinqa.py
   ```

2. Train the model:
   ```bash
   python train/train_phi4.py --model_name "unsloth/phi-4" --dataset_name "processed_data/convfinqa_program_tokens"
   ```

3. Run inference and format predictions:
   ```bash
   python evaluate/run_inference.py --model_path "outputs/phi4-convfinqa/final_model" --test_data "data/test.json"
   ```

4. Evaluate the predictions:
   ```bash
   python evaluate/general_utils.py --json_in "predictions/formatted_predictions.json" --json_ori "data/test.json"
   ```
