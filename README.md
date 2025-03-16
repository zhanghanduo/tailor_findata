# Fine-Tuning phi-4 for Financial Question Answering

This repository is dedicated to enhancing the performance of the [phi-4 model](https://huggingface.co/microsoft/phi-4) through supervised fine-tuning. The goal is to improve the model's ability to retrieve and generate answers for questions in the ConvFinQA test set, a dataset focused on conversational financial question answering.

## Strategy Overview

The chosen strategy involves fine-tuning the phi-4 model using the train set of the ConvFinQA dataset. This approach is selected for its potential to deliver the most accurate results, particularly in generating programs for numerical reasoning tasks. Fine-tuning allows the model to learn specific patterns and nuances of the dataset, ensuring precision and reliability in its outputs.

## Why Fine-Tuning?

- **Accuracy**: Fine-tuning on the train set enables the model to learn the exact patterns required for generating correct answers, especially for tasks involving numerical reasoning.
- **Customization**: Tailors the model to the specific requirements of the ConvFinQA dataset, improving its performance in this niche domain.
- **Reliability**: While resource-intensive, this method is expected to yield the best results in terms of accuracy and consistency.

## Why Phi-4?

The Phi-4 model exhibits robust logical reasoning and strong capabilities in STEM (Science, Technology, Engineering, and Mathematics) fields; however, its performance in code generation and instruction following is relatively mediocre. Its default output is notably neutral and sterile. So I just choose it as the base model as this task has higher requirement in the multi-turn and instruction following fields.

## Project Structure

```
├── data/                  # Raw ConvFinQA dataset files
├── data_process/          # Scripts for data processing
│   ├── prepare_dataset_convfinqa.py  # Converts raw data to training format
│   ├── generation_pipeline.py        # Pipeline for financial QA generation
│   ├── inference_example.py          # Example script for model inference
│   └── inference_utils.py            # Utility functions for inference
├── processed_data/        # Processed datasets ready for training
├── train/                 # Training scripts
│   └── train_phi4.py      # Main training script for phi-4
├── evaluate/              # Evaluation scripts
│   ├── run_inference.py   # Script to run inference on test set
│   ├── post_process.py    # Process model outputs for evaluation
│   ├── general_utils.py   # Utilities for evaluation
│   └── test_post_process.py # Tests for post-processing
├── docs/                  # Documentation
├── requirements.txt       # Project dependencies
└── run_training.sh        # Shell script to run training with optimized settings
```

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Data Processing

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

### Training

To train a model on the processed dataset, you can use the provided shell script:

```bash
./run_training.sh
```

Or run the training script directly with custom parameters:

```bash
python train/train_phi4.py \
  --model_name "unsloth/phi-4" \
  --dataset_name "christlurker/finqa_sharegpt" \
  --output_dir "outputs/phi4-convfinqa" \
  --max_seq_length 4096 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 2 \
  --eval_steps 5 \
  --save_steps 20
```

### Inference and Generation

#### Using the Generation Pipeline

The `generation_pipeline.py` provides a convenient way to use the fine-tuned model for financial question answering:

```python
from data_process.generation_pipeline import FinancialQAPipeline

# Initialize the pipeline with your fine-tuned model
pipeline = FinancialQAPipeline("path/to/your/model")

# Example financial context
financial_context = """
Item | 2007 | 2008 | 2009
-----|------|------|------
Revenue | 100.5 | 120.3 | 95.7
Expenses | 80.2 | 85.6 | 70.3
"""

# Ask a question
result = pipeline("What was the difference between revenue and expenses in 2008?", 
                  context=financial_context)

# Get the program tokens and answer
print("Program:", result["program"])
print("Answer:", result["answer"])
```

#### Running Inference on Test Set

To run inference on the ConvFinQA test set and format the predictions for evaluation:

```bash
python evaluate/run_inference.py \
  --model_path "outputs/phi4-finqa/final_model" \
  --test_data "data/test_private.json" \
  --output_dir "predictions" \
  --batch_size 1 \
  --max_length 4096 \
  --max_new_tokens 512 \
  --load_in_4bit
```

This script:
1. Loads the fine-tuned model
2. Prepares the test data with proper formatting of tables and context
3. Applies the system prompt to guide the model to generate outputs in the expected format:
   - Program tokens with `<begin_of_program>` and `<end_of_program>` tags
   - Final answers with `<begin_of_answer>` and `<end_of_answer>` tags
4. Runs inference on each test example
5. Extracts the assistant's response and saves both raw and formatted predictions for evaluation

The expected output format from the model is:
```
<begin_of_program>
operation_name( number1 number2 ) EOF
<end_of_program>

<begin_of_answer>
numerical_result
<end_of_answer>
```

For example:
```
<begin_of_program>
subtract( 120.3 85.6 ) EOF
<end_of_program>

<begin_of_answer>
34.7
<end_of_answer>
```

#### Post-Processing Model Outputs

If you already have model outputs and need to format them for evaluation:

```bash
python evaluate/post_process.py \
  --model_outputs "path/to/model_outputs.json" \
  --output_file "formatted_predictions.json" \
  --dataset_file "data/test.json"
```

### Evaluation

To evaluate the formatted predictions against the gold standard:

```bash
python evaluate/general_utils.py \
  --json_in "predictions/formatted_predictions.json" \
  --json_ori "data/test.json" \
  --all_res_file "all_results.json" \
  --error_file "error_cases.json" \
  --program_mode "flat"
```

This script calculates:
1. **Execution Accuracy**: Whether the program executes to the correct answer
2. **Program Accuracy**: Whether the predicted program is semantically equivalent to the gold program

## Example Workflow

1. Process the dataset:
   ```bash
   python data_process/prepare_dataset_convfinqa.py
   ```

2. Train the model:
   ```bash
   ./run_training.sh
   ```

3. Run inference and format predictions:
   ```bash
   python evaluate/run_inference.py --model_path "outputs/phi4-convfinqa/final_model" --test_data "data/test.json"
   ```

4. Evaluate the predictions:
   ```bash
   python evaluate/general_utils.py --json_in "predictions/formatted_predictions.json" --json_ori "data/test.json"
   ```

## Using the Model for Inference

For quick testing of the model, you can use the inference example script:

```bash
python data_process/inference_example.py "path/to/your/model"
```

This script demonstrates how to:
1. Load the model and tokenizer
2. Format prompts with financial context
3. Generate responses
4. Extract program tokens and answers from the model output
