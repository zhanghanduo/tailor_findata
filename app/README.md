# Financial Document QA System

This project demonstrates a prototype system that can answer questions based on financial documents (texts, tables, figures, etc.) using Large Language Models (LLMs).

## Overview

The system uses the ConvFinQA dataset, which contains financial documents with pre-text, post-text, tables, and question-answer pairs. The system can:

1. Parse and display financial documents
2. Answer questions about the financial data
3. Show the calculation steps used to derive the answer
4. Generate a final.json file with the answers

## Files

- `main.py`: Streamlit application using OpenAI API
- `local_model.py`: Streamlit application using local models
- `generate_final.py`: Command-line script to generate final.json using local models
- `create_final_json.py`: Simple script to create final.json with the specific example

## Example

The system can answer questions like:

> What was the percentage change in the net cash from operating activities from 2008 to 2009?

And provide the answer:

> 14.1%

Along with the calculation steps:

```
subtract(206588, 181001), divide(#0, 181001)
```

## Usage

### Using the Streamlit App with OpenAI API

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
```

### Using the Streamlit App with Local Models

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/local_model.py
```

### Generating final.json with Local Models

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python app/generate_final.py --model meta-llama/Llama-2-7b-chat-hf --dataset data/sample_train.json --output app/final.json
```

### Creating final.json with the Specific Example

```bash
# Run the script
python app/create_final_json.py
```

## Requirements

- Python 3.8+
- Streamlit
- Transformers
- PyTorch
- LangChain (for OpenAI API version)

## Dataset

The system uses the ConvFinQA dataset, which contains financial documents with:
- Pre-text: Text before the table
- Table: Financial data in tabular format
- Post-text: Text after the table
- Question-Answer pairs: Questions about the financial data and their answers 