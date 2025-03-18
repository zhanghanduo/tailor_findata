# Financial RAG System Evaluation

This repository contains tools for evaluating a Retrieval-Augmented Generation (RAG) system on financial question-answering tasks. The evaluation framework is designed to test a RAG system's ability to handle multi-turn dialogue interactions where questions build upon previous context.

## Overview

The system consists of several components organized in a clean directory structure:

1. **RAG System** (`rag/` directory):
   - `rag_system.py` - Core RAG system implementation that:
     - Retrieves relevant context based on query IDs
     - Generates answers using a language model (mocked for demonstration)

2. **Evaluation Framework** (`evaluation/` directory):
   - `evaluator.py` - Core evaluation logic:
     - Processes questions turn-by-turn as in a real conversation
     - Extracts answers from system responses
     - Compares against gold standard answers
     - Calculates accuracy metrics
   - `run.py` - Script to run the evaluation process

3. **Examples** (`examples/` directory):
   - `openai_integration.py` - Example showing how to integrate with OpenAI's API

4. **Main Runner** (`run.py`) - Script to run the evaluation from the root directory

## Project Structure

```
.
├── data/                     # Data directory
│   ├── sample_train.json     # Multi-turn dialogue examples
│   └── sample_knowledge.json # Knowledge base for the RAG system
├── rag/                      # RAG system implementation
│   ├── __init__.py           # Package initialization
│   └── rag_system.py         # RAG system implementation
├── evaluation/               # Evaluation framework
│   ├── __init__.py           # Package initialization
│   ├── evaluator.py          # Core evaluation logic
│   └── run.py                # Evaluation runner
├── examples/                 # Example integrations
│   └── openai_integration.py # OpenAI API integration example
├── results/                  # Evaluation results
├── run.py                    # Main runner script
├── setup.sh                  # Setup script
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Data Format

The evaluation uses two main data files:

- `sample_train.json` - Contains multi-turn dialogues with questions, gold answers, and annotations
- `sample_knowledge.json` - Contains the knowledge base for the RAG system

### Sample Train Format

The training data contains examples with:
- `pre_text` and `post_text` - Text before and after a table
- `table` - Tabular data
- `qa` - Question and gold answer
- `annotation` - Detailed annotations including:
  - `dialogue_break` - Questions broken down into conversation turns
  - `answer_list` - Gold answers for each turn
  - Additional annotations like program steps

### Sample Knowledge Format

The knowledge data contains:
- `id` - Unique identifier for the knowledge item
- `content` - The content to be retrieved
- `metadata` - Additional information about the source

## Metrics

The evaluation calculates several metrics:

- **Overall Turn Accuracy** - Percentage of all turns answered correctly
- **Average Example Accuracy** - Average of per-example accuracies
- **First Turn Accuracy** - Percentage of first turns answered correctly
- **Last Turn Accuracy** - Percentage of final answers that are correct (most important)

## Usage

### Prerequisites

- Python 3.6+
- Required packages: `numpy`, `tqdm`, `requests`, `python-dotenv`, `regex`

### Setup

1. Clone this repository
2. Run the setup script:
   ```bash
   ./setup.sh
   ```
   This will:
   - Create a virtual environment
   - Install dependencies
   - Set up necessary directories
   - Copy sample data files if available

3. Alternatively, manually install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Evaluation

To run the evaluation with default settings:

```bash
python run.py
```

Additional options:

```bash
python run.py --train_path <path_to_train> --knowledge_path <path_to_knowledge> --output_path <path_to_output> --verbose
```

- `--train_path`: Path to the training data JSON (default: `data/sample_train.json`)
- `--knowledge_path`: Path to the knowledge data JSON (default: `data/sample_knowledge.json`)
- `--output_path`: Path to save evaluation results (default: `results/evaluation_results.json`)
- `--verbose`: Print detailed evaluation information

### Integrating with OpenAI

To run the evaluation with OpenAI's API:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Run the OpenAI integration example
python examples/openai_integration.py
```

### Integrating Your Own RAG System

To integrate your own RAG system:

1. Create a new class that extends `RAGSystem` in the `rag` package
2. Override the `generate_answer` method to use your preferred LLM
3. Update the `query_rag` function to use your custom implementation

## Prompt Design

The RAG system uses a carefully designed prompt to ensure that the language model:

1. Provides concise, direct answers
2. Formats responses with `<answer>` tags for easy extraction
3. Only uses information from the provided context
4. Takes into account conversation history

Example prompt:

```
You are an expert financial analyst assistant. Answer the following question based ONLY on the provided context.
Be concise and to the point. For numerical answers, provide just the number or percentage without explanation.

CONTEXT:
{context}

QUESTION:
{question}

PREVIOUS CONVERSATION:
{history}

Answer with just the final result. Format your answer like this:
<answer>Your concise answer here</answer>
```

## Extending the Evaluation

You can extend this evaluation framework by:

1. Implementing more sophisticated answer comparison logic
2. Adding more metrics (e.g., precision/recall for complex answers)
3. Integrating with different LLM providers
4. Adding support for more complex dialogue patterns

## License

This project is licensed under the MIT License - see the LICENSE file for details.
