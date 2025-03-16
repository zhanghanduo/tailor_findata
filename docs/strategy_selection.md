# 1. Brainstorm a proper strategy

- **Idea 1: Building a RAG System**  
  This involves merging all report segments into one big knowledge base and using retrieval to answer questions. It seems like overkill since each question already has its own context (text and table). Merging everything could be messy and time-consuming, and it might not help much if questions are self-contained per page. It’s like trying to read the whole library when you only need one book. It is good if the user ask a question that is beyond the context of the page, which is more likely to happen in a real-world scenario.

- **Idea 2: Fine-Tuning a Model on the Train Set**  
  This means training a model specifically for your dataset using examples from the train set. It’s likely the most reliable way, as it can learn the exact patterns, like how to generate the right program for answers, which involves numerical reasoning. It takes more effort and computer power, but it should give the best results. Think of it as customizing a tool for your specific job.

- **Idea 3: Using Modern LLMs Like Grok-3 or GPT-4o**  
  This is the quickest option—feeding the context and question to advanced AI models and asking them to answer. It’s easy and fast, but it might not be as accurate, especially for tricky number-based questions. It’s like asking a smart friend for help without much preparation—they might get it right, but not always.

### My Choice

I choose **Idea 2: Fine-tuning a model on the train set**. It seems likely to give the most accurate answers, especially since the task involves generating programs for numerical reasoning, which needs precision. While it requires more resources, it’s worth it for reliability. If you’re short on time, Idea 3 could work as a quick start, but expect some errors. Idea 1 feels unnecessary given the dataset’s structure.

## Comprehensive Analysis

This section provides a detailed evaluation of the three proposed ideas for obtaining all answers for the ConvFinQA test set, a conversational financial question-answering dataset grounded in financial reports. Each data example includes multi-turn conversations based on a single report page, with context comprising textual passages and a financial table. The goal is to generate programs representing the chain of numerical reasoning, as indicated by the dataset’s submission format. The analysis considers the dataset’s structure, the nature of the task, and the feasibility of each approach, drawing on available information from the dataset repository and related research.

### Dataset Context and Task Description

The ConvFinQA dataset, introduced in the 2022 EMNLP paper “ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering” ([ConvFinQA Paper](https://ar5iv.labs.arxiv.org/html/2210.03849)), is designed to study numerical reasoning in conversational QA over financial data. It contains 3,892 conversations with 14,115 questions, split into train (3,037 examples), dev (421 examples), and test (434 examples) sets at the conversation level, and additional turn-level files for finer granularity. Each example includes fields such as “pre_text” (text before the table), “post_text” (text after the table), “table” (the financial table), and “annotation” (containing the conversation and program details). The task requires generating a program that represents the reasoning steps, ending with “EOF”, as seen in the leaderboard submission format on Codalab ([ConvFinQA GitHub Repository](https://github.com/czyssrs/ConvFinQA)).

Given the focus on numerical reasoning and the specific output format, the task is challenging, requiring models to handle both textual and tabular data and generate precise programs. The dataset’s design, with each conversation grounded in a single page, suggests that the context provided for each example is sufficient for answering questions, reducing the need for broader document-level retrieval.

### Evaluation of Proposed Ideas

#### Idea 1: Building a RAG System with a Golden Knowledge Base

This approach involves processing the dataset (train, dev, and test sets) to merge segments into a “golden knowledge base” approximating the original financial report, then using Retrieval Augmented Generation (RAG) techniques for QA, handling both numeric and semantic queries.

- **Pros:**
  - Offers flexibility to handle questions requiring information from the entire report, potentially useful if some questions span multiple pages.
  - Provides a comprehensive knowledge base for retrieval, which could be beneficial for broader QA tasks beyond the test set.

- **Cons:**
  - Merging segments into a coherent knowledge base is complex and error-prone, especially given potential overlaps or misalignments in financial report pages.
  - Unnecessary for the current task, as each data example is grounded in a single page, and the provided context (pre_text, post_text, table) is likely sufficient.
  - Adds unnecessary complexity, as RAG systems are typically designed for scenarios where context is not pre-provided, whereas here, context is already given per example.
  - The generator component would still need to be trained or prompted to generate programs in the specific format (ending with “EOF”), which may not be inherently supported by standard RAG setups.

Given the dataset’s structure, where each conversation is self-contained within its page, this approach seems overkill. It introduces additional data processing challenges without clear benefits for the task of generating programs for the test set.

#### Idea 2: Fine-Tuning a Retriever and Generator Using the Train Set

This idea involves following the codebase to fine-tune a retriever and generator specifically for the ConvFinQA dataset, using annotations in the train set to ensure the model outputs the expected program format with correct patterns.

- **Pros:**
  - Tailored to the dataset, allowing the model to learn the specific language, numerical reasoning patterns, and output format (e.g., programs ending with “EOF”).
  - Utilizes the train set (3,037 examples) effectively, which is sufficiently large for fine-tuning, especially given the dataset’s focus on a niche domain (financial QA).
  - Likely to achieve high accuracy, as the model can be trained to handle both textual context and tabular data, and generate the required programs.
  - The codebase likely provides a framework, making implementation more straightforward, as seen in the GitHub repository ([ConvFinQA GitHub Repository](https://github.com/czyssrs/ConvFinQA)).

- **Cons:**
  - Requires computational resources and time for fine-tuning, which could be a barrier if resources are limited.
  - May not generalize well to new, unseen formats or types of questions if the train set does not cover all variations, though this is mitigated by the dataset’s design for financial reports.

Given the task’s requirements, this approach is well-suited. The generator can be fine-tuned to take the provided context (pre_text, post_text, table) and the question as input, outputting the program. The retriever component might be relevant for extracting specific information from the context (e.g., relevant table cells), but given the context is already provided, it may be integrated into the generator’s capabilities. This method aligns with standard practices for QA tasks with provided context, making it reliable for generating accurate programs.

#### Idea 3: Using Modern LLMs Like Grok-3 or GPT-4o with Proper Prompting

This approach leverages advanced large language models (LLMs) like Grok-3 or GPT-4o, feeding them the provided context for each data example and prompting them to directly generate the correct answer (program) in the required format.

- **Pros:**
  - Quick and easy to implement, requiring no fine-tuning, making it resource-efficient in terms of setup time.
  - Leverages the general knowledge and reasoning capabilities of modern LLMs, which, by 2025, are likely advanced in handling textual and tabular data, especially with financial contexts.
  - Can be done by crafting a prompt like: “Given the following context from a financial report page, which includes pre_text, post_text, and a table, and a question, generate a program that represents the steps to answer the question, ending with ‘EOF’.”

- **Cons:**
  - Accuracy may not be as high as a fine-tuned model, particularly for complex numerical reasoning tasks, as LLMs can make arithmetic errors or misinterpret financial data.
  - Requires careful prompt engineering to ensure the output matches the exact format (e.g., program steps ending with “EOF”), which may involve trial and error.
  - Cost could be prohibitive for processing a large test set (434 examples at the conversation level, with multiple turns), depending on the LLM’s pricing model.
  - Handling tables can be tricky, as LLMs may struggle with parsing and reasoning over tabular data without specific fine-tuning.

While this approach is convenient, its reliability is uncertain, especially given the task’s focus on generating precise programs for numerical reasoning. Modern LLMs in 2025, like Grok-3 or GPT-4o, may perform well, but they are likely less accurate than a fine-tuned model for this specific, domain-focused task.

### Comparative Analysis and Preference

To determine the best approach, consider the task’s requirements: generating programs for numerical reasoning in a specific format, with each example providing its own context. Idea 1 (RAG system) is unnecessary, as merging segments adds complexity without benefit, given the self-contained nature of each example. Idea 3 (using LLMs) is quick but risks lower accuracy, especially for numerical precision and format adherence. Idea 2 (fine-tuning) is the most reliable, leveraging the train set to learn the task’s specifics, though it requires more resources.

Given the importance of accuracy for leaderboard submissions or practical use, **Idea 2 is preferred**. It aligns with the dataset’s design and the need for precise program generation, ensuring the model learns the required patterns. While it demands more effort, it is the most effective for getting all answers for the test set. If resources are constrained, Idea 3 could serve as a starting point, but expect potential errors, particularly with complex numerical reasoning.

# 2. Model Selection

For this specific task, a function call approach would be highly beneficial for several reasons:

1. **The task has a well-defined set of operations** (subtract, divide, etc.)
2. **Output format is strictly defined** (list of tokens ending with EOF)
3. **Consistency is critical** for evaluation purposes
4. **Error reduction** is important for financial calculations

## Function Calls vs. Instruction-Based LLMs for Financial Data Processing

### Advantages of Function Call Approach

1. **Structured Output Format**
   - Function calls enforce a strict output structure, ensuring the model produces exactly the format required (tokens like "subtract(", "divide(", etc.)
   - The expected output format is a list of dictionaries with "id" and "predicted" fields, where "predicted" is a list of program tokens ending with "EOF"

2. **Limited Operation Set**
   - The task involves a finite set of mathematical operations (subtract, divide, etc.)
   - Function calls can be constrained to only these valid operations, eliminating the possibility of the model inventing new operations

3. **Parameter Validation**
   - Function calls can enforce parameter validation (e.g., ensuring numeric arguments for mathematical operations)
   - This reduces errors where the model might try to perform operations on non-numeric data

4. **Improved Consistency**
   - Function calls provide a consistent interface for the model to interact with
   - This reduces variability in how the model expresses mathematical operations

5. **Easier Parsing**
   - The structured nature of function calls makes post-processing simpler
   - No need for complex regex patterns to extract operations from free-form text

### Disadvantages of Function Call Approach

1. **Training Complexity**
   - Fine-tuning an LLM with function call capability may require more specialized training data
   - You'll need examples that demonstrate the correct function call format for each operation

2. **Reduced Flexibility**
   - If new operations need to be added, the function call interface must be updated
   - This could require retraining the model or updating the function definitions

3. **Potential Overhead**
   - Function call parsing adds a layer of complexity to the model's processing
   - This could potentially impact inference speed for very simple operations

### Implementation Considerations

1. **Function Definition Structure**
   - Define clear function schemas for each mathematical operation (subtract, divide, etc.)
   - Include parameter types and descriptions to guide the model

2. **Training Data Preparation**
   - Convert existing examples to use the function call format
   - Ensure a diverse set of examples covering all operations and edge cases

3. **Evaluation Metrics**
   - Track not just the final answer accuracy but also the correctness of function call usage
   - Monitor for cases where the model attempts to use undefined functions

```
subtract(arg1: number, arg2: number) -> number
divide(arg1: number, arg2: number) -> number
multiply(arg1: number, arg2: number) -> number
add(arg1: number, arg2: number) -> number
```

This approach would likely yield more consistent and accurate results compared to a purely instruction-based approach, especially for complex multi-step calculations common in financial analysis.
