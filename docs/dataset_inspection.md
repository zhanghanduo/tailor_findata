# Dataset Inspection

## Data Structure

[ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering](https://arxiv.org/abs/2209.08795) is a conversational financial QA dataset built on financial reports. Each data example is a multi-turn conversation grounded in a single financial report page. The context from the report includes textual passages and a financial table, provided separately as "pre_text" (text before the table), "table", and "post_text" (text after the table).

This means each conversation has a mix of unstructured text (e.g. narrative) and structured data (numeric table). The dataset also provides detailed annotations per turn – including the question text, the ground truth reasoning program (a sequence of operations like add, subtract, etc.), and the executed answer value for that turn. These annotations indicate whether a question is a direct number lookup or requires computations.

The ConvFinQA dataset is organized in two primary formats, each serving different analytical purposes.



The conversation-level format contains complete dialogues and includes three main files: train.json (3,037 examples), dev.json (421 examples), and test.json (434 examples). Each entry in these files contains several standardized fields that provide context for the conversation. These fields include pre_text (text before the table), post_text (text after the table), table (the tabular data), and id (a unique identifier for each example). The most significant component is the annotation field, which contains the core information about the conversation, including the original program, dialogue turns, ground truth programs, and execution results.

The turn-level format breaks down conversations into individual question-answer pairs, resulting in a larger number of examples: train_turn.json (11,104 examples), dev_turn.json (1,490 examples), and test_turn.json (1,521 examples). This format adds several fields specific to each turn, such as cur_program (the program for the current turn), cur_dial (list of questions up to the current turn), exe_ans (execution result), cur_type (question type), turn_ind (turn index), and gold_ind (supporting facts).

The ConvFinQA dataset's two formats offer distinct advantages, depending on specific retrieval and generation objectives:

## Conversation-level format:

   - More context for each conversation, including the pre_text, post_text, and table.
   - More complete information about the conversation, including the original program, dialogue turns, ground truth programs, and execution results.

## Turn-level format:

   - Duplicate context for each turn, which is redundant.
   - More information about each turn, including the current program, dialogue history, execution result, question type, turn index, and supporting facts.

So conversation-level format is more suitable for generation system implementation, as it provides more context for each conversation.
