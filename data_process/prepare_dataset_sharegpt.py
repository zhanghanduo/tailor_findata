import json
import os
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
# from transformers import AutoTokenizer, AutoConfig
import re
from bs4 import BeautifulSoup


def format_table_for_llm(html_table):
    """Convert HTML table to a markdown-style text table that's easier for LLMs to understand"""
    soup = BeautifulSoup(html_table, 'html.parser')
    
    # Extract all rows
    rows = soup.find_all('tr')
    if not rows:
        return "No table data found."
    
    # Process the table data
    table_data = []
    for row in rows:
        cells = row.find_all('td')
        # Skip the first cell if it's just a row number
        if cells and cells[0].text.strip().isdigit():
            row_data = [cell.text.strip() for cell in cells[1:]]
        else:
            row_data = [cell.text.strip() for cell in cells]
        table_data.append(row_data)
    
    # Format as a clean text table
    if not table_data:
        return "Empty table."
    
    # Determine column widths for proper alignment
    col_widths = [max(len(row[i]) for row in table_data if i < len(row)) 
                    for i in range(max(len(row) for row in table_data))]
    
    # Create header row with column names
    header_row = table_data[0]
    header_separator = ['-' * width for width in col_widths]
    
    # Format the table with proper alignment
    formatted_table = []
    formatted_table.append(' | '.join(header_row[i].ljust(col_widths[i]) for i in range(len(header_row))))
    formatted_table.append(' | '.join(header_separator[i] for i in range(len(header_separator))))
    
    # Add data rows
    for row in table_data[1:]:
        formatted_table.append(' | '.join(row[i].ljust(col_widths[i]) if i < len(row) else ' ' * col_widths[i] 
                                        for i in range(len(col_widths))))
    
    return '\n'.join(formatted_table)


def get_convfinqa_dataset(json_file):
    """
    Convert ConvFinQA dataset to ShareGPT format with proper multi-turn conversations.
    Includes structured thinking process using step_list information.
    """
    conversations = []
    
    # System prompt for structured thinking
    system_prompt = """Your role as an assistant involves exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive process to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include analysing or relevant findings. In the Solution section, based on various attempts from the Thought section, systematically present the final solution that you deem correct. The solution should be the final answer (information provided or numeric value calculated), formatted as follows: <|begin_of_solution|> {final precise and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"""
    
    samples = json.load(open(json_file))
    for sample in samples:
        annos = sample['annotation']

        # Extract context information
        pre_text = annos['amt_pre_text']
        post_text = annos['amt_post_text']
        html_table = annos['amt_table'].replace('<td></td>', '<td>-</td>')
        formatted_table = format_table_for_llm(html_table)
        
        # Create the financial context
        financial_context = f'{pre_text}\n\n{formatted_table}\n\n{post_text}\n'
        
        # Get dialogue components
        questions = annos['dialogue_break']
        answers = annos['exe_ans_list']
        
        # Get step_list if available
        step_list = annos.get('step_list', [])
        
        # Create conversation with proper turns
        conversation = []
        
        # Add system message
        conversation.append({"role": "system", "content": system_prompt})
        
        # First turn includes the financial context
        first_question = f"I'm looking at some financial data. Here's the context:\n\n{financial_context}\n\n{questions[0]}"
        conversation.append({"role": "human", "content": first_question})
        
        # For the first answer, use step_list if available
        if len(step_list) > 0 and 0 < len(questions):
             # Find how many steps are needed for the first turn
            first_turn_steps = []
            current_step_index = 0
            
            # If turn_program is available, use it to determine how many steps to include
            if 'turn_program' in annos and len(annos['turn_program']) > 0:
                # Parse the first turn program to determine how many operations it uses
                first_program = annos['turn_program'][0]
                operations = first_program.split(',')
                steps_for_first_turn = len(operations)
                
                # Get only the steps needed for the first turn
                first_turn_steps = step_list[:steps_for_first_turn]
            else:
                # If we can't determine precisely, use a reasonable default (e.g., first step only)
                first_turn_steps = [step_list[0]] if step_list else []
             
            # Create thought process from the appropriate steps
            thought_process = "\n\n".join([f"Step {i+1}: {step}" for i, step in enumerate(first_turn_steps)])
            structured_answer = f"<|begin_of_thought|>\n{thought_process}\n<|end_of_thought|>\n\n<|begin_of_solution|>\n{answers[0]}\n<|end_of_solution|>"
            conversation.append({"role": "assistant", "content": structured_answer})
        else:
            # Fallback if no step_list
            conversation.append({"role": "assistant", "content": f"<|begin_of_thought|>\nLet me analyze the financial data to find the answer.\n<|end_of_thought|>\n\n<|begin_of_solution|>\n{answers[0]}\n<|end_of_solution|>"})
        
        # Add remaining turns
        for i in range(1, len(questions)):
            conversation.append({"role": "human", "content": questions[i]})
            
            # For subsequent turns, determine which steps to include based on turn_program
            if 'turn_program' in annos and i < len(annos['turn_program']):
                program = annos['turn_program'][i]
                
                # If we can determine which steps to use for this turn
                if 'step_list' in annos and len(step_list) > 0:
                    # Calculate which steps correspond to this turn
                    # This is a simplified approach - you may need to adjust based on your data structure
                    prev_operations_count = sum(len(p.split(',')) for p in annos['turn_program'][:i])
                    current_operations = program.split(',')
                    current_operations_count = len(current_operations)
                    
                    # Get steps for this turn
                    turn_steps = step_list[prev_operations_count:prev_operations_count + current_operations_count]
                    if turn_steps:
                        thought_process = "\n\n".join([f"Step {j+1}: {step}" for j, step in enumerate(turn_steps)])
                        structured_answer = f"<|begin_of_thought|>\n{thought_process}\n<|end_of_thought|>\n\n<|begin_of_solution|>\n{answers[i]}\n<|end_of_solution|>"
                    else:
                        # Fallback to using the program as thought
                        thought = f"I need to {program.replace(',', ' and then ')}"
                        structured_answer = f"<|begin_of_thought|>\n{thought}\n<|end_of_thought|>\n\n<|begin_of_solution|>\n{answers[i]}\n<|end_of_solution|>"
                else:
                    thought = f"I need to {program.replace(',', ' and then ')}"
                    structured_answer = f"<|begin_of_thought|>\n{thought}\n<|end_of_thought|>\n\n<|begin_of_solution|>\n{answers[i]}\n<|end_of_solution|>"
            else:
                structured_answer = f"<|begin_of_thought|>\nLet me continue analyzing the financial data.\n<|end_of_thought|>\n\n<|begin_of_solution|>\n{answers[i]}\n<|end_of_solution|>"
            
            conversation.append({"role": "assistant", "content": structured_answer})
        
        conversations.append({"conversations": conversation})
    
    return Dataset.from_list(conversations)


def main():
    train_file = 'data/train_turn.json'
    test_file = 'data/dev_turn.json'
    
    train_dataset = get_convfinqa_dataset(train_file)
    test_dataset = get_convfinqa_dataset(test_file)
    
    convfinqa_dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    convfinqa_dataset.save_to_disk('fingpt-convfinqa-sharegpt')
    # print(convfinqa_dataset)
    
    # Print a sample conversation to verify format
    if len(convfinqa_dataset['train']) > 0:
        print("\nSample conversation:")
        for turn in convfinqa_dataset['train'][0]['conversations'][:4]:  # Print first two turns only
            print(f"\n{turn['role'].upper()}:\n{turn['content']}")


if __name__ == "__main__":
    main()
