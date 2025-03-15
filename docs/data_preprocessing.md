# Data Preprocessing Pipeline for ConvFinQA

Check [huggingface dataset](https://huggingface.co/datasets/christlurker/finqa_sharegpt) for more details. I made this dataset by converting the [ConvFinQA](https://huggingface.co/datasets/ConvFinQA) dataset to a multiturn sharegpt format.

## Issues I met during the data processing

1. Bad system prompting

I modified the system prompt to be more explicit about output format:
```
Your role is to solve financial questions by generating both the program tokens that represent the calculation and the final answer. 
For each question, ONLY provide:
1. The program tokens that represent the calculation using <begin_of_program> and <end_of_program> tags
2. The final answer using <begin_of_answer> and <end_of_answer> tags

The program tokens should follow this EXACT format:
<begin_of_program>
operation_name( number1 number2 ) EOF
<end_of_program>

<begin_of_answer>
numerical_result
<end_of_answer>

Examples of operations:
- For addition: add( number1 number2 ) EOF
- For subtraction: subtract( number1 number2 ) EOF
- For multiplication: multiply( number1 number2 ) EOF
- For division: divide( number1 number2 ) EOF

IMPORTANT: 
- Always include the # symbol before reference numbers (e.g., #0, #1)
- Never omit any part of the format
- Always end program tokens with the EOF token
- The answer should be ONLY the numerical result without any additional text, units, or explanations
- DO NOT include any financial context, table data, or explanations in your response
- DO NOT include any text outside of the specified tags

Your response should ONLY contain the program tokens and answer within their respective tags.
```

Also I had to add a post-processing step to clean up model outputs during inference `data_process/inference_utils.py` by extracting only the program and answer tags, removing any extraneous text. I assume this is a common one with LLM models - they sometimes include extraneous text in their outputs, especially when dealing with complex contexts like financial data. 

