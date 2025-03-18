import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from inference_utils import clean_model_output, extract_program_and_answer

class FinancialQAPipeline:
    """
    A pipeline for financial question answering that automatically cleans model outputs.
    """
    
    def __init__(self, model_path, device="auto"):
        """
        Initialize the pipeline with a model and tokenizer.
        
        Args:
            model_path (str): Path to the model directory or HuggingFace model ID
            device (str): Device to use for inference ("auto", "cpu", "cuda", etc.)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        # Create the generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1
        )
    
    def format_prompt(self, question, context=None):
        """
        Format the prompt for the model.
        
        Args:
            question (str): The question to ask
            context (str, optional): Financial context to include
            
        Returns:
            str: Formatted prompt
        """
        system_prompt = """Your role is to solve financial questions by generating both the program tokens that represent the calculation and the final answer. 
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
"""
        
        if context:
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nI'm looking at some financial data. Here's the context:\n\n{context}\n\n{question} [/INST]"
        else:
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{question} [/INST]"
        
        return prompt
    
    def __call__(self, question, context=None, return_raw=False):
        """
        Generate a response to a financial question.
        
        Args:
            question (str): The question to answer
            context (str, optional): Financial context to include
            return_raw (bool): Whether to return the raw model output
            
        Returns:
            dict: Dictionary with program and answer
        """
        # Format the prompt
        prompt = self.format_prompt(question, context)
        
        # Generate response
        raw_output = self.pipe(prompt)[0]["generated_text"]
        
        # Extract the generated part (after the prompt)
        generated_text = raw_output[len(prompt):]
        
        # Clean the output
        cleaned_output = clean_model_output(generated_text)
        
        # Extract program and answer
        result = extract_program_and_answer(cleaned_output)
        
        if return_raw:
            result["raw_output"] = generated_text
            result["cleaned_output"] = cleaned_output
        
        return result


def main():
    """
    Example usage of the FinancialQAPipeline.
    """
    # Initialize the pipeline
    model_path = "your/model/path"  # Replace with your model path
    pipeline = FinancialQAPipeline(model_path)
    
    # Example financial context
    financial_context = """
    Here's a table showing financial data:
    
    Item | 2007 | 2008 | 2009
    -----|------|------|------
    Revenue | 100.5 | 120.3 | 95.7
    Expenses | 80.2 | 85.6 | 70.3
    Net Income | 20.3 | 34.7 | 25.4
    Assets | 500.0 | 550.0 | 580.0
    Liabilities | 300.0 | 320.0 | 330.0
    Equity | 200.0 | 230.0 | 250.0
    """
    
    # Example questions
    questions = [
        "What was the revenue in 2008?",
        "What was the difference between revenue and expenses in 2008?",
        "What was the percentage increase in revenue from 2007 to 2008?"
    ]
    
    # Process each question
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        
        # First question includes context, subsequent questions don't
        context = financial_context if i == 0 else None
        
        # Get response with raw output for demonstration
        result = pipeline(question, context, return_raw=True)
        
        print("\nRaw model output:")
        print(result["raw_output"])
        
        print("\nCleaned output:")
        print(result["cleaned_output"])
        
        print("\nExtracted program:", result["program"])
        print("Extracted answer:", result["answer"])
        
        print("-" * 80)


if __name__ == "__main__":
    main() 