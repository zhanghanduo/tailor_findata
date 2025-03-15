import json
from post_process import extract_program_tokens, extract_answer, format_predictions_for_evaluation

# Sample model output with program tokens
sample_output = """
<begin_of_step>
Looking at the financial data, I need to find the difference between the total assets in 2016 and 2015.

From the table, I can see:
- Total assets in 2016: 5829
- Total assets in 2015: 5735

I need to subtract the 2015 value from the 2016 value.
<end_of_step>

<begin_of_program>
subtract( 5829 5735 ) EOF
<end_of_program>

<begin_of_answer>
94
<end_of_answer>
"""

# Sample model output with nested program
sample_output_nested = """
<begin_of_step>
I need to calculate the percentage of goodwill to total assets.

From the table:
- Goodwill: 8.1
- Total assets: 56.0

I'll divide the goodwill by total assets and convert to percentage.
<end_of_step>

<begin_of_program>
divide( 8.1 56.0 ) EOF
<end_of_program>

<begin_of_answer>
14.5%
<end_of_answer>
"""

# Sample model output with more complex program
sample_output_complex = """
<begin_of_step>
I need to calculate the average of revenue for the years 2014, 2015, and 2016.

From the table:
- Revenue in 2014: 55.9
- Revenue in 2015: 59.4
- Revenue in 2016: 62.8

I'll add these values and then divide by 3.
<end_of_step>

<begin_of_program>
divide( add( add( 55.9 59.4 ) 62.8 ) 3 ) EOF
<end_of_program>

<begin_of_answer>
59.37
<end_of_answer>
"""

# Test extraction functions
def test_extraction():
    print("Testing program token extraction...")
    
    # Test basic program
    program_tokens = extract_program_tokens(sample_output)
    print(f"Basic program tokens: {program_tokens}")
    
    # Test nested program
    nested_program_tokens = extract_program_tokens(sample_output_nested)
    print(f"Nested program tokens: {nested_program_tokens}")
    
    # Test complex program
    complex_program_tokens = extract_program_tokens(sample_output_complex)
    print(f"Complex program tokens: {complex_program_tokens}")
    
    print("\nTesting answer extraction...")
    
    # Test basic answer
    answer = extract_answer(sample_output)
    print(f"Basic answer: {answer}")
    
    # Test percentage answer
    percentage_answer = extract_answer(sample_output_nested)
    print(f"Percentage answer: {percentage_answer}")
    
    # Test decimal answer
    decimal_answer = extract_answer(sample_output_complex)
    print(f"Decimal answer: {decimal_answer}")


# Test formatting for evaluation
def test_formatting():
    print("\nTesting formatting for evaluation...")
    
    # Sample predictions and IDs
    predictions = [sample_output, sample_output_nested, sample_output_complex]
    example_ids = ["ETR/2016/page_23.pdf-2", "INTC/2015/page_41.pdf-4", "AAPL/2016/page_15.pdf-3"]
    
    # Format predictions
    formatted_predictions = format_predictions_for_evaluation(predictions, example_ids)
    
    # Print formatted predictions
    print(json.dumps(formatted_predictions, indent=2))
    
    # Save to file for inspection
    with open("sample_formatted_predictions.json", "w") as f:
        json.dump(formatted_predictions, f, indent=2)
    
    print(f"Formatted predictions saved to sample_formatted_predictions.json")


if __name__ == "__main__":
    test_extraction()
    test_formatting() 