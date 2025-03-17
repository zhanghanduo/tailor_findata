import os
import json
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import re

# Set page configuration
st.set_page_config(
    page_title="Financial Document QA",
    page_icon="💰",
    layout="wide"
)

# Load the ConvFinQA dataset
@st.cache_data
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Format table for display
def format_table(table_data):
    if not table_data:
        return "No table data available"
    
    # Convert to markdown table
    table_md = ""
    for i, row in enumerate(table_data):
        if i == 0:
            table_md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
            table_md += "| " + " | ".join(["---"] * len(row)) + " |\n"
        else:
            table_md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
    
    return table_md

# Define the output parser
class FinancialAnswer(BaseModel):
    program: str = Field(description="The calculation program used to derive the answer")
    answer: str = Field(description="The final numerical answer")
    explanation: Optional[str] = Field(description="Explanation of the calculation")

# Main application
def main():
    st.title("Financial Document QA System")
    
    # Sidebar
    st.sidebar.title("Settings")
    dataset_path = st.sidebar.selectbox(
        "Select Dataset",
        ["data/train.json", "data/dev.json", "data/sample_train.json"],
        index=2  # Default to sample_train.json
    )
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
        st.sidebar.success(f"Loaded {len(dataset)} examples from {dataset_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")
        return
    
    # Example selection
    example_idx = st.sidebar.number_input(
        "Select Example Index",
        min_value=0,
        max_value=len(dataset)-1,
        value=0
    )
    
    # Display selected example
    if example_idx < len(dataset):
        example = dataset[example_idx]
        
        # Display document content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Document Text")
            
            # Pre-text
            if "pre_text" in example:
                st.markdown("**Pre Text:**")
                st.markdown("\n".join(example["pre_text"]))
            
            # Post-text
            if "post_text" in example:
                st.markdown("**Post Text:**")
                st.markdown("\n".join(example["post_text"]))
        
        with col2:
            st.subheader("Financial Table")
            if "table_ori" in example:
                st.markdown(format_table(example["table_ori"]))
        
        # Question answering section
        st.subheader("Question Answering")
        
        # Get question from example or let user input
        default_question = example.get("qa", {}).get("question", "") if "qa" in example else ""
        question = st.text_input("Question:", value=default_question)
        
        if st.button("Answer Question"):
            if question:
                with st.spinner("Generating answer..."):
                    answer = answer_question(example, question)
                    
                    # Display answer
                    st.success(f"Answer: {answer['answer']}")
                    
                    # Display calculation
                    with st.expander("See calculation"):
                        st.code(answer["program"])
                        
                    # Display explanation if available
                    if answer.get("explanation"):
                        with st.expander("See explanation"):
                            st.markdown(answer["explanation"])
                    
                    # If ground truth is available, show it
                    if "qa" in example and "answer" in example["qa"]:
                        st.info(f"Ground Truth Answer: {example['qa']['answer']}")
            else:
                st.warning("Please enter a question")

# Function to answer questions
def answer_question(document, question):
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        st.warning("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return {"program": "", "answer": "API key not set", "explanation": ""}
    
    # Create context from document
    context = ""
    
    # Add pre-text
    if "pre_text" in document:
        context += "Document Text:\n" + "\n".join(document["pre_text"]) + "\n\n"
    
    # Add table
    if "table_ori" in document:
        context += "Financial Table:\n"
        for row in document["table_ori"]:
            context += " | ".join(str(cell) for cell in row) + "\n"
        context += "\n"
    
    # Add post-text
    if "post_text" in document:
        context += "\n" + "\n".join(document["post_text"])
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0,
        api_key=api_key
    )
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst assistant that answers questions about financial documents.
        
For each question, you need to:
1. Analyze the financial data provided
2. Generate the calculation program that represents the steps to solve the problem
3. Calculate the final answer
4. Provide a brief explanation

Your response should include:
- The program tokens that represent the calculation using <begin_of_program> and <end_of_program> tags
- The final answer using <begin_of_answer> and <end_of_answer> tags

The program tokens should follow this format:
<begin_of_program>
operation_name(number1, number2), operation_name(#0, number3)
<end_of_program>

Where:
- operation_name can be: add, subtract, multiply, divide
- #0, #1, etc. refer to the results of previous operations
- All numbers should be extracted directly from the document

Example:
<begin_of_program>
subtract(206588, 181001), divide(#0, 181001)
<end_of_program>

<begin_of_answer>
14.1%
<end_of_answer>

Be precise with your calculations and format the answer appropriately (include % for percentages, $ for dollar amounts, etc.).
"""),
        ("user", "Context: {context}\n\nQuestion: {question}")
    ])
    
    # Create the chain
    chain = prompt_template | llm
    
    # Run the chain
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    # Extract program and answer from response
    program_match = re.search(r'<begin_of_program>(.*?)<end_of_program>', response.content, re.DOTALL)
    answer_match = re.search(r'<begin_of_answer>(.*?)<end_of_answer>', response.content, re.DOTALL)
    
    program = program_match.group(1).strip() if program_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Extract explanation (anything after the answer tag)
    explanation = ""
    if answer_match:
        remaining_text = response.content[answer_match.end():]
        if remaining_text.strip():
            explanation = remaining_text.strip()
    
    return {
        "program": program,
        "answer": answer,
        "explanation": explanation
    }

if __name__ == "__main__":
    main() 