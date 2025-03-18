import os
import sys
import json
import requests
from typing import List, Tuple, Dict, Any

# Add the parent directory to the path so we can import from the main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from evaluate_rag import load_data, evaluate_dialogue, calculate_metrics

class OpenAIRAGSystem(RAGSystem):
    """
    RAG system implementation that uses OpenAI's API for the language model.
    """
    
    def __init__(self, knowledge_base_path: str, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAI RAG system.
        
        Args:
            knowledge_base_path: Path to the knowledge base JSON file
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: OpenAI model to use
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please provide it as an argument or set the OPENAI_API_KEY environment variable.")
        
        self.model = model
        super().__init__(knowledge_base_path)
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the OpenAI API with the prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert financial analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 100
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            print(f"Error calling OpenAI API: {response.status_code} {response.text}")
            return f"<answer>Error: {response.status_code}</answer>"
        
        return response.json()["choices"][0]["message"]["content"]
    
    def generate_answer(self, question: str, context: str, 
                        history: List[Tuple[str, str]] = None) -> str:
        """
        Generate an answer using OpenAI's API.
        
        Args:
            question: The user's question
            context: The retrieved context
            history: Previous conversation history
            
        Returns:
            The generated answer
        """
        # Prepare conversation history string
        history_str = ""
        if history:
            for i, (q, a) in enumerate(history):
                history_str += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"
        
        # Create prompt for the LLM
        prompt = f"""
Answer the following question based ONLY on the provided context.
Be concise and to the point. For numerical answers, provide just the number or percentage without explanation.

CONTEXT:
{context}

QUESTION:
{question}

PREVIOUS CONVERSATION:
{history_str}

Answer with just the final result. Format your answer like this:
<answer>Your concise answer here</answer>
"""
        
        # Call the OpenAI API
        return self._call_llm_api(prompt)

def openai_query_rag(question: str, context: str, history: List[Tuple[str, str]] = None) -> str:
    """
    Query function to be used by the evaluation script with OpenAI.
    
    Args:
        question: The user's question
        context: The retrieved context
        history: Previous QA turns [(q1, a1), (q2, a2), ...]
    
    Returns:
        The RAG system's answer
    """
    # Initialize the OpenAI RAG system
    rag = OpenAIRAGSystem("data/sample_knowledge.json")
    
    # Generate answer
    return rag.generate_answer(question, context, history)

def main():
    """
    Example of how to run the evaluation with the OpenAI RAG system.
    """
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
        print("Example: export OPENAI_API_KEY=your_api_key_here")
        return
    
    # Replace the query_rag function in evaluate_rag with our OpenAI version
    import evaluate_rag
    evaluate_rag.query_rag = openai_query_rag
    
    # Load data
    train_data, knowledge_dict = load_data(
        "data/sample_train.json",
        "data/sample_knowledge.json"
    )
    
    # For testing, let's just evaluate the first example
    example = train_data[0]
    
    # Run evaluation
    result = evaluate_dialogue(example, knowledge_dict, verbose=True)
    
    # Print result
    print(f"\nExample ID: {result['example_id']}")
    print(f"Accuracy: {result['accuracy']:.2f} ({result['correct_count']}/{result['total_turns']})")
    
    # Calculate metrics
    metrics = calculate_metrics([result])
    
    # Print metrics
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 