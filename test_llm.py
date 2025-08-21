# test_qwen.py
from local_llm import LocalLLM
import torch

def test_qwen():
    print("Testing Qwen2-8B model...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    test_prompt = """You are a business review analysis agent. You have access to various tools to help analyze business reviews.

Available tools:
- search_reviews: Search for reviews based on a query. Input: {'query': 'search terms', 'k': number of results}
- analyze_sentiment: Analyze sentiment of a list of reviews. Input: {'reviews': ['review1', 'review2']}
- get_data_summary: Get summary statistics for reviews. Input: {'business_id': 'optional business id'}

When you need to use a tool, respond in the following format:
Thought: [Your reasoning about what to do next]
Action: [Tool Name]
Action Input: [JSON input for the tool]

When you have a final answer, respond with:
Final Answer: [Your final response to the user]

User: What are people saying about service quality of business with id 7ATYjTIgM3jUlt4UM3IypQ?
Agent:"""

    try:
        # Try with 4-bit quantization first
        llm = LocalLLM("Qwen/Qwen2-8B-Instruct", use_4bit=True)
        response = llm.generate(test_prompt)
        print("Response:")
        print(response)
        
    except Exception as e:
        print(f"Error with 4-bit: {e}")
        print("Trying without quantization...")
        
        try:
            # Fallback to without quantization
            llm = LocalLLM("Qwen/Qwen2-8B-Instruct", use_4bit=False)
            response = llm.generate(test_prompt)
            print("Response:")
            print(response)
            
        except Exception as e2:
            print(f"Error without quantization: {e2}")
            print("Model is too large for available memory.")

if __name__ == "__main__":
    test_qwen()