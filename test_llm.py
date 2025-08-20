# test_llm.py
from local_llm import LocalLLM

def test_llm():
    llm = LocalLLM()
    
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

User: What are people saying about service quality?
Agent:"""

    response = llm.generate(test_prompt)
    print("LLM Response:")
    print(response)

if __name__ == "__main__":
    test_llm()