# demo.py (updated)
from agent import Agent, Tool
from tools2 import SentimentSummaryTool, DataSummaryTool
from local_llm import LocalLLM

def setup_agent():
    # Initialize tools
    sentiment_tool = SentimentSummaryTool()
    data_tool = DataSummaryTool("data/processed/review_cleaned.csv")
    
    # Create tool wrappers
    tools = [
        Tool(
            name="analyze_sentiment",
            description="Analyze sentiment of a list of reviews. Input: {'reviews': ['review1', 'review2']}",
            func=sentiment_tool
        ),
        Tool(
            name="get_data_summary",
            description="Get summary statistics for reviews. Input: {'business_id': 'optional business id'}",
            func=data_tool
        )
    ]
    
    # Initialize LLM
    llm = LocalLLM()
    
    # Create agent - pass only tools to the constructor
    agent = Agent(tools=tools, verbose=True)
    
    return agent, llm.generate

def main():
    agent, llm_generate_func = setup_agent()
    
    # Demo queries
    queries = [
        "What are people saying about service quality?",
        "Analyze sentiment of reviews for business XQfwVwDr-v0ZS3_CbbE5Xw",
        "Give me a summary of review statistics for business ID XQfwVwDr-v0ZS3_CbbE5Xw"
    ]
    
    for i, query in enumerate(queries):
        print(f"\n=== Query {i+1}: {query} ===")
        response = agent.run(query, llm_generate_func)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()