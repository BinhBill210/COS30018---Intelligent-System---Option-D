# demo.py (updated)
from agent import Agent, Tool
from tools.review_search_tool import ReviewSearchTool
from tools.sentiment_summary_tool import SentimentSummaryTool
from tools.data_summary_tool import DataSummaryTool
from local_llm import LocalLLM

def setup_agent():
    # Initialize tools
    sentiment_tool = SentimentSummaryTool()
    data_tool = DataSummaryTool("data/processed/review_cleaned.csv")
    search_tool = ReviewSearchTool("./chroma_db")

    # Create tool wrappers
    tools = [
        Tool(
            name="analyze_sentiment",
            description="Analyze sentiment of a list of reviews. Input should be a list of review texts separated by '|'.",
            func=sentiment_tool
        ),
        Tool(
            name="get_data_summary",
            description="Get summary statistics for reviews. Optional input should be business ID",
            func=data_tool
        ),
        Tool(
            name="search_reviews",
            description="Search for relevant reviews based on semantic similarity. Input should be a search query string.",
            func=search_tool
        )
    ]
    
    # Initialize LLM
    llm = LocalLLM(use_4bit=False)
    
    # Create agent - pass only tools to the constructor
    agent = Agent(tools=tools, verbose=True)
    
    return agent, llm.generate

def main():
    agent, llm_generate_func = setup_agent()
    
    # Demo queries
    queries = [
        "What are people saying about service quality?",
        "Search for reviews about food quality and analyze their sentiment",
        "Give me a summary of review statistics for business ID XQfwVwDr-v0ZS3_CbbE5Xw",
        "Find reviews mentioning delivery issues"
    ]
    
    for i, query in enumerate(queries):
        print(f"\n=== Query {i+1}: {query} ===")
        response = agent.run(query, llm_generate_func)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()