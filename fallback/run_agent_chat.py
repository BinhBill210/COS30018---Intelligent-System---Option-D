from agent import Agent, Tool
from tools.review_search_tool import ReviewSearchTool
from tools.sentiment_summary_tool import SentimentSummaryTool
from tools.data_summary_tool import DataSummaryTool
from local_llm import LocalLLM


def setup_agent():
    # Initialize tools
    search_tool = Tool(
        name="search_reviews",
        description="Search for relevant reviews based on semantic similarity. Input should be a search query string.",
        func=lambda query, k=5: ReviewSearchTool("./chroma_db")(query, k)
    )
    sentiment_tool = Tool(
        name="analyze_sentiment",
        description="Analyze sentiment of a list of reviews. Input should be a list of review texts.",
        func=lambda reviews: SentimentSummaryTool()(reviews)
    )
    data_tool = Tool(
        name="get_data_summary",
        description="Get summary statistics for reviews. Optionally filter by business_id.",
        func=lambda business_id=None: DataSummaryTool("data/processed/review_cleaned.parquet")(business_id)
    )
    tools = [search_tool, sentiment_tool, data_tool]
    llm = LocalLLM(use_4bit=False)
    agent = Agent(tools=tools, max_iterations=5, verbose=True)
    return agent, llm.generate


def main():
    print("\n=== Autonomous Custom Agent Chat ===")
    print("Type your query and press Enter. Type 'exit' to quit.\n")
    agent, llm_generate_func = setup_agent()
    turn = 1
    while True:
        user_input = input(f"[User {turn}]: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break
        print(f"\n[Conversation History before turn {turn}]:\n" + "\n".join(agent.conversation_history) + "\n")
        response = agent.run(user_input, llm_generate_func)
        print(f"[Agent]: {response}\n")
        print(f"[Conversation History after turn {turn}]:\n" + "\n".join(agent.conversation_history) + "\n")
        turn += 1

if __name__ == "__main__":
    main()
