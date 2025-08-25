# demo2_langchain.py - Updated to use LangChain
from langchain_agent import create_business_agent

def main():
    # Create the LangChain agent
    agent_executor = create_business_agent()
    
    # Demo queries
    queries = [
        "What are people saying about service quality?",
        "Analyze sentiment of reviews for business XQfwVwDr-v0ZS3_CbbE5Xw",
        "Give me a summary of review statistics for business ID XQfwVwDr-v0ZS3_CbbE5Xw"
    ]
    
    for i, query in enumerate(queries):
        print(f"\n=== Query {i+1}: {query} ===")
        try:
            response = agent_executor.invoke({
                "input": query,
                "chat_history": ""
            })
            print(f"Response: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
