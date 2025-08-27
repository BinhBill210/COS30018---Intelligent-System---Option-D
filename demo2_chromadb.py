# demo2_chromadb.py - Updated to use LangChain with ChromaDB
from langchain_agent_chromadb import create_business_agent_chromadb

def main():
    """Demo using LangChain format with ChromaDB backend"""
    
    print("🚀 ChromaDB Demo - LangChain Tools")
    print("=" * 50)
    
    try:
        # Create the LangChain agent with ChromaDB (same pattern as demo2_langchain.py)
        agent_executor = create_business_agent_chromadb()
        
        # Demo queries (same as demo2_langchain.py)
        queries = [
            "What are people saying about service quality?",
            "Analyze sentiment of reviews for business XQfwVwDr-v0ZS3_CbbE5Xw",
            "Give me a summary of review statistics for business ID XQfwVwDr-v0ZS3_CbbE5Xw",
            "Search for reviews about food quality and taste"
        ]
        
        for i, query in enumerate(queries):
            print(f"\n=== Query {i+1}: {query} ===")
            try:
                # Use LangChain invoke method (same as demo2_langchain.py)
                response = agent_executor.invoke({
                    "input": query,
                    "chat_history": ""
                })
                print(f"Response: {response['output']}")
            except Exception as e:
                print(f"Error: {e}")
                
        print(f"\n{'='*50}")
        print("✅ ChromaDB Demo completed!")
        print("💡 Same LangChain interface as demo2_langchain.py")
        print("💡 But now powered by ChromaDB vector search instead of custom tools")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
