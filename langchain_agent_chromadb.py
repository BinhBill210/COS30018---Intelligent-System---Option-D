# langchain_agent_chromadb.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool as LangChainTool
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from local_llm import LocalLLM
import torch

# 1. Create a LangChain-compatible LLM wrapper (same as before)
class LangChainLocalLLM(LLM):
    """Custom LangChain LLM wrapper for LocalLLM"""
    
    local_llm: Any  # Declare the field for Pydantic
    
    def __init__(self, local_llm: LocalLLM, **kwargs):
        super().__init__(local_llm=local_llm, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "local_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        return self.local_llm.generate(prompt, **kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.local_llm.model_name}

# 2. Convert tools to LangChain tools with ChromaDB support
def create_langchain_tools_chromadb():
    """Convert tools to LangChain format using ChromaDB"""
    from tools2 import SentimentSummaryTool, DataSummaryTool
    from tools_chromadb import ReviewSearchTool
    
    # Initialize tools
    search_tool = ReviewSearchTool("./chroma_db")  # ChromaDB search
    sentiment_tool = SentimentSummaryTool()
    data_tool = DataSummaryTool("data/processed/review_cleaned.parquet")
    
    # Convert to LangChain tools
    langchain_tools = [
        LangChainTool(
            name="search_reviews",
            description="Search for relevant reviews based on semantic similarity. Input should be a search query string.",
            func=lambda query: search_tool(query, k=5)
        ),
        LangChainTool(
            name="analyze_sentiment",
            description="Analyze sentiment of a list of reviews. Input should be a list of review texts separated by '|'.",
            func=lambda reviews_input: sentiment_tool(
                reviews_input.split('|') if isinstance(reviews_input, str) and '|' in reviews_input else [reviews_input]
            )
        ),
        LangChainTool(
            name="get_data_summary",
            description="Get summary statistics for reviews. Optionally filter by business_id.",
            func=lambda business_id=None: data_tool(business_id if business_id and business_id.strip() else None)
        )
    ]
    
    return langchain_tools

# 3. Create the LangChain agent with ChromaDB
def create_business_agent_chromadb():
    """Create a LangChain-based business review analysis agent with ChromaDB"""
    
    # Initialize LocalLLM with the original Qwen model
    local_llm = LocalLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", use_4bit=False)
    
    # Wrap it for LangChain
    llm = LangChainLocalLLM(local_llm)
    
    # Get tools with ChromaDB support
    tools = create_langchain_tools_chromadb()
    
    # Create custom prompt template for ReAct pattern
    react_prompt = PromptTemplate.from_template("""
You are a business review analysis agent with access to a vector database of reviews.
You can search for relevant reviews, analyze sentiment, and provide data summaries.

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Available capabilities:
- 🔍 search_reviews: Find relevant reviews using semantic similarity (powered by ChromaDB)
- 😊 analyze_sentiment: Analyze sentiment patterns in review texts
- 📊 get_data_summary: Get statistical summaries of review data

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")
    
    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    return agent_executor

# 4. Usage example
def main():
    """Example usage of the LangChain agent with ChromaDB"""
    
    print("🚀 LangChain Agent with ChromaDB")
    print("=" * 50)
    
    # Create the agent
    agent_executor = create_business_agent_chromadb()
    
    # Example queries
    queries = [
        "What are people saying about service quality?",
        "Search for reviews about food quality and analyze their sentiment",
        "Give me a summary of review statistics for business ID XQfwVwDr-v0ZS3_CbbE5Xw",
        "Find reviews mentioning delivery issues"
    ]
    
    for i, query in enumerate(queries):
        print(f"\n{'='*50}")
        print(f"Query {i+1}: {query}")
        print('='*50)
        
        try:
            response = agent_executor.invoke({
                "input": query,
                "chat_history": ""  # You can maintain chat history here
            })
            print(f"Response: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
