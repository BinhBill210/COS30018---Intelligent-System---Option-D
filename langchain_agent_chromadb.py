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
    from tools.review_search_tool import ReviewSearchTool
    from tools.sentiment_summary_tool import SentimentSummaryTool
    from tools.data_summary_tool import DataSummaryTool
    from tools.business_search_tool import BusinessSearchTool

    # Initialize tools
    search_tool = ReviewSearchTool("./chroma_db")  # ChromaDB search
    sentiment_tool = SentimentSummaryTool()
    data_tool = DataSummaryTool("data/processed/review_cleaned.parquet")
    business_tool = BusinessSearchTool("data/processed/business_cleaned.csv", "./business_chroma_db")

    # Convert to LangChain tools
    langchain_tools = [
        LangChainTool(
            name="search_reviews",
            description="Search for relevant reviews based on semantic similarity. Input can be a string (query) and optional 'k'.",
            func=lambda input: (
                print(f"[TOOL CALLED] search_reviews with input: {input}") or
                (search_tool(input, k=5) if isinstance(input, str)
                else search_tool(input.get("query", ""), k=input.get("k", 5)))
            )
        ),
        LangChainTool(
            name="analyze_sentiment",
            description="Analyze sentiment of a list of reviews. Input should be a list of review texts separated by '|'.",
            func=lambda reviews_input: (
                print(f"[TOOL CALLED] analyze_sentiment with input: {reviews_input}") or
                sentiment_tool(
                    reviews_input.split('|') if isinstance(reviews_input, str) and '|' in reviews_input else [reviews_input]
                )
            )
        ),
        LangChainTool(
            name="get_data_summary",
            description="Get summary statistics for reviews. Optionally filter by business_id.",
            func=lambda business_id=None: (
                print(f"[TOOL CALLED] get_data_summary with input: {business_id}") or
                data_tool(business_id if business_id and business_id.strip() else None)
            )
        ),
        LangChainTool(
            name="get_business_id",
            description="Get the business_id for a given business name (exact match). Input should be a string (business name).",
            func=lambda name: (
                print(f"[TOOL CALLED] get_business_id with input: {name}") or
                business_tool.get_business_id(name)
            )
        ),
        LangChainTool(
            name="search_businesses",
            description="Semantic search for businesses. Input should be a string (query/description) or a dict with 'query' and optional 'k'.",
            func=lambda input: (
                print(f"[TOOL CALLED] search_businesses with input: {input}") or
                (business_tool.search_businesses(input, k=5) if isinstance(input, str)
                else business_tool.search_businesses(input.get("query", ""), k=input.get("k", 5)))
            )
        ),
        LangChainTool(
            name="get_business_info",
            description="Get general info for a business_id. Input should be a string (business_id).",
            func=lambda input: (
                print(f"[TOOL CALLED] get_business_info with input: {input}") or
                business_tool.get_business_info(
                    input if isinstance(input, str) else input.get("business_id", "")
                )
            )
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
You are a business review analysis agent with access to a vector database of reviews and business information.
You can search for relevant reviews, analyze sentiment, provide data summaries, and answer questions about businesses.

TOOLS:
------
You have access to the following tools:

{tools}

STRICT TOOL INPUT FORMATS:
You must use the exact input format for each tool below. Do not invent or guess formats. If you are unsure, do not use the tool.

- search_reviews: Input must be a string (query) or a dict with 'query' (string) and optional 'k' (int). Example: "Find reviews about pizza" or {{{{"query": "pizza", "k": 5}}}}
- analyze_sentiment: Input must be a string of review texts separated by '|'. Example: "Great food|Bad service|Nice ambiance"
- get_data_summary: Input must be a string (business_id) or None. 
- get_business_id: Input must be a string (business name). 
- search_businesses: Input must be a string (query/description) or a dict with 'query' (string) and optional 'k' (int). Example: "vegan restaurant" or {{{{"query": "vegan restaurant", "k": 5}}}}
- get_business_info: Input must be a string (business_id). 

You must never use Action Input with extra quotes, double braces, or incorrect JSON. Only use the formats above.

REASONING AND OUTPUT:
You must only reason based on the actual output (Observation) from the tools. Do not hallucinate, invent, or assume information that is not present in the tool output. If the tool output is empty, say so. If the tool output is unclear, do not guess.

To use a tool, use the following format for each tool you use:

```
Thought: [Your reasoning about which tool(s) to use and why]
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (strictly follow the required format above)
```

If you need to use multiple tools, repeat the Thought/Action/Action Input block for each tool, and update your Thought after each Observation.

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: [Your reasoning about why no further tools are needed]
Final Answer: [your response here]
```

Available capabilities:
- 🔍 search_reviews: Find relevant reviews using semantic similarity (powered by ChromaDB)
- 😊 analyze_sentiment: Analyze sentiment patterns in review texts
- 📊 get_data_summary: Get statistical summaries of review data
- 🏢 get_business_id: Get the business_id for a given business name
- 🏢 search_businesses: Semantic search for businesses by description or name
- 🏢 get_business_info: Get general info for a business_id

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
