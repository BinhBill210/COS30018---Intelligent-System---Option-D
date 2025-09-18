# langchain_agent_chromadb.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool as LangChainTool
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Literal
from local_llm import LocalLLM
from gemini_llm import GeminiLLM, GeminiConfig
from config.api_keys import APIKeyManager
from config.logging_config import get_performance_logger
import torch
import json
import logging
import time
import os

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


# 1b. Model factory function
def create_llm_instance(
    model_type: Literal["local", "gemini"] = "local",
    gemini_config: Optional[GeminiConfig] = None,
    local_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    use_4bit: bool = False
) -> LLM:
    """Factory function to create LLM instances
    
    Args:
        model_type: Type of model to create ("local", "gemini")
        gemini_config: Configuration for Gemini model
        local_model_name: Local model name
        use_4bit: Whether to use 4-bit quantization for local model
        
    Returns:
        Configured LLM instance
    """
    api_manager = APIKeyManager()
    
    def _create_local_llm() -> LangChainLocalLLM:
        local_llm = LocalLLM(model_name=local_model_name, use_4bit=use_4bit)
        return LangChainLocalLLM(local_llm)
    
    def _create_gemini_llm() -> GeminiLLM:
        gemini_key = api_manager.get_api_key('gemini')
        if not gemini_key:
            raise ValueError("Gemini API key not found. Please configure it using the API key manager.")
        
        config = gemini_config or GeminiConfig()
        return GeminiLLM(api_key=gemini_key, config=config)
    
    if model_type == "local":
        return _create_local_llm()
    
    elif model_type == "gemini":
        return _create_gemini_llm()
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")




# 2. Convert tools to LangChain tools with ChromaDB support
def create_langchain_tools_chromadb():
    """Convert tools to LangChain format using ChromaDB"""
    from tools.review_search_tool import ReviewSearchTool
    from tools.sentiment_summary_tool import SentimentSummaryTool
    from tools.data_summary_tool import DataSummaryTool
    from tools.business_search_tool import BusinessSearchTool
    from tools.aspect_analysis import AspectABSAToolHF
    from tools.ActionPlanner import ActionPlannerTool
    from tools.ReviewResponseTool import ReviewResponseTool

    chroma_host=os.environ.get("CHROMA_HOST", "localhost")
    

    # Initialize tools
    search_tool = ReviewSearchTool(host=chroma_host)  # ChromaDB search
    sentiment_tool = SentimentSummaryTool()
    data_tool = DataSummaryTool("data/processed/review_cleaned.parquet")
    business_tool = BusinessSearchTool(host=chroma_host)
    aspect_tool = AspectABSAToolHF("data/processed/business_cleaned.parquet", "data/processed/review_cleaned.parquet")
    action_planner_tool = ActionPlannerTool()
    review_response_tool = ReviewResponseTool()
    # Convert to LangChain tools
    langchain_tools = [
        LangChainTool(
            name="search_reviews",
            description="Search for relevant reviews based on semantic similarity. Input can be a string (query), or a dict with 'query', optional 'k', and optional 'business_id'. The business_id is to only return the records that have that business_id.",
            func=lambda input: (
                print(f"[TOOL CALLED] search_reviews with input: {input}") or
                (
                    search_tool(input, k=5) if isinstance(input, str)
                    else search_tool(
                        input.get("query", ""),
                        k=input.get("k", 5),
                        business_id=input.get("business_id", None)
                    )
                )
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
            name="business_fuzzy_search",
            description="Fuzzy search for businesses by name. Input can be a string (query) or a dict with 'query' and optional 'top_n'. The input query is used to search the business record with the business name most similar to the input query. Returns a list of similar business records.",
            func=lambda input: (
                print(f"[TOOL CALLED] fuzzy_search with input: {input}") or
                (business_tool.fuzzy_search(input) if isinstance(input, str)
                 else business_tool.fuzzy_search(input.get('query', ''), top_n=input.get('top_n', 5)))
            )
        ),
        LangChainTool(
            name="search_businesses",
            description="Semantic search for businesses. Return a business record. Input should be a string (query/description) or a dict with 'query' and optional 'k'. Input query represent any information about the business",
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
        ),
        LangChainTool(
            name="analyze_aspects",
            description="Analyze aspects for a business_id. Input should plain business_id string, no JSON.",
            func=lambda input: (
                print(f"[TOOL CALLED] analyze_aspects with input: {input}") or
                (lambda bid: aspect_tool.analyze_aspects(aspect_tool.read_data(business_id=bid)))(
                    (json.loads(input).get("business_id")
                    if isinstance(input, str) and input.strip().startswith("{")
                    else str(input).strip())
                )
            )
        ),
        LangChainTool(
            name="create_action_plan",
            description=(
                "Generate an actionable business improvement plan. Input must be a JSON string or dict with optional keys: 'business_id' (string), 'goals' (list of strings), 'constraints' (dict with 'budget' number and 'timeline_weeks' number), 'priority_issues' (list of strings from: 'quality', 'service', 'value', 'customer_experience'). "
                "Example: {\"business_id\": \"ABC123\", \"goals\": [\"improve_customer_satisfaction\"], \"constraints\": {\"budget\": 5000, \"timeline_weeks\": 8}, \"priority_issues\": [\"quality\", \"service\"]}"
            ),
            func=lambda input: (
                print(f"[TOOL CALLED] create_action_plan with input: {input}") or
                action_planner_tool(
                    **(json.loads(input) if isinstance(input, str) and input.strip().startswith('{')
                       else input if isinstance(input, dict)
                       else {})
                )
            )
        ),
        LangChainTool(
            name="generate_review_response",
            description=(
                "Generate personalized responses to customer reviews with appropriate tone and sentiment handling. "
                "Input must be a JSON string or dict with REQUIRED keys: 'business_id' (string), 'review_text' (string - cannot be empty), and optional 'response_tone' (string from: 'professional', 'friendly', 'formal'). "
                "Example: {\"business_id\": \"ABC123\", \"review_text\": \"Great food but slow service\", \"response_tone\": \"professional\"}"
            ),
            func=lambda input: (
                print(f"[TOOL CALLED] generate_review_response with input: {input}") or
                review_response_tool(
                    **(json.loads(input) if isinstance(input, str) and input.strip().startswith('{')
                       else input if isinstance(input, dict)
                       else {})
                )
            )
        )
    ]
    return langchain_tools

# 3. Create the LangChain agent with ChromaDB
def create_business_agent_chromadb(
    model_type: Literal["local", "gemini"] = "local",
    gemini_config: Optional[GeminiConfig] = None,
    local_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    use_4bit: bool = False,
    max_iterations: int = 15,  # Increased from 5 to 15
    verbose: bool = True
):
    """Create a LangChain-based business review analysis agent with ChromaDB
    
    Args:
        model_type: Type of model to use ("local", "gemini")
        gemini_config: Configuration for Gemini model
        local_model_name: Local model name
        use_4bit: Whether to use 4-bit quantization for local model
        max_iterations: Maximum iterations for agent
        verbose: Whether to enable verbose logging
        
    Returns:
        Configured AgentExecutor
    """
    
    # Create LLM instance based on configuration
    llm = create_llm_instance(
        model_type=model_type,
        gemini_config=gemini_config,
        local_model_name=local_model_name,
        use_4bit=use_4bit
    )
    
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
- analyze_aspects: Input must be a string (business_id).
- business_fuzzy_search: Input must be a string (query) or a dict with 'query' (string) and optional 'top_n' (int). 
- create_action_plan: Input must be a JSON string or dict with 'business_id', 'goals' (list), 'constraints' (dict), 'priority_issues' (list). Example: {{{{"business_id": "ABC123", "goals": ["improve_customer_satisfaction"], "constraints": {{"budget": 5000, "timeline_weeks": 8}}, "priority_issues": ["quality", "service"]}}}}
- generate_review_response: Input must be a JSON string or dict with 'business_id', 'review_text', and optional 'response_tone'. Example: {{{{"business_id": "ABC123", "review_text": "Great food but slow service", "response_tone": "professional"}}}}
- create_action_plan: Input must be a JSON string or dict with optional keys: 'business_id' (string), 'goals' (list of strings), 'constraints' (dict with 'budget' number and 'timeline_weeks' number), 'priority_issues' (list of strings from: 'quality', 'service', 'value', 'customer_experience'). Example: {{{{"business_id": "ABC123", "goals": ["improve_customer_satisfaction"], "constraints": {{"budget": 5000, "timeline_weeks": 8}}, "priority_issues": ["quality", "service"]}}}}
- generate_review_response: Input must be a JSON string or dict with REQUIRED keys: 'business_id' (string), 'review_text' (string - cannot be empty), and optional 'response_tone' (string from: 'professional', 'friendly', 'formal'). Example: {{{{"business_id": "ABC123", "review_text": "Great food but slow service", "response_tone": "professional"}}}}


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
- search_reviews: Find relevant reviews using semantic similarity (powered by ChromaDB)
- analyze_sentiment: Analyze sentiment patterns in review texts
- get_data_summary: Get statistical summaries of review data
- get_business_id: Get the business_id for a given business name
- search_businesses: Semantic search for businesses by description or name
- get_business_info: Get general info for a business_id
- analyze_aspects: Analyze aspects of a list of reviews from a business_id
- business_fuzzy_search: Fuzzy search for businesses by name
- create_action_plan: Generate comprehensive business improvement plans with actions, timelines, costs, and KPIs
- generate_review_response: Create personalized, professional responses to customer reviews with sentiment analysis


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
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True
    )
    
    return agent_executor

# 4. Usage example
def main():
    """Example usage of the LangChain agent with ChromaDB"""
    
    print("🚀 LangChain Agent with ChromaDB")
    print("=" * 50)
    
    # Create the agent (default to local model)
    agent_executor = create_business_agent_chromadb(model_type="local")
    
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
