import time
import os
import sys
from gemini_llm import GeminiLLM, GeminiConfig
from local_llm import LocalLLM
from langchain_agent_chromadb import LangChainLocalLLM
from config.api_keys import load_dotenv
import torch

# Load API keys from .env file
load_dotenv()

def measure_local_llm_load_time():
    print("Testing Local LLM loading time...")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping local LLM test")
        return 0
    
    start_time = time.time()
    
    try:
        # Initialize Local LLM
        local_llm = LocalLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", use_4bit=False)
        
        # Convert to LangChain wrapper
        langchain_local = LangChainLocalLLM(local_llm)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Test a quick generation to verify it works
        test_prompt = "Say hello in one word"
        try:
            response = local_llm.generate(test_prompt)
            print(f"Local LLM loaded in {elapsed:.2f} seconds")
            print(f"Test response: {response}")
        except Exception as e:
            print(f"Model loaded but generation failed: {e}")
        
        return elapsed
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Error loading Local LLM after {elapsed:.2f} seconds: {e}")
        return elapsed

def measure_gemini_load_time():
    print("Testing Gemini LLM loading time...")
    start_time = time.time()
    
    try:
        # Create Gemini configuration
        gemini_config = GeminiConfig(temperature=0.1, max_output_tokens=2048)
        
        # Initialize Gemini LLM
        gemini_llm = GeminiLLM(config=gemini_config)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Test a quick generation to ensure it's working
        test_prompt = "Say hello in one word"
        response = gemini_llm._call(test_prompt)
        
        print(f"Gemini LLM loaded in {elapsed:.2f} seconds")
        print(f"Test response: {response}")
        return elapsed
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Error loading Gemini LLM after {elapsed:.2f} seconds: {e}")
        return elapsed

def measure_aspect_tool_load_time():
    print("\nTesting Aspect Analysis Tool loading time (HuggingFace model)...")
    start_time = time.time()
    
    try:
        # Import here to avoid loading unnecessarily if skipped
        from tools.aspect_analysis import AspectABSAToolHF
        
        # Initialize the tool
        aspect_tool = AspectABSAToolHF(
            business_data_path="data/processed/business_cleaned.parquet",
            review_data_path="data/processed/review_cleaned.parquet"
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Aspect Analysis Tool loaded in {elapsed:.2f} seconds")
        return elapsed
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Error loading Aspect Analysis Tool after {elapsed:.2f} seconds: {e}")
        return elapsed

def measure_tools_load_time():
    print("\nTesting tools loading time (excluding ChromaDB)...")
    start_time = time.time()
    
    try:
        # Import the necessary tools
        from tools.sentiment_summary_tool import SentimentSummaryTool
        from tools.data_summary_tool import DataSummaryTool
        
        # Initialize tools without ChromaDB
        sentiment_tool = SentimentSummaryTool()
        data_tool = DataSummaryTool("data/processed/review_cleaned.parquet")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Basic tools loaded in {elapsed:.2f} seconds")
        return elapsed
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Error loading basic tools after {elapsed:.2f} seconds: {e}")
        return elapsed

def measure_agent_creation_time_gemini():
    print("\nTesting agent creation time with Gemini (excluding ChromaDB tools)...")
    start_time = time.time()
    
    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain.prompts import PromptTemplate
        from langchain.tools import Tool as LangChainTool
        
        # Create a simplified Gemini LLM
        gemini_config = GeminiConfig(temperature=0.1, max_output_tokens=2048)
        llm = GeminiLLM(config=gemini_config)
        
        # Create a dummy tool that doesn't use ChromaDB
        tools = [
            LangChainTool(
                name="dummy_tool",
                description="A dummy tool for testing agent creation time",
                func=lambda x: f"Processed: {x}"
            )
        ]
        
        # Simple prompt template
        prompt = PromptTemplate.from_template("""
        You are a test agent.
        
        TOOLS:
        ------
        You have access to the following tools:
        
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        {agent_scratchpad}
        """)
        
        # Create the agent
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Agent creation with Gemini completed in {elapsed:.2f} seconds")
        return elapsed
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Error creating agent with Gemini after {elapsed:.2f} seconds: {e}")
        return elapsed

def measure_agent_creation_time_local():
    print("\nTesting agent creation time with Local LLM (excluding ChromaDB tools)...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping local LLM agent test")
        return 0
    
    start_time = time.time()
    
    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain.prompts import PromptTemplate
        from langchain.tools import Tool as LangChainTool
        
        # Create local LLM first
        local_llm = LocalLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", use_4bit=False)
            
        # Convert to LangChain wrapper
        llm = LangChainLocalLLM(local_llm)
        
        # Create a dummy tool that doesn't use ChromaDB
        tools = [
            LangChainTool(
                name="dummy_tool",
                description="A dummy tool for testing agent creation time",
                func=lambda x: f"Processed: {x}"
            )
        ]
        
        # Simple prompt template
        prompt = PromptTemplate.from_template("""
        You are a test agent.
        
        TOOLS:
        ------
        You have access to the following tools:
        
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        {agent_scratchpad}
        """)
        
        # Create the agent
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Agent creation with Local LLM completed in {elapsed:.2f} seconds")
        return elapsed
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Error creating agent with Local LLM after {elapsed:.2f} seconds: {e}")
        return elapsed

def main():
    print("=== LLM and Agent Loading Time Test ===")
    print("This test measures loading time without ChromaDB\n")
    
    # Measure Local LLM loading time
    local_time = measure_local_llm_load_time()
    
    # Measure Gemini loading time
    gemini_time = measure_gemini_load_time()
    
    # Measure Aspect Analysis Tool loading time (usually slow due to HuggingFace model)
    aspect_time = measure_aspect_tool_load_time()
    
    # Measure basic tools loading time
    tools_time = measure_tools_load_time()
    
    # Measure agent creation time with both models
    agent_gemini_time = measure_agent_creation_time_gemini()
    agent_local_time = measure_agent_creation_time_local()
    
    # Summary
    print("\n=== Summary ===")
    print(f"Local LLM load time: {local_time:.2f} seconds")
    print(f"Gemini LLM load time: {gemini_time:.2f} seconds")
    print(f"Aspect Analysis Tool load time: {aspect_time:.2f} seconds")
    print(f"Basic tools load time: {tools_time:.2f} seconds")
    print(f"Agent creation time (Gemini): {agent_gemini_time:.2f} seconds")
    print(f"Agent creation time (Local): {agent_local_time:.2f} seconds")
    
    # Calculate total times
    gemini_total = gemini_time + aspect_time + tools_time + agent_gemini_time
    local_total = local_time + aspect_time + tools_time + agent_local_time
    
    print(f"\nTotal with Gemini (no ChromaDB): {gemini_total:.2f} seconds")
    print(f"Total with Local LLM (no ChromaDB): {local_total:.2f} seconds")
    
if __name__ == "__main__":
    main()
