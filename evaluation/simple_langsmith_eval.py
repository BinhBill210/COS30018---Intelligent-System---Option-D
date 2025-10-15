"""
Simple LangSmith Evaluation Script for G1 Agent
================================================

This script evaluates the G1 LangChain agent using LangSmith.

This version uses your ACTUAL G1 agent implementation from langchain_agent_chromadb.py

Evaluation Metrics:
1. Agent Capabilities: Checks if the tool sequence matches expected tools
2. Agent Behavior: Checks if the final output is valid and non-empty

Setup Requirements:
- Set LANGSMITH_API_KEY in your environment
- Set GEMINI_API_KEY in your environment (if using Gemini LLM)
- Activate conda environment: conda activate langchain-demo
- Make sure ChromaDB and other required services are running

Usage:
    conda activate langchain-demo
    python evaluation/simple_langsmith_eval.py
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
from pathlib import Path

# Add project root to Python path so we can import from parent directory
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.run_helpers import traceable
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# STEP 1: SETUP LANGSMITH CLIENT
# ============================================================================

# Initialize LangSmith client (reads LANGSMITH_API_KEY from environment)
client = Client()

# Project name for organizing evaluation runs
PROJECT_NAME = "G1 Agent Evaluation"
DATASET_NAME = "G1 Agent Test Dataset"

print("=" * 60)
print("LangSmith Evaluation Script for G1 Agent")
print("=" * 60)


# ============================================================================
# STEP 2: LOAD TEST DATASET
# ============================================================================

def load_test_dataset(json_file_path: str) -> List[Dict[str, Any]]:
    """
    Load test cases from the golden dataset JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing test cases
        
    Returns:
        List of test case dictionaries
    """
    print(f"\n[1] Loading test dataset from: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = data.get("test_cases", [])
    print(f"    ✓ Loaded {len(test_cases)} test cases")
    
    return test_cases


def create_langsmith_dataset(test_cases: List[Dict[str, Any]]) -> str:
    """
    Create a LangSmith dataset from test cases.
    
    Each example has:
    - inputs: The query to test
    - outputs: Expected tool chain and answer summary (for reference)
    
    Args:
        test_cases: List of test case dictionaries
        
    Returns:
        Dataset name
    """
    print(f"\n[2] Creating LangSmith dataset: {DATASET_NAME}")
    
    # Check if dataset already exists
    try:
        existing_datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
        if existing_datasets:
            dataset = existing_datasets[0]
            print(f"    ⚠ Dataset already exists. Using existing dataset.")
            print(f"    Dataset ID: {dataset.id}")
            return DATASET_NAME
    except Exception as e:
        print(f"    No existing dataset found. Creating new one...")
    
    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Golden test dataset for evaluating G1 agent performance"
    )
    print(f"    ✓ Created dataset with ID: {dataset.id}")
    
    # Add examples to the dataset
    print(f"    Adding {len(test_cases)} examples to dataset...")
    
    for test_case in test_cases:
        # Input is the query
        inputs = {
            "query": test_case["query"]
        }
        
        # Outputs are the expected results (reference data)
        outputs = {
            "expected_tool_chain": test_case.get("expected_tool_chain", []),
            "expected_answer_summary": test_case.get("expected_answer_summary", []),
            "test_id": test_case.get("test_id", ""),
            "category": test_case.get("category", "")
        }
        
        # Create example in dataset
        client.create_example(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset.id
        )
    
    print(f"    ✓ Added all examples to dataset")
    return DATASET_NAME


# ============================================================================
# STEP 3: DEFINE G1 AGENT (ACTUAL IMPLEMENTATION)
# ============================================================================

# Global variable to store the agent executor (loaded once)
_agent_executor = None
_call_count = 0  # Track number of API calls for rate limiting

def initialize_g1_agent(model_type: str = "gemini"):
    """
    Initialize the G1 agent.
    
    This is called once at the start of evaluation to load the agent.
    The agent is reused for all test cases.
    
    Args:
        model_type: "local" or "gemini"
        
    Returns:
        Configured AgentExecutor
    """
    global _agent_executor
    
    if _agent_executor is not None:
        return _agent_executor
    
    print(f"    Initializing G1 agent with {model_type} LLM...")
    
    try:
        from langchain_agent_chromadb import create_business_agent_chromadb
        from gemini_llm import GeminiConfig
        
        # Configure Gemini (or can use local)
        gemini_config = GeminiConfig(
            temperature=0.1,
            max_output_tokens=2048
        )
        
        # Create the agent
        _agent_executor = create_business_agent_chromadb(
            model_type=model_type,
            gemini_config=gemini_config,
            local_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_4bit=False,
            max_iterations=30,  # Increased from 15 to 30 for complex queries
            verbose=False  # Set to False to reduce noise during evaluation
        )
        
        print(f"    ✓ Agent initialized successfully")
        return _agent_executor
        
    except Exception as e:
        print(f"    ❌ Error initializing agent: {e}")
        raise


def run_g1_agent(query: str) -> Dict[str, Any]:
    """
    Run the G1 agent on a query.
    
    Args:
        query: The user query to process
        
    Returns:
        Dictionary containing:
        - answer: The agent's final answer
        - tool_calls: List of tools that were called
        - query: The original query (for reference)
    """
    global _agent_executor, _call_count
    
    # Make sure agent is initialized
    if _agent_executor is None:
        initialize_g1_agent()
    
    try:
        # Call the agent exactly as streamlit_agent.py does (line 401-404)
        # IMPORTANT: Pass both 'input' AND 'chat_history' as the agent expects both
        response = _agent_executor.invoke({
            "input": query,
            "chat_history": ""  # Empty for single-turn evaluation
        })
        
        # Extract the final answer
        answer = response.get("output", "")
        
        # Extract tool calls from intermediate steps
        tool_calls = []
        if "intermediate_steps" in response:
            print(f"    📋 Found {len(response['intermediate_steps'])} intermediate steps")
            for i, step in enumerate(response["intermediate_steps"]):
                if isinstance(step, tuple) and len(step) > 0:
                    action = step[0]
                    # Extract tool name from AgentAction
                    if hasattr(action, 'tool'):
                        tool_name = action.tool
                        # Filter out _Exception (error handling) and only include real tools
                        if tool_name != '_Exception':
                            tool_calls.append(tool_name)
                            print(f"       Step {i+1}: {tool_name} ✓")
                        else:
                            print(f"       Step {i+1}: {tool_name} (skipped - error handling)")
                    else:
                        print(f"       Step {i+1}: No 'tool' attribute found, type: {type(action)}")
        else:
            print(f"    ⚠ No 'intermediate_steps' found in response. Keys: {list(response.keys())}")
        
        print(f"    🔧 Total tools called: {tool_calls}")
        
        # Increment call count and add delay to respect API rate limits
        _call_count += 1
        print(f"    ⏳ Completed test case #{_call_count}. Waiting 60 seconds to respect API rate limits...")
        time.sleep(60)  # Wait 60 seconds between test cases to respect RPM limits
        
        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "query": query
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"    ❌ Error running agent on query: {query[:50]}...")
        print(f"       Error: {error_msg}")
        
        # Still increment counter and wait even on error
        _call_count += 1
        print(f"    ⏳ Waiting 60 seconds before next test case...")
        time.sleep(60)
        
        # Return error details for debugging
        return {
            "answer": f"Error: {error_msg}",
            "tool_calls": [],
            "query": query,
            "error": error_msg
        }


# ============================================================================
# STEP 4: DEFINE EVALUATORS
# ============================================================================

def exact_tool_sequence_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 1: Agent Capabilities - Tool Sequence Match
    
    Checks if the agent used the exact sequence of tools as expected.
    This measures whether the agent has the capability to use the right
    tools in the right order.
    
    Args:
        run: The actual run data from LangSmith
        example: The example from the dataset (contains expected outputs)
        
    Returns:
        Dictionary with score and feedback
    """
    # Get expected tool chain from the example
    expected_tools = example.outputs.get("expected_tool_chain", [])
    
    # Get actual tool calls from the run
    # Note: In a real implementation with @traceable decorators,
    # you would extract tool calls from run.outputs
    actual_tools = run.outputs.get("tool_calls", []) if run.outputs else []
    
    # Check if sequences match exactly
    is_match = actual_tools == expected_tools
    
    return {
        "key": "tool_sequence_match",
        "score": 1.0 if is_match else 0.0,
        "comment": f"Expected: {expected_tools}, Got: {actual_tools}"
    }


def answer_quality_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 2: Agent Behavior - Answer Quality
    
    Checks if the agent's final answer contains key expected information.
    This is a simple keyword-based check for demonstration.
    
    In a production system, you might use:
    - Semantic similarity (embeddings)
    - LLM-as-a-judge evaluation
    - Exact match for structured outputs
    
    Args:
        run: The actual run data from LangSmith
        example: The example from the dataset (contains expected outputs)
        
    Returns:
        Dictionary with score and feedback
    """
    # Get expected answer summary
    expected_summary = example.outputs.get("expected_answer_summary", [])
    
    # Get actual answer from the run
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    
    # Simple check: For demonstration, we just check if answer is non-empty
    # TODO: Implement more sophisticated answer validation
    # For example:
    # - Check if key terms from expected_summary appear in actual_answer
    # - Use semantic similarity
    # - Use structured comparison for JSON outputs
    
    has_answer = len(actual_answer) > 0
    
    return {
        "key": "answer_quality",
        "score": 1.0 if has_answer else 0.0,
        "comment": f"Answer provided: {has_answer}. Expected summary: {expected_summary[:1] if expected_summary else 'None'}"
    }


# ============================================================================
# STEP 5: RUN EVALUATION
# ============================================================================

def run_evaluation(dataset_name: str, model_type: str = "gemini"):
    """
    Run the evaluation on the G1 agent using the test dataset.
    
    This function:
    1. Initializes the agent
    2. Runs the agent on each test case in the dataset
    3. Applies custom evaluators to each run
    4. Collects and displays results
    
    Args:
        dataset_name: Name of the LangSmith dataset to evaluate
        model_type: "local" or "gemini" (default: "gemini")
    """
    print(f"\n[3] Initializing G1 agent...")
    
    # Initialize the agent once before running evaluation
    try:
        initialize_g1_agent(model_type=model_type)
    except Exception as e:
        print(f"    ❌ Failed to initialize agent: {e}")
        print(f"\n    Make sure:")
        print(f"       - ChromaDB is running (if using ChromaDB)")
        print(f"       - GEMINI_API_KEY is set (if using Gemini)")
        print(f"       - All dependencies are installed")
        return None
    
    print(f"\n[4] Running evaluation on dataset: {dataset_name}")
    print("    ⏳ This may take several minutes depending on dataset size...")
    print(f"    Using {model_type} LLM")
    
    # Wrapper function for LangSmith evaluation  
    def evaluate_agent(inputs: dict) -> dict:
        """Wrapper for evaluation that LangSmith can trace"""
        query = inputs.get("query", "")
        result = run_g1_agent(query)
        return result
    
    # Run evaluation
    # The evaluate function will:
    # - Run evaluate_agent on each example in the dataset
    # - Apply the custom evaluators to each run
    # - Store results in LangSmith
    results = client.evaluate(
        evaluate_agent,
        data=dataset_name,
        evaluators=[
            exact_tool_sequence_evaluator,
            answer_quality_evaluator
        ],
        experiment_prefix="G1-Agent-Eval",
        max_concurrency=1  # Run one at a time to avoid overloading
    )
    
    print(f"    ✓ Evaluation complete!")
    
    return results


# ============================================================================
# STEP 6: DISPLAY RESULTS
# ============================================================================

def display_results(results):
    """
    Display evaluation results in a simple, readable format.
    
    Args:
        results: Results object from client.evaluate()
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Note: The actual structure of results may vary
    # This is a simple example display
    
    print("\n📊 Summary:")
    print(f"   Project: {PROJECT_NAME}")
    print(f"   Dataset: {DATASET_NAME}")
    
    # Display aggregate results
    print("\n📈 Aggregate Metrics:")
    if hasattr(results, 'aggregate_results'):
        for metric_name, metric_value in results.aggregate_results.items():
            print(f"   {metric_name}: {metric_value}")
    else:
        print("   Results available in LangSmith dashboard")
    
    # Instructions for viewing detailed results
    print("\n🔗 View detailed results:")
    print("   1. Go to https://smith.langchain.com/")
    print(f"   2. Navigate to project: {PROJECT_NAME}")
    print("   3. View individual test runs and traces")
    
    print("\n✅ Evaluation complete!")
    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the complete evaluation pipeline.
    """
    # Configuration: Choose which LLM to use
    # Options: "local" or "gemini"
    MODEL_TYPE = "gemini"  # Change to "local" if you want to use local LLM
    
    # Check environment variables
    print("\n🔍 Checking environment setup...")
    
    if MODEL_TYPE == "gemini":
        # Check for Gemini API key
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            # Try loading from .env or system keyring
            try:
                from config.api_keys import APIKeyManager
                api_manager = APIKeyManager()
                gemini_key = api_manager.get_api_key('gemini')
            except Exception:
                pass
        
        if not gemini_key:
            print("    ⚠ GEMINI_API_KEY not found!")
            print("    Falling back to local LLM...")
            MODEL_TYPE = "local"
        else:
            print(f"    ✓ GEMINI_API_KEY is set")
    
    if MODEL_TYPE == "local":
        print(f"    ℹ Using local LLM (Qwen2.5)")
    
    # Path to the test dataset
    dataset_file = "golden_test_dataset_v2.json"
    
    # Check if file exists
    if not os.path.exists(dataset_file):
        # Try in evaluation folder
        dataset_file = os.path.join("evaluation", "golden_test_dataset_v2.json")
        if not os.path.exists(dataset_file):
            print(f"\n❌ Error: Dataset file not found!")
            print(f"   Looking for: golden_test_dataset_v2.json")
            return
    
    # Step 1 & 2: Load dataset and create LangSmith dataset
    test_cases = load_test_dataset(dataset_file)
    dataset_name = create_langsmith_dataset(test_cases)
    
    # Step 3, 4, 5: Run evaluation with agent and evaluators
    results = run_evaluation(dataset_name, model_type=MODEL_TYPE)
    
    if results is None:
        print("\n❌ Evaluation failed. Please check the errors above.")
        return
    
    # Step 6: Display results
    display_results(results)
    
    print("\n💡 Next Steps:")
    print("   1. Review results in LangSmith dashboard")
    print("   2. Analyze failed test cases")
    print("   3. Improve agent based on findings")
    print("   4. Add custom evaluators for more specific metrics")
    print("   5. Re-run evaluation to measure improvements")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("❌ Error: LANGSMITH_API_KEY environment variable not set!")
        print("   Please set it before running this script:")
        print("   export LANGSMITH_API_KEY='your-api-key-here'  # Linux/Mac")
        print("   $env:LANGSMITH_API_KEY='your-api-key-here'    # Windows PowerShell")
    else:
        main()

