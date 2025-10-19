"""
Simple LangSmith Evaluation Script for G1 Agent - WITH ACTUAL AGENT
====================================================================

This version shows how to integrate your actual G1 agent with the evaluation.

This script demonstrates:
1. Loading your actual LangChain agent
2. Tracking tool calls with @traceable
3. Evaluating real agent responses

Setup Requirements:
- Set LANGSMITH_API_KEY in your environment
- Set LANGCHAIN_TRACING_V2=true to enable LangSmith tracing
- Activate conda environment: conda activate ...
- Make sure ChromaDB and other services are running
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
# STEP 1: SETUP LANGSMITH CLIENT AND TRACING
# ============================================================================

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "G1 Agent Evaluation"

# Initialize LangSmith client
client = Client()

DATASET_NAME = "G1 Agent Test Dataset"

print("=" * 60)
print("LangSmith Evaluation - WITH ACTUAL G1 AGENT")
print("=" * 60)


# ============================================================================
# STEP 2: LOAD TEST DATASET (Same as before)
# ============================================================================

def load_test_dataset(json_file_path: str) -> List[Dict[str, Any]]:
    """Load test cases from the golden dataset JSON file."""
    print(f"\n[1] Loading test dataset from: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = data.get("test_cases", [])
    print(f"    ✓ Loaded {len(test_cases)} test cases")
    
    return test_cases


def create_langsmith_dataset(test_cases: List[Dict[str, Any]]) -> str:
    """Create a LangSmith dataset from test cases."""
    print(f"\n[2] Creating LangSmith dataset: {DATASET_NAME}")
    
    # Check if dataset already exists
    try:
        existing_datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
        if existing_datasets:
            dataset = existing_datasets[0]
            print(f"    ⚠ Dataset already exists. Using existing dataset.")
            return DATASET_NAME
    except Exception:
        pass
    
    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Golden test dataset for evaluating G1 agent performance"
    )
    print(f"    ✓ Created dataset with ID: {dataset.id}")
    
    # Add examples to the dataset
    for test_case in test_cases:
        inputs = {"query": test_case["query"]}
        outputs = {
            "expected_tool_chain": test_case.get("expected_tool_chain", []),
            "expected_answer_summary": test_case.get("expected_answer_summary", []),
            "test_id": test_case.get("test_id", ""),
            "category": test_case.get("category", "")
        }
        client.create_example(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
    
    print(f"    ✓ Added {len(test_cases)} examples to dataset")
    return DATASET_NAME


# ============================================================================
# STEP 3: LOAD YOUR ACTUAL G1 AGENT
# ============================================================================

def load_g1_agent():
    """
    Load and initialize your actual G1 LangChain agent.
    
    This function imports and sets up your agent from langchain_agent_chromadb.py
    """
    print("\n[3] Loading G1 Agent...")
    
    try:
        from langchain_agent_chromadb import create_business_agent_chromadb
        from gemini_llm import GeminiConfig
        
        # Configure Gemini
        gemini_config = GeminiConfig(
            temperature=0.1,
            max_output_tokens=2048
        )
        
        # Create the agent with your preferred LLM
        # Options: "local" or "gemini"
        agent = create_business_agent_chromadb(
            model_type="gemini",  # or "local"
            gemini_config=gemini_config,
            local_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_4bit=False,
            max_iterations=30,  # Increased from 15 to 30 for complex queries
            verbose=False
        )
        
        print("    ✓ G1 Agent loaded successfully")
        return agent
        
    except Exception as e:
        print(f"    ❌ Error loading agent: {e}")
        print("    Make sure:")
        print("       - ChromaDB is running (if using ChromaDB)")
        print("       - All API keys are set")
        print("       - Dependencies are installed")
        raise


# ============================================================================
# STEP 4: AGENT WRAPPER WITH TRACING
# ============================================================================

# Track number of API calls for rate limiting
_call_count = 0

def run_g1_agent_with_tracing(agent, query: str) -> Dict[str, Any]:
    """
    Run the G1 agent (removed @traceable to avoid parameter interference).
    
    Args:
        agent: The initialized agent executor
        query: The user query to process
        
    Returns:
        Dictionary containing answer and tool calls
    """
    global _call_count
    
    try:
        # Run the agent exactly as streamlit does - WITH chat_history
        result = agent.invoke({
            "input": query,
            "chat_history": ""  # Required by the agent's prompt template
        })
        
        # Extract the final answer
        answer = result.get("output", "")
        
        # Extract tool calls from intermediate steps
        tool_calls = []
        if "intermediate_steps" in result:
            print(f"    📋 Found {len(result['intermediate_steps'])} intermediate steps")
            for i, step in enumerate(result["intermediate_steps"]):
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
            print(f"No 'intermediate_steps' found in response. Keys: {list(result.keys())}")
        
        print(f"Total tools called: {tool_calls}")
        
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
        print(f" Error running agent on query: {query[:50]}...")
        print(f" Error: {e}")
        
        # Still increment counter and wait even on error
        _call_count += 1
        print(f"    ⏳ Waiting 60 seconds before next test case...")
        time.sleep(60)
        
        return {
            "answer": f"Error: {str(e)}",
            "tool_calls": [],
            "query": query
        }


# ============================================================================
# STEP 5: DEFINE EVALUATORS (Same as before, with improvements)
# ============================================================================

def exact_tool_sequence_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 1: Tool Sequence Match
    
    Checks if the agent used the expected sequence of tools.
    """
    expected_tools = example.outputs.get("expected_tool_chain", [])
    actual_tools = run.outputs.get("tool_calls", []) if run.outputs else []
    
    # Exact match
    is_exact_match = actual_tools == expected_tools
    
    # Partial match: check if all expected tools were used (in any order)
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    has_all_tools = expected_set.issubset(actual_set)
    
    # Calculate score
    if is_exact_match:
        score = 1.0
        comment = "Perfect match: exact sequence"
    elif has_all_tools:
        score = 0.5
        comment = "Partial match: all tools used but wrong order"
    else:
        missing_tools = expected_set - actual_set
        extra_tools = actual_set - expected_set
        score = 0.0
        comment = f"Missing: {missing_tools}, Extra: {extra_tools}"
    
    return {
        "key": "tool_sequence_match",
        "score": score,
        "comment": comment
    }


def answer_presence_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 2: Answer Presence
    
    Simple check that the agent produced a non-empty, non-error answer.
    """
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    
    # Check if answer exists and is not an error
    has_valid_answer = (
        len(actual_answer) > 0 and
        not actual_answer.startswith("Error:")
    )
    
    return {
        "key": "answer_presence",
        "score": 1.0 if has_valid_answer else 0.0,
        "comment": f"Answer length: {len(actual_answer)} chars"
    }


# ============================================================================
# STEP 6: RUN EVALUATION
# ============================================================================

def run_evaluation_with_agent(dataset_name: str, agent):
    """
    Run the evaluation using the actual G1 agent.
    """
    print(f"\n[4] Running evaluation on dataset: {dataset_name}")
    print("    ⏳ This may take several minutes depending on dataset size...")
    
    # Wrapper function for the evaluator
    def agent_wrapper(inputs: dict) -> dict:
        """Wrapper that LangSmith will call for each test case."""
        return run_g1_agent_with_tracing(agent, inputs["query"])
    
    # Run evaluation
    results = client.evaluate(
        agent_wrapper,
        data=dataset_name,
        evaluators=[
            exact_tool_sequence_evaluator,
            answer_presence_evaluator
        ],
        experiment_prefix="G1-Agent-Eval",
        max_concurrency=1  # Run one at a time to avoid overloading
    )
    
    print(f"    ✓ Evaluation complete!")
    return results


# ============================================================================
# STEP 7: DISPLAY RESULTS
# ============================================================================

def display_results(results):
    """Display evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n📊 Summary:")
    print(f"   Project: G1 Agent Evaluation")
    print(f"   Dataset: {DATASET_NAME}")
    
    print("\n📈 Aggregate Metrics:")
    if hasattr(results, 'aggregate_results'):
        for metric_name, metric_value in results.aggregate_results.items():
            print(f"   {metric_name}: {metric_value}")
    
    print("\n🔗 View detailed results:")
    print("   1. Go to https://smith.langchain.com/")
    print("   2. Navigate to project: G1 Agent Evaluation")
    print("   3. View individual test runs and traces")
    
    print("\n✅ Evaluation complete!")
    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete evaluation pipeline."""
    
    # Check environment setup
    print("\n🔍 Checking environment setup...")
    
    required_vars = {
        "LANGSMITH_API_KEY": "LangSmith API key for evaluation",
        "GEMINI_API_KEY": "Gemini API key (if using Gemini LLM)"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.environ.get(var):
            if var == "GEMINI_API_KEY":
                print(f"    ⚠ {var} not set ({description}) - OK if using local LLM")
            else:
                print(f"    ❌ {var} not set ({description})")
                missing_vars.append(var)
        else:
            print(f"    ✓ {var} is set")
    
    if missing_vars:
        print("\n❌ Please set required environment variables and try again.")
        return
    
    # Find dataset file
    dataset_file = "golden_test_dataset_v2.json"
    if not os.path.exists(dataset_file):
        dataset_file = os.path.join("evaluation", "golden_test_dataset_v2.json")
        if not os.path.exists(dataset_file):
            print(f"\n❌ Error: Dataset file not found!")
            return
    
    try:
        # Load dataset and create LangSmith dataset
        test_cases = load_test_dataset(dataset_file)
        dataset_name = create_langsmith_dataset(test_cases)
        
        # Load the actual G1 agent
        agent = load_g1_agent()
        
        # Run evaluation
        results = run_evaluation_with_agent(dataset_name, agent)
        
        # Display results
        display_results(results)
        
        print("\n💡 Next Steps:")
        print("   1. Review results in LangSmith dashboard")
        print("   2. Analyze failed test cases")
        print("   3. Improve agent based on findings")
        print("   4. Re-run evaluation to measure improvements")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

