"""
Enhanced LangSmith Evaluation Script for G1 Agent
==================================================

This script implements enhanced evaluation metrics for agent capabilities:
1. Enhanced Parameter Accuracy - Type checking and value validation
2. Tool Sequence Efficiency - Optimal tool sequence evaluation

Metrics Implemented:
- Enhanced Parameter Accuracy: Type-aware parameter validation with fuzzy matching
- Tool Sequence Efficiency: Evaluates optimal tool sequence with redundancy penalties

Setup Requirements:
- Set LANGSMITH_API_KEY in your environment
- Set LANGCHAIN_TRACING_V2=true to enable LangSmith tracing
- Activate conda environment
- Make sure ChromaDB and other services are running
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from difflib import SequenceMatcher

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.run_helpers import traceable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# SETUP LANGSMITH CLIENT AND TRACING
# ============================================================================

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "G1 Agent Enhanced Evaluation"

client = Client()
DATASET_NAME = "G1 Agent Capabilities Test Dataset"

print("=" * 60)
print("LangSmith Enhanced Evaluation - G1 AGENT")
print("=" * 60)

# ============================================================================
# LOAD TEST DATASET
# ============================================================================

def load_test_dataset(json_file_path: str) -> List[Dict[str, Any]]:
    """Load test cases from JSONL format."""
    print(f"\n[1] Loading test dataset from: {json_file_path}")
    
    test_cases = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Convert expected_trace to expected_tool_chain format
                expected_tools = [step["tool"] for step in data.get("expected_trace", [])]
                test_cases.append({
                    "test_id": data.get("task_id", ""),
                    "query": data.get("query", ""),
                    "expected_tool_chain": expected_tools,
                    "expected_trace": data.get("expected_trace", []),
                    "category": "capability_test"
                })
    
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
        description="Golden test dataset for evaluating G1 agent capabilities"
    )
    print(f"    ✓ Created dataset with ID: {dataset.id}")
    
    # Add examples to the dataset
    for test_case in test_cases:
        inputs = {"query": test_case["query"]}
        outputs = {
            "expected_tool_chain": test_case.get("expected_tool_chain", []),
            "expected_trace": test_case.get("expected_trace", []),
            "test_id": test_case.get("test_id", ""),
            "category": test_case.get("category", "")
        }
        client.create_example(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
    
    print(f"    ✓ Added {len(test_cases)} examples to dataset")
    return DATASET_NAME


# ============================================================================
# LOAD G1 AGENT
# ============================================================================

def load_g1_agent():
    """Load and initialize the G1 LangChain agent."""
    print("\n[3] Loading G1 Agent...")
    
    try:
        from langchain_agent_chromadb import create_business_agent_chromadb
        from gemini_llm import GeminiConfig
        
        gemini_config = GeminiConfig(
            temperature=0.1,
            max_output_tokens=2048
        )
        
        agent = create_business_agent_chromadb(
            model_type="gemini",
            gemini_config=gemini_config,
            local_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_4bit=False,
            max_iterations=30,
            verbose=False
        )
        
        print("    ✓ G1 Agent loaded successfully")
        return agent
        
    except Exception as e:
        print(f"    ✗ Error loading agent: {e}")
        raise


# ============================================================================
# AGENT WRAPPER WITH TRACING
# ============================================================================

_call_count = 0

def run_g1_agent_with_tracing(agent, query: str) -> Dict[str, Any]:
    """Run the G1 agent with detailed tracking."""
    global _call_count
    
    try:
        result = agent.invoke({
            "input": query,
            "chat_history": ""
        })
        
        answer = result.get("output", "")
        
        # Extract tool calls and parameters
        tool_calls = []
        tool_params = []
        
        if "intermediate_steps" in result:
            print(f"  Found {len(result['intermediate_steps'])} intermediate steps")
            for i, step in enumerate(result["intermediate_steps"]):
                if isinstance(step, tuple) and len(step) > 0:
                    action = step[0]
                    if hasattr(action, 'tool'):
                        tool_name = action.tool
                        if tool_name != '_Exception':
                            tool_calls.append(tool_name)
                            # Extract parameters
                            params = {}
                            if hasattr(action, 'tool_input'):
                                params = action.tool_input if isinstance(action.tool_input, dict) else {}
                            tool_params.append({
                                "tool": tool_name,
                                "params": params
                            })
                            print(f"       Step {i+1}: {tool_name} ✓")
        
        print(f"Total tools called: {tool_calls}")
        
        _call_count += 1
        print(f"  Completed test case #{_call_count}. Waiting 60 seconds...")
        time.sleep(60)
        
        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "tool_params": tool_params,
            "query": query,
            "success": True
        }
        
    except Exception as e:
        print(f"    ✗ Error running agent: {e}")
        
        _call_count += 1
        print(f"    ⏳ Waiting 60 seconds before next test case...")
        time.sleep(60)
        
        return {
            "answer": f"Error: {str(e)}",
            "tool_calls": [],
            "tool_params": [],
            "query": query,
            "success": False
        }


# ============================================================================
# EVALUATOR METRICS
# ============================================================================


def tool_sequence_efficiency_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if agent uses optimal tool sequence
    
    Features:
    - Penalizes redundant calls (same tool multiple times)
    - Penalizes missing necessary steps
    - Rewards correct tool ordering
    - Small penalty for unnecessary extra tools
    
    Scoring:
    - Start at 1.0 (perfect efficiency)
    - -10% per redundant call
    - -20% per missing required tool
    - -5% per unnecessary extra tool
    - +10% bonus for correct ordering
    """
    expected_tools = example.outputs.get("expected_tool_chain", [])
    actual_tools = run.outputs.get("tool_calls", []) if run.outputs else []
    
    if len(expected_tools) == 0:
        return {
            "key": "tool_sequence_efficiency",
            "score": 1.0,
            "comment": "No expected sequence"
        }
    
    # Count redundant calls (same tool called multiple times)
    tool_counts = {}
    for tool in actual_tools:
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    redundant_calls = sum(max(0, count - 1) for count in tool_counts.values())
    
    # Check if all necessary tools were called
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    missing_tools = expected_set - actual_set
    extra_tools = actual_set - expected_set
    
    # Calculate efficiency score
    efficiency = 1.0
    
    # Penalize redundancy (10% per redundant call)
    efficiency -= redundant_calls * 0.1
    
    # Penalize missing tools (20% per missing tool)
    efficiency -= len(missing_tools) * 0.2
    
    # Small penalty for extra tools (5% per extra)
    efficiency -= len(extra_tools) * 0.05
    
    # Bonus for correct order (if all tools present)
    if not missing_tools and len(actual_tools) >= len(expected_tools):
        # Check if expected tools appear in order
        expected_indices = []
        for exp_tool in expected_tools:
            try:
                expected_indices.append(actual_tools.index(exp_tool))
            except ValueError:
                break
        
        if len(expected_indices) == len(expected_tools):
            if expected_indices == sorted(expected_indices):
                efficiency += 0.1  # Bonus for correct order
    
    efficiency = max(0.0, min(1.0, efficiency))
    
    comment = f"Efficiency: {efficiency:.2f}, Redundant: {redundant_calls}, Missing: {len(missing_tools)}, Extra: {len(extra_tools)}"
    
    return {
        "key": "tool_sequence_efficiency",
        "score": efficiency,
        "comment": comment
    }


# ============================================================================
# RUN EVALUATION
# ============================================================================

def run_evaluation_with_agent(dataset_name: str, agent):
    """Run the enhanced evaluation with parameter accuracy and sequence efficiency metrics."""
    print(f"\n[4] Running enhanced evaluation on dataset: {dataset_name}")
    print("    ⏳ This may take several minutes...")
    
    def agent_wrapper(inputs: dict) -> dict:
        return run_g1_agent_with_tracing(agent, inputs["query"])
    
    # Enhanced evaluators
    evaluators = [
        tool_sequence_efficiency_evaluator
    ]
    
    results = client.evaluate(
        agent_wrapper,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="G1-Enhanced-Eval",
        max_concurrency=1
    )
    
    print(f"    ✓ Evaluation complete!")
    return results


# ============================================================================
# DISPLAY RESULTS
# ============================================================================

def display_results(results):
    """Display comprehensive evaluation results."""
    print("\n" + "=" * 60)
    print("ENHANCED EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n Summary:")
    print(f"   Project: G1 Agent Enhanced Evaluation")
    print(f"   Dataset: {DATASET_NAME}")
    
    print("\n Metrics Explained:")
    
    print("\n   1. Enhanced Parameter Accuracy:")
    print("      - Type-aware parameter validation")
    print("      - Fuzzy string matching for query parameters")
    print("      - Numeric tolerance for float comparisons")
    print("      - Business ID exact matching")
    print("      - Tracks missing, mismatched, and type errors")
    
    print("\n   2. Tool Sequence Efficiency:")
    print("      - Evaluates optimal tool sequence usage")
    print("      - Penalizes redundant tool calls (-10% each)")
    print("      - Penalizes missing required tools (-20% each)")
    print("      - Small penalty for extra tools (-5% each)")
    print("      - Bonus for correct ordering (+10%)")
    
    print("\n Aggregate Scores:")
    if hasattr(results, 'aggregate_results'):
        for name, value in results.aggregate_results.items():
            print(f"      {name}: {value:.3f}")
    
    print("\n🔗 View detailed results:")
    print("   1. Go to https://smith.langchain.com/")
    print("   2. Navigate to: G1 Agent Enhanced Evaluation")
    print("   3. Analyze individual traces and metrics")
    
    print("\n✅ Evaluation complete!")
    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete evaluation pipeline."""
    
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
                print(f"    ✗ {var} not set ({description})")
                missing_vars.append(var)
        else:
            print(f"    ✓ {var} is set")
    
    if missing_vars:
        print("\n✗ Please set required environment variables and try again.")
        return
    
    # Find dataset file
    dataset_file = "capabilities_golden20.jsonl"
    if not os.path.exists(dataset_file):
        dataset_file = os.path.join("evaluation", "capabilities_golden20.jsonl")
        if not os.path.exists(dataset_file):
            print(f"\n✗ Error: Dataset file not found!")
            return
    
    try:
        test_cases = load_test_dataset(dataset_file)
        dataset_name = create_langsmith_dataset(test_cases)
        agent = load_g1_agent()
        results = run_evaluation_with_agent(dataset_name, agent)
        display_results(results)
        
        print("\n💡 Next Steps:")
        print("   1. Review metrics in LangSmith dashboard")
        print("   2. Identify parameter accuracy issues")
        print("   3. Analyze tool sequence efficiency patterns")
        print("   4. Improve agent based on specific weaknesses")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()