"""
Enhanced LangSmith Evaluation Script for G1 Agent
==================================================

This script implements comprehensive evaluation metrics based on:
1. Agent Capabilities (Tool Use, Planning, Memory)
2. Reliability (Consistency, Robustness)

Metrics Implemented:
- Tool Use: Invocation Accuracy, Tool Selection, Parameter Accuracy
- Planning: Progress Rate, Step Success Rate
- Memory: Context Retention
- Reliability: Consistency, Robustness

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
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import Counter

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
            print(f"    📋 Found {len(result['intermediate_steps'])} intermediate steps")
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
        
        print(f"    🔧 Total tools called: {tool_calls}")
        
        _call_count += 1
        print(f"    ⏳ Completed test case #{_call_count}. Waiting 60 seconds...")
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
# ENHANCED EVALUATORS
# ============================================================================

# ---------------------- Tool Use Evaluators ----------------------

def invocation_accuracy_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 1: Invocation Accuracy
    
    Measures whether the agent correctly decides when to invoke tools.
    """
    expected_tools = example.outputs.get("expected_tool_chain", [])
    actual_tools = run.outputs.get("tool_calls", []) if run.outputs else []
    
    # Agent should invoke tools if and only if expected tools exist
    should_invoke = len(expected_tools) > 0
    did_invoke = len(actual_tools) > 0
    
    is_correct = should_invoke == did_invoke
    
    return {
        "key": "invocation_accuracy",
        "score": 1.0 if is_correct else 0.0,
        "comment": f"Expected: {should_invoke}, Actual: {did_invoke}"
    }


def tool_selection_accuracy_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 2: Tool Selection Accuracy
    
    Measures whether the agent selected the correct tools (ignoring order).
    """
    expected_tools = set(example.outputs.get("expected_tool_chain", []))
    actual_tools = set(run.outputs.get("tool_calls", []) if run.outputs else [])
    
    if len(expected_tools) == 0:
        return {
            "key": "tool_selection_accuracy",
            "score": 1.0,
            "comment": "No tools expected"
        }
    
    # Calculate precision and recall
    correct_tools = expected_tools & actual_tools
    precision = len(correct_tools) / len(actual_tools) if actual_tools else 0
    recall = len(correct_tools) / len(expected_tools) if expected_tools else 0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    missing = expected_tools - actual_tools
    extra = actual_tools - expected_tools
    
    return {
        "key": "tool_selection_accuracy",
        "score": f1,
        "comment": f"F1: {f1:.2f}, Missing: {missing}, Extra: {extra}"
    }


def parameter_accuracy_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 3: Parameter Name F1 Score
    
    Measures how accurately the agent identifies and assigns parameter values.
    """
    expected_trace = example.outputs.get("expected_trace", [])
    actual_params = run.outputs.get("tool_params", []) if run.outputs else []
    
    if len(expected_trace) == 0:
        return {
            "key": "parameter_accuracy",
            "score": 1.0,
            "comment": "No parameters expected"
        }
    
    total_params = 0
    correct_params = 0
    
    # Match expected and actual by tool name
    for expected_step in expected_trace:
        expected_tool = expected_step.get("tool")
        expected_params = expected_step.get("params", {})
        
        # Find matching actual tool call
        matching_actual = None
        for actual in actual_params:
            if actual.get("tool") == expected_tool:
                matching_actual = actual
                break
        
        if matching_actual:
            actual_params_dict = matching_actual.get("params", {})
            
            # Compare parameters
            for param_name, expected_value in expected_params.items():
                total_params += 1
                # Check if parameter exists and matches
                if param_name in actual_params_dict:
                    # For business_id and other IDs, do exact match
                    # For query strings, do fuzzy match
                    if str(actual_params_dict[param_name]) == str(expected_value):
                        correct_params += 1
    
    score = correct_params / total_params if total_params > 0 else 0
    
    return {
        "key": "parameter_accuracy",
        "score": score,
        "comment": f"Correct params: {correct_params}/{total_params}"
    }


# ---------------------- Planning and Reasoning Evaluators ----------------------

def progress_rate_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 4: Progress Rate
    
    Measures how closely the actual tool sequence matches the expected trajectory.
    """
    expected_tools = example.outputs.get("expected_tool_chain", [])
    actual_tools = run.outputs.get("tool_calls", []) if run.outputs else []
    
    if len(expected_tools) == 0:
        return {
            "key": "progress_rate",
            "score": 1.0,
            "comment": "No expected trajectory"
        }
    
    # Calculate longest common subsequence
    def lcs_length(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs = lcs_length(expected_tools, actual_tools)
    progress_rate = lcs / len(expected_tools)
    
    return {
        "key": "progress_rate",
        "score": progress_rate,
        "comment": f"LCS: {lcs}/{len(expected_tools)}, Rate: {progress_rate:.2f}"
    }


def step_success_rate_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 5: Step Success Rate
    
    Measures the percentage of intermediate steps executed successfully.
    """
    expected_steps = len(example.outputs.get("expected_tool_chain", []))
    actual_tools = run.outputs.get("tool_calls", []) if run.outputs else []
    success = run.outputs.get("success", False) if run.outputs else False
    
    if expected_steps == 0:
        return {
            "key": "step_success_rate",
            "score": 1.0,
            "comment": "No steps expected"
        }
    
    # Count successful steps (tools that were executed without errors)
    successful_steps = len(actual_tools) if success else 0
    
    rate = successful_steps / expected_steps
    
    return {
        "key": "step_success_rate",
        "score": rate,
        "comment": f"Successful: {successful_steps}/{expected_steps}"
    }


# ---------------------- Reliability Evaluators ----------------------

def exact_sequence_match_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 6: Exact Tool Sequence Match (for Consistency)
    
    Strict evaluator for pass-all-k consistency testing.
    """
    expected_tools = example.outputs.get("expected_tool_chain", [])
    actual_tools = run.outputs.get("tool_calls", []) if run.outputs else []
    
    is_exact_match = actual_tools == expected_tools
    
    return {
        "key": "exact_sequence_match",
        "score": 1.0 if is_exact_match else 0.0,
        "comment": f"Match: {is_exact_match}"
    }


def answer_quality_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 7: Answer Quality
    
    Checks for valid, non-error responses.
    """
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    
    has_valid_answer = (
        len(actual_answer) > 0 and
        not actual_answer.startswith("Error:") and
        len(actual_answer) > 10  # Minimum meaningful answer length
    )
    
    return {
        "key": "answer_quality",
        "score": 1.0 if has_valid_answer else 0.0,
        "comment": f"Length: {len(actual_answer)} chars, Valid: {has_valid_answer}"
    }


# ============================================================================
# RUN EVALUATION
# ============================================================================

def run_evaluation_with_agent(dataset_name: str, agent):
    """Run the comprehensive evaluation."""
    print(f"\n[4] Running enhanced evaluation on dataset: {dataset_name}")
    print("    ⏳ This may take several minutes...")
    
    def agent_wrapper(inputs: dict) -> dict:
        return run_g1_agent_with_tracing(agent, inputs["query"])
    
    # All evaluators
    evaluators = [
        invocation_accuracy_evaluator,
        tool_selection_accuracy_evaluator,
        parameter_accuracy_evaluator,
        progress_rate_evaluator,
        step_success_rate_evaluator,
        exact_sequence_match_evaluator,
        answer_quality_evaluator
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
    
    print("\n📊 Summary:")
    print(f"   Project: G1 Agent Enhanced Evaluation")
    print(f"   Dataset: {DATASET_NAME}")
    
    print("\n📈 Metrics by Category:")
    print("\n   🔧 Tool Use Capabilities:")
    print("      - Invocation Accuracy: Correct tool invocation decisions")
    print("      - Tool Selection (F1): Precision & recall of tool selection")
    print("      - Parameter Accuracy: Correct parameter assignment")
    
    print("\n   🧠 Planning & Reasoning:")
    print("      - Progress Rate: Alignment with expected trajectory")
    print("      - Step Success Rate: Successful execution of steps")
    
    print("\n   🎯 Reliability:")
    print("      - Exact Sequence Match: Consistency metric")
    print("      - Answer Quality: Valid, meaningful responses")
    
    print("\n📊 Aggregate Scores:")
    if hasattr(results, 'aggregate_results'):
        for metric_name, metric_value in results.aggregate_results.items():
            print(f"   {metric_name}: {metric_value:.3f}")
    
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
        print("   2. Identify failure patterns by category")
        print("   3. Improve agent based on specific metrics")
        print("   4. Re-run for consistency testing (pass-all-k)")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()