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

from safety_evaluators import (
    privacy_pii_evaluator,
    toxicity_evaluator,
    compliance_refusal_evaluator,
    data_access_compliance_evaluator
)
from retrieval_evaluators import (
    retrieval_precision_evaluator,
    retrieval_recall_evaluator,
    retrieval_mrr_evaluator,
    retrieval_ndcg_evaluator
)
from robustness_evaluators import (
    robustness_typo_evaluator,
    error_recovery_evaluator
)
from llm_judge import get_llm_judge
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
    
def answer_completeness_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: Answer Completeness
    Checks if the agent's answer contains all expected key points from the ground truth.
    (Logic adapted from agent_evaluator.py)
    """
    # Lấy "dàn ý" từ ground truth
    expected_summary = example.outputs.get("expected_answer_summary", [])

    # Lấy câu trả lời thực tế của agent
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""

    if not expected_summary:
        return {"key": "answer_completeness", "score": 1.0, "comment": "No summary points to check."}

    found_points = 0
    missing_points = []
    for point in expected_summary:
        # Kiểm tra xem từng ý chính có trong câu trả lời không
        if point.lower() in actual_answer.lower():
            found_points += 1
        else:
            missing_points.append(point)

    # Tính điểm dựa trên tỷ lệ các ý tìm thấy
    score = found_points / len(expected_summary) if expected_summary else 1.0
    comment = f"Found {found_points}/{len(expected_summary)} expected points. Missing: {missing_points}"

    return {
        "key": "answer_completeness",
        "score": score,
        "comment": comment
    }

def response_relevance_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: Response Relevance
    Checks keyword overlap between the query and the final answer.
    (Logic adapted from agent_evaluator.py)
    """
    query = example.inputs.get("query", "")
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""

    query_words = set(query.lower().split())
    response_words = set(actual_answer.lower().split())

    # Loại bỏ các từ phổ biến để tăng độ chính xác
    common_words = {'the', 'a', 'and', 'or', 'is', 'are', 'in', 'on', 'what', 'give', 'me'}
    query_words -= common_words
    response_words -= common_words

    if not query_words:
        return {"key": "response_relevance", "score": 1.0, "comment": "Query has no specific keywords."}

    overlap = len(query_words.intersection(response_words))
    score = min(overlap / len(query_words), 1.0)

    comment = f"Found {overlap}/{len(query_words)} keyword overlaps."

    return {
        "key": "response_relevance",
        "score": score,
        "comment": comment
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

    # ---------------------- LLM-as-a-Judge Evaluators ----------------------

def llm_judge_relevance_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 8: LLM-as-a-Judge for Answer Relevance
    
    Uses LLM to evaluate how relevant the answer is to the query.
    """
    judge = get_llm_judge()
    
    query = example.inputs.get("query", "")
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    expected_trace = example.outputs.get("expected_trace", [])
    
    # Get expected answer description from trace or use a default
    expected_summary = "Complete the task correctly"
    if expected_trace:
        expected_summary = f"Use tools: {[step['tool'] for step in expected_trace]}"
    
    try:
        result = judge.evaluate_relevance(query, actual_answer, expected_summary)
        return {
            "key": "llm_judge_relevance",
            "score": result["score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "llm_judge_relevance",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }


def llm_judge_helpfulness_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 9: LLM-as-a-Judge for Helpfulness
    
    Evaluates how helpful the answer is to the user.
    """
    judge = get_llm_judge()
    
    query = example.inputs.get("query", "")
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    
    try:
        result = judge.evaluate_helpfulness(query, actual_answer)
        return {
            "key": "llm_judge_helpfulness",
            "score": result["score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "llm_judge_helpfulness",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }


def llm_judge_explanation_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 10: LLM-as-a-Judge for Explanation Quality
    
    Evaluates the quality of reasoning and explanation.
    """
    judge = get_llm_judge()
    
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    
    try:
        result = judge.evaluate_explanation_quality(actual_answer)
        return {
            "key": "llm_judge_explanation",
            "score": result["score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "llm_judge_explanation",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }

# ---------------------- Performance Evaluators ----------------------

def latency_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: Latency Measurement
    
    Measures total execution time.
    """
    # LangSmith tracks execution time automatically
    latency_ms = run.latency if hasattr(run, 'latency') else 0
    
    # Convert to seconds
    latency_sec = latency_ms / 1000.0 if latency_ms else 0.0
    
    # Score: 1.0 if < 30s, linear decay to 0 at 120s
    if latency_sec <= 30:
        score = 1.0
    elif latency_sec >= 120:
        score = 0.0
    else:
        score = 1.0 - (latency_sec - 30) / 90
    
    return {
        "key": "latency",
        "score": score,
        "comment": f"Latency: {latency_sec:.2f}s"
    }


def token_usage_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: Token Usage Efficiency
    
    Measures token efficiency (lower is better).
    """
    # Try to extract token usage from run metadata
    # This depends on LangSmith's tracking
    total_tokens = 0
    
    if hasattr(run, 'metadata') and run.metadata:
        total_tokens = run.metadata.get('total_tokens', 0)
    
    # Rough cost estimation (Gemini pricing as example)
    # Input: $0.075 / 1M tokens, Output: $0.30 / 1M tokens
    # Assuming 50/50 split for simplicity
    estimated_cost = (total_tokens / 1_000_000) * 0.1875
    
    # Score: 1.0 if < 1000 tokens, linear decay to 0 at 10000 tokens
    if total_tokens <= 1000:
        score = 1.0
    elif total_tokens >= 10000:
        score = 0.0
    else:
        score = 1.0 - (total_tokens - 1000) / 9000
    
    return {
        "key": "token_efficiency",
        "score": score,
        "comment": f"Tokens: {total_tokens}, Est. cost: ${estimated_cost:.4f}"
    }


def cost_efficiency_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: Cost Efficiency
    
    Measures cost per successful task.
    """
    success = run.outputs.get("success", False) if run.outputs else False
    total_tokens = 0
    
    if hasattr(run, 'metadata') and run.metadata:
        total_tokens = run.metadata.get('total_tokens', 0)
    
    estimated_cost = (total_tokens / 1_000_000) * 0.1875
    
    # Score based on cost-effectiveness
    if not success:
        score = 0.0
        comment = "Task failed, cost wasted"
    elif estimated_cost <= 0.01:  # Less than 1 cent
        score = 1.0
        comment = f"Excellent: ${estimated_cost:.4f}"
    elif estimated_cost <= 0.05:  # Less than 5 cents
        score = 0.7
        comment = f"Good: ${estimated_cost:.4f}"
    else:
        score = 0.3
        comment = f"Expensive: ${estimated_cost:.4f}"
    
    return {
        "key": "cost_efficiency",
        "score": score,
        "comment": comment
    }
# ============================================================================
# RUN EVALUATION
# ============================================================================

def run_evaluation_with_agent(dataset_name: str, agent):
    """Run the comprehensive evaluation."""
    print(f"\n[4] Running enhanced evaluation on dataset: {dataset_name}")
    print("This may take several minutes...")
    
    def agent_wrapper(inputs: dict) -> dict:
        return run_g1_agent_with_tracing(agent, inputs["query"])
    
    # All evaluators - UPDATED LIST
    evaluators = [
        # Tool Use
        invocation_accuracy_evaluator,
        tool_selection_accuracy_evaluator,
        parameter_accuracy_evaluator,
        
        # Planning & Reasoning
        progress_rate_evaluator,
        step_success_rate_evaluator,
        
        # Reliability
        exact_sequence_match_evaluator,
        answer_quality_evaluator,
        
        # LLM-as-a-Judge (NEW!)
        llm_judge_relevance_evaluator,
        llm_judge_helpfulness_evaluator,
        llm_judge_explanation_evaluator,

        # Retrieval Quality (NEW!)
        retrieval_precision_evaluator,
        retrieval_recall_evaluator,
        retrieval_mrr_evaluator,
        retrieval_ndcg_evaluator,

            # Safety & Compliance (NEW!)
        privacy_pii_evaluator,
        toxicity_evaluator,
        compliance_refusal_evaluator,
        data_access_compliance_evaluator,

        # Robustness (NEW!)
        robustness_typo_evaluator,
        error_recovery_evaluator,
    
        # Performance (NEW!)
        latency_evaluator,
        token_usage_evaluator,
        cost_efficiency_evaluator,
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
    """Display comprehensive evaluation results with category breakdown."""
    print("\n" + "=" * 60)
    print("ENHANCED EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n📊 Summary:")
    print(f"   Project: G1 Agent Enhanced Evaluation")
    print(f"   Dataset: {DATASET_NAME}")
    
    # Display results by category
    print("\n📈 Results by Category:")
    
    for category_name, subcategories in EVALUATION_TAXONOMY.items():
        print(f"\n{'='*60}")
        print(f"📁 {category_name.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        
        for subcategory_name, evaluators in subcategories.items():
            print(f"\n  📊 {subcategory_name.replace('_', ' ').title()}:")
            
            for evaluator in evaluators:
                evaluator_key = evaluator.__name__.replace('_evaluator', '')
                print(f"     - {evaluator_key}")
    
    # Aggregate scores
    print("\n" + "="*60)
    print("📊 Aggregate Scores:")
    print("="*60)
    
    if hasattr(results, 'results'):
        # Calculate category averages
        category_scores = {}
        
        for category_name, subcategories in EVALUATION_TAXONOMY.items():
            scores = []
            for subcategory_name, evaluators in subcategories.items():
                for evaluator in evaluators:
                    key = evaluator.__name__.replace('_evaluator', '')
                    # Try to find score in results
                    # This is simplified - actual implementation depends on results structure
                    if hasattr(results, 'aggregate_results'):
                        score = results.aggregate_results.get(key, None)
                        if score is not None:
                            scores.append(score)
            
            if scores:
                category_scores[category_name] = sum(scores) / len(scores)
        
        # Display category scores
        for category, avg_score in category_scores.items():
            print(f"   {category.replace('_', ' ').title()}: {avg_score:.3f}")
    
    print("\n🔗 View detailed results:")
    print("   1. Go to https://smith.langchain.com/")
    print("   2. Navigate to: G1 Agent Enhanced Evaluation")
    print("   3. Analyze individual traces and metrics by category")
    
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
        
        # Ask user which evaluation to run
        print("\n" + "="*60)
        print("EVALUATION OPTIONS")
        print("="*60)
        print("1. Run FULL evaluation (all metrics)")
        print("2. Run by category:")
        print("   - agent_capabilities")
        print("   - agent_behavior")
        print("   - reliability")
        print("   - safety")
        print("3. Run original evaluation (backward compatible)")
        
        choice = input("\nEnter choice (1-3, or category name): ").strip()
        
        if choice == "1":
            results = run_full_evaluation_organized(dataset_name, agent)
        elif choice == "2" or choice in EVALUATION_TAXONOMY:
            if choice == "2":
                category = input("Enter category name: ").strip()
            else:
                category = choice
            
            if category in EVALUATION_TAXONOMY:
                results = run_evaluation_by_category(dataset_name, agent, category)
            else:
                print(f"Invalid category. Choose from: {list(EVALUATION_TAXONOMY.keys())}")
                return
        elif choice == "3":
            results = run_evaluation_with_agent(dataset_name, agent)
        else:
            print("Invalid choice. Running full evaluation by default.")
            results = run_full_evaluation_organized(dataset_name, agent)
        
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

# ============================================================================
# TAXONOMY-BASED EVALUATION
# ============================================================================

EVALUATION_TAXONOMY = {
    "agent_capabilities": {
        "tool_use": [
            invocation_accuracy_evaluator,
            tool_selection_accuracy_evaluator,
            parameter_accuracy_evaluator,
        ],
        "planning_reasoning": [
            progress_rate_evaluator,
            step_success_rate_evaluator,
        ],
        "retrieval": [
            retrieval_precision_evaluator,
            retrieval_recall_evaluator,
            retrieval_mrr_evaluator,
            retrieval_ndcg_evaluator,
        ]
    },
    "agent_behavior": {
        "task_completion": [
            exact_sequence_match_evaluator,
            answer_quality_evaluator,
        ],
        "output_quality": [
            llm_judge_relevance_evaluator,
            llm_judge_helpfulness_evaluator,
            llm_judge_explanation_evaluator,
        ],
        "performance": [
            latency_evaluator,
            token_usage_evaluator,
            cost_efficiency_evaluator,
        ]
    },
    "reliability": {
        "consistency": [
            exact_sequence_match_evaluator,
        ],
        "robustness": [
            robustness_typo_evaluator,
            error_recovery_evaluator,
        ]
    },
    "safety": {
        "privacy": [
            privacy_pii_evaluator,
        ],
        "harm": [
            toxicity_evaluator,
        ],
        "compliance": [
            compliance_refusal_evaluator,
            data_access_compliance_evaluator,
        ]
    }
}


def run_evaluation_by_category(dataset_name: str, agent, category: str):
    """
    Run evaluation for a specific category only.
    
    Args:
        dataset_name: LangSmith dataset name
        agent: Initialized agent
        category: One of ["agent_capabilities", "agent_behavior", "reliability", "safety"]
    """
    print(f"\n[4] Running {category} evaluation on dataset: {dataset_name}")
    print("    ⏳ This may take several minutes...")
    
    def agent_wrapper(inputs: dict) -> dict:
        return run_g1_agent_with_tracing(agent, inputs["query"])
    
    # Get evaluators for this category
    evaluators = []
    for subcategory in EVALUATION_TAXONOMY[category].values():
        evaluators.extend(subcategory)
    
    print(f"    Using {len(evaluators)} evaluators for category: {category}")
    
    results = client.evaluate(
        agent_wrapper,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"G1-{category.title()}-Eval",
        max_concurrency=1
    )
    
    print(f"    ✓ Evaluation complete!")
    return results


def run_full_evaluation_organized(dataset_name: str, agent):
    """
    Run complete evaluation organized by taxonomy.
    
    This runs ALL evaluators but organizes results by category.
    """
    print(f"\n[4] Running FULL evaluation (organized by taxonomy)")
    print(f"    Dataset: {dataset_name}")
    print("    ⏳ This will take significant time...")
    
    def agent_wrapper(inputs: dict) -> dict:
        return run_g1_agent_with_tracing(agent, inputs["query"])
    
    # Collect ALL evaluators from taxonomy
    all_evaluators = []
    for category_name, subcategories in EVALUATION_TAXONOMY.items():
        for subcategory_name, evaluators in subcategories.items():
            all_evaluators.extend(evaluators)
    
    # Remove duplicates (some evaluators appear in multiple categories)
    seen = set()
    unique_evaluators = []
    for evaluator in all_evaluators:
        if evaluator.__name__ not in seen:
            seen.add(evaluator.__name__)
            unique_evaluators.append(evaluator)
    
    print(f"    Total unique evaluators: {len(unique_evaluators)}")
    
    results = client.evaluate(
        agent_wrapper,
        data=dataset_name,
        evaluators=unique_evaluators,
        experiment_prefix="G1-Full-Eval",
        max_concurrency=1
    )
    
    print(f"    ✓ Full evaluation complete!")
    return results

if __name__ == "__main__":
    main()