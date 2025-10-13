# new_evaluator.py
import json
import re
import time
from typing import List, Dict, Any
from tqdm import tqdm

# Import necessary components from your project
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_agent_chromadb import create_business_agent_chromadb
from evaluation.metrics_computation import evaluate_test_result

def load_test_cases(filepath: str = "evaluation/golden_test_dataset_v2.json") -> List[Dict[str, Any]]:
    """Loads test cases from the JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f" Successfully loaded {len(data['test_cases'])} test cases from '{filepath}'.")
        return data['test_cases']
    except FileNotFoundError:
        print(f" Error: File '{filepath}' not found. Please ensure you created it in Step 1.")
        return []

def extract_tool_calls_from_log(log: str) -> List[str]:
    """Extracts tool calls from LangChain's verbose log."""
    # This pattern looks for the string "[TOOL CALLED] tool_name"
    pattern = r"\[TOOL CALLED\]\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    matches = re.findall(pattern, log)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tools = [x for x in matches if not (x in seen or seen.add(x))]
    return unique_tools

def run_single_test(agent_executor: Any, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a single test case and collects the results."""
    query = test_case['query']
    print(f"\n Running Test Case: {test_case.get('test_id', 'N/A')}...")
    print(f"   Query: {query}")

    start_time = time.time()
    
    # To capture the log, we need a way to redirect stdout
    import io
    from contextlib import redirect_stdout

    log_capture_string = io.StringIO()
    with redirect_stdout(log_capture_string):
        try:
            response = agent_executor.invoke({
                "input": query,
                "chat_history": ""
            })
        except Exception as e:
            response = {"output": f"Agent execution error: {e}"}

    execution_time = time.time() - start_time
    full_log = log_capture_string.getvalue()

    actual_tools = extract_tool_calls_from_log(full_log)
    final_answer = response.get('output', 'No final answer was produced.')

    print(f"   Tools Called: {actual_tools}")
    print(f"   Execution Time: {execution_time:.2f} seconds")

    return {
        "actual_tools": actual_tools,
        "final_answer": final_answer,
        "execution_time": execution_time,
        "full_log": full_log
    }

def main():
    """Main function to run the entire evaluation process."""
    print(" Starting LLM Agent Evaluation Process...")
    
    # 1. Load the golden dataset
    test_cases = load_test_cases()
    if not test_cases:
        return

    # 2. Initialize the Agent
    # We use verbose=True to capture the logs of called tools
    print("\n Initializing agent (using Gemini)...")
    agent_executor = create_business_agent_chromadb(model_type="gemini", verbose=True)
    print(" Agent is ready.")

    # 3. Run and Evaluate Each Test Case
    evaluation_results = []
    for test_case in tqdm(test_cases, desc="Processing Test Cases"):
        agent_result = run_single_test(agent_executor, test_case)
        
        # Compute metrics (to be implemented in Step 3)
        metrics = evaluate_test_result(test_case, agent_result)
        
        # Store all results
        result_entry = {
            **test_case,
            "agent_output": agent_result,
            "evaluation_metrics": metrics
        }
        evaluation_results.append(result_entry)

        time.sleep(60)

    # 4. Save the detailed report
    report_path = "evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"\n Detailed evaluation report saved to: '{report_path}'")

    # 5. Analyze and print summary
    analyze_and_print_summary(evaluation_results)

def analyze_and_print_summary(evaluation_results: List[Dict[str, Any]]):
    """Analyzes the results and prints a summary report."""
    if not evaluation_results:
        print("No results to analyze.")
        return

    print("\n" + "="*80)
    print(" EVALUATION RESULTS SUMMARY")
    print("="*80)

    num_tests = len(evaluation_results)
    
    # Calculate average metrics
    avg_tool_f1 = sum(r['evaluation_metrics']['code_based_metrics']['tool_usage']['f1_score'] for r in evaluation_results) / num_tests
    avg_completeness = sum(r['evaluation_metrics']['code_based_metrics']['answer_completeness'] for r in evaluation_results) / num_tests
    avg_relevance = sum(r['evaluation_metrics']['llm_as_judge_metrics']['relevance']['score'] for r in evaluation_results) / num_tests
    avg_coherence = sum(r['evaluation_metrics']['llm_as_judge_metrics']['coherence']['score'] for r in evaluation_results) / num_tests
    avg_correctness = sum(r['evaluation_metrics']['llm_as_judge_metrics']['correctness']['score'] for r in evaluation_results) / num_tests
    avg_time = sum(r['agent_output']['execution_time'] for r in evaluation_results) / num_tests

    print(f"Total Test Cases Executed: {num_tests}\n")

    print("--- Code-Based Metrics ---")
    print(f" Tool Usage Accuracy (F1-Score): {avg_tool_f1:.2f}")
    print(f" Answer Completeness: {avg_completeness:.2%}")

    print("\n--- LLM-as-a-Judge Metrics (on a scale of 5) ---")
    print(f" Relevance Score: {avg_relevance:.2f} / 5")
    print(f" Coherence Score: {avg_coherence:.2f} / 5")
    print(f" Correctness Score: {avg_correctness:.2f} / 5")

    print("\n--- Performance Metrics ---")
    print(f" Average Execution Time: {avg_time:.2f} seconds/query")

    print("\n--- Analysis by Task Category ---")
    results_by_category = {}
    for r in evaluation_results:
        category = r['category']
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(r['evaluation_metrics']['code_based_metrics']['tool_usage']['f1_score'])
    
    for category, scores in results_by_category.items():
        avg_score = sum(scores) / len(scores)
        print(f"  - {category}: Average Tool Accuracy = {avg_score:.2f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()