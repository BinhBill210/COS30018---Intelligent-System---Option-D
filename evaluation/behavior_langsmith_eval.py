"""
Behavioral Evaluation Script for G1 Agent
==========================================

Metrics Implemented:
1. Enhanced Task Success Rate (TSR) - Context-Aware with Ambiguity Handling

Dataset: behavior_golden_v2.jsonl (with ambiguity handling)

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
import re

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
os.environ["LANGCHAIN_PROJECT"] = "G1 Agent Behavioral Evaluation"

client = Client()
DATASET_NAME = "G1 Agent Behavioral Test Dataset2"

print("=" * 60)
print("LangSmith Behavioral Evaluation - G1 AGENT")
print("=" * 60)

# ============================================================================
# LOAD TEST DATASET
# ============================================================================

def load_test_dataset(json_file_path: str) -> List[Dict[str, Any]]:
    """Load test cases from JSONL format with ambiguity metadata."""
    print(f"\n[1] Loading test dataset from: {json_file_path}")
    
    test_cases = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                test_cases.append({
                    "test_id": data.get("task_id", ""),
                    "query": data.get("user_query", ""),
                    "business_name": data.get("business_name", ""),
                    "business_id": data.get("business_id", ""),
                    "answer_gt": data.get("answer_gt", ""),
                    "citations_gt": data.get("citations_gt", []),
                    "topic": data.get("topic", ""),
                    "difficulty": data.get("difficulty", "easy"),
                    "query_type": data.get("query_type", "specific"),
                    "expected_behavior": data.get("expected_behavior", "answer_directly"),
                    "valid_business_ids": data.get("valid_business_ids", []),
                    "ambiguity_level": data.get("ambiguity_level", "low"),
                    "need_reviews": data.get("need_reviews", False),
                    "need_business_info": data.get("need_business_info", False),
                    "review_text": data.get("review_text", ""),
                    "response_tone": data.get("response_tone", ""),
                    "issues": data.get("issues", []),
                    "keyword": data.get("keyword", "")
                })
    
    print(f"    ✓ Loaded {len(test_cases)} test cases")
    ambiguous_count = sum(1 for tc in test_cases if tc["query_type"] == "ambiguous")
    print(f"     Ambiguous queries: {ambiguous_count}/{len(test_cases)}")
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
        description="Ambiguity-aware test dataset for G1 agent behavioral evaluation"
    )
    print(f"    ✓ Created dataset with ID: {dataset.id}")
    
    # Add examples to the dataset
    for test_case in test_cases:
        inputs = {
            "query": test_case["query"],
            "business_name": test_case["business_name"]
        }
        outputs = test_case
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
            temperature=0.3,
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
# AGENT WRAPPER WITH TRACING AND TIMING
# ============================================================================

_call_count = 0

def run_g1_agent_with_tracing(agent, query: str) -> Dict[str, Any]:
    """Run the G1 agent with detailed tracking including timing."""
    global _call_count
    
    start_time = time.time()
    
    try:
        result = agent.invoke({
            "input": query,
            "chat_history": ""
        })
        
        end_time = time.time()
        latency = end_time - start_time
        
        answer = result.get("output", "")
        
        # Extract tool calls
        tool_calls = []
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) > 0:
                    action = step[0]
                    if hasattr(action, 'tool') and action.tool != '_Exception':
                        tool_calls.append(action.tool)
        
        # Estimate token count
        token_count = len(answer.split()) * 1.3
        
        # Check for clarification patterns
        asked_clarification = any(pattern in answer.lower() for pattern in [
            "which", "which one", "multiple", "several", "many", "did you mean",
            "please specify", "could you clarify", "need more information",
            "there are", "found multiple"
        ])
        
        # Check for business listing
        has_business_list = bool(re.search(r'\d+[\.\)]\s+', answer))  # Numbered list pattern
        
        print(f"    - Tools used: {len(tool_calls)}")
        print(f"    - Latency: {latency:.2f}s")
        print(f"    - Asked clarification: {asked_clarification}")
        
        _call_count += 1
        print(f"    ⏳ Completed test case #{_call_count}. Waiting 60 seconds...")
        time.sleep(60)
        
        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "query": query,
            "success": True,
            "latency": latency,
            "token_count": int(token_count),
            "asked_clarification": asked_clarification,
            "has_business_list": has_business_list
        }
        
    except Exception as e:
        print(f"    ✗ Error running agent: {e}")
        return {
            "answer": "",
            "tool_calls": [],
            "query": query,
            "success": False,
            "error": str(e),
            "latency": 0,
            "token_count": 0,
            "asked_clarification": False,
            "has_business_list": False
        }


# ============================================================================
# EVALUATOR METRIC
# ============================================================================

def task_success_rate_evaluator(run: Run, example: Example) -> dict:
    """
    Enhanced Task Success Rate with better ambiguity handling
    
    For specific queries: Validates correct answer
    For ambiguous queries: 
        - Full credit (1.0) for asking clarification
        - Partial credit (0.4-0.6) for listing valid options
        - Zero credit (0.0) for making assumptions
    
    Scoring adapts based on ambiguity level (high/medium/low)
    """
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    query_type = example.outputs.get("query_type", "specific")
    expected_behavior = example.outputs.get("expected_behavior", "answer_directly")
    valid_business_ids = example.outputs.get("valid_business_ids", [])
    ambiguity_level = example.outputs.get("ambiguity_level", "low")
    
    asked_clarification = run.outputs.get("asked_clarification", False) if run.outputs else False
    has_business_list = run.outputs.get("has_business_list", False) if run.outputs else False
    
    if query_type == "specific":
        # Existing logic for specific queries
        actual_norm = actual_answer.lower().strip()
        answer_gt = example.outputs.get("answer_gt", "")
        gt_norm = answer_gt.lower().strip()
        
        gt_components = re.split(r'[•\n,;]', gt_norm)
        gt_components = [c.strip() for c in gt_components if len(c.strip()) > 3]
        
        if len(gt_components) > 0:
            matches = sum(1 for comp in gt_components if comp in actual_norm)
            match_rate = matches / len(gt_components)
            is_successful = match_rate >= 0.5
        else:
            similarity = SequenceMatcher(None, actual_norm, gt_norm).ratio()
            is_successful = similarity >= 0.3
        
        score = 1.0 if is_successful else 0.0
        comment = f"Specific query, Match rate: {match_rate if 'match_rate' in locals() else 'N/A'}"
        
    else:  # ambiguous query
        # Enhanced scoring based on ambiguity level
        if expected_behavior == "clarify_before_selecting":
            if asked_clarification:
                # Full credit for asking clarification
                score = 1.0
                comment = "Ambiguous query: Asked clarification (full credit)"
            elif has_business_list:
                # Check quality of business list
                listed_valid = sum(1 for bid in valid_business_ids if bid in actual_answer)
                total_valid = len(valid_business_ids)
                
                # Adjust score based on ambiguity level
                if ambiguity_level == "high":
                    # High ambiguity: require listing most options
                    if listed_valid >= max(2, total_valid * 0.7):
                        score = 0.6  # Good listing but no explicit clarification
                        comment = f"High ambiguity: Listed {listed_valid}/{total_valid} options (good coverage)"
                    elif listed_valid >= 2:
                        score = 0.4
                        comment = f"High ambiguity: Listed {listed_valid}/{total_valid} options (partial coverage)"
                    else:
                        score = 0.2
                        comment = "High ambiguity: Insufficient options listed"
                else:
                    # Medium/low ambiguity: more lenient
                    if listed_valid >= 2:
                        score = 0.5
                        comment = f"Listed {listed_valid} valid options"
                    else:
                        score = 0.2
                        comment = "Listed options but incomplete"
            else:
                # Penalize making assumptions without clarification
                score = 0.0
                comment = "Ambiguous query: Made assumption without clarifying (penalty)"
        else:
            # Direct answer expected
            score = 1.0 if len(actual_answer) > 10 else 0.0
            comment = "Direct answer expected"
    
    return {
        "key": "task_success_rate",
        "score": score,
        "comment": comment
    }


# ============================================================================
# RUN EVALUATION
# ============================================================================

def run_evaluation_with_agent(dataset_name: str, agent):
    """Run the ambiguity-aware behavioral evaluation."""
    print(f"\n[4] Running ambiguity-aware evaluation on dataset: {dataset_name}")
    print("    ⏳ This may take several minutes...")
    
    def agent_wrapper(inputs: dict) -> dict:
        return run_g1_agent_with_tracing(agent, inputs["query"])
    
    # Enhanced Task Success Rate evaluator
    evaluators = [
        task_success_rate_evaluator
    ]
    
    results = client.evaluate(
        agent_wrapper,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="G1-Behavioral-Eval",
        max_concurrency=1
    )
    
    print(f"    ✓ Evaluation complete!")
    return results


# ============================================================================
# DISPLAY RESULTS
# ============================================================================

def display_results(results):
    """Display comprehensive behavioral evaluation results."""
    print("\n" + "=" * 60)
    print("BEHAVIORAL EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n Summary:")
    print(f"   Project: G1 Agent Behavioral Evaluation")
    print(f"   Dataset: {DATASET_NAME}")
    
    print("\n Metric Explained:")
    
    print("\n    Enhanced Task Success Rate (TSR):")
    print("      - Specific queries: Correct answer required")
    print("      - Ambiguous queries:")
    print("        • Full credit (1.0): Asks for clarification")
    print("        • Partial credit (0.4-0.6): Lists valid options")
    print("        • Zero credit (0.0): Makes assumptions")
    print("      - Scoring adapts based on ambiguity level")
    
    print("\n Aggregate Scores:")
    if hasattr(results, 'aggregate_results'):
        for name, value in results.aggregate_results.items():
            print(f"      {name}: {value:.3f}")
    
    print("\n🔗 View detailed results:")
    print("   1. Go to https://smith.langchain.com/")
    print("   2. Navigate to: G1 Agent Behavioral Evaluation")
    print("   3. Analyze individual test cases and metrics")
    
    print("\n Evaluation complete!")
    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete evaluation pipeline."""
    
    print("\n Checking environment setup...")
    
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
    dataset_file = "behavior_golden_v2.jsonl"
    if not os.path.exists(dataset_file):
        dataset_file = os.path.join("evaluation", "behavior_golden_v2.jsonl")
        if not os.path.exists(dataset_file):
            print(f"\n✗ Error: Dataset file not found!")
            return
    
    try:
        test_cases = load_test_dataset(dataset_file)
        dataset_name = create_langsmith_dataset(test_cases)
        agent = load_g1_agent()
        results = run_evaluation_with_agent(dataset_name, agent)
        display_results(results)
        
    except Exception as e:
        print(f"\n✗ Evaluation failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()