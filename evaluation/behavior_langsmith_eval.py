"""
Behavioral Evaluation Script for G1 Agent
==========================================


Metrics Implemented:
1. Task Success Rate (TSR) - Context-Aware
2. Factual Correctness (FC) - Context-Sensitive
3. Instruction Adherence (IA) - Expanded with Clarification
4. Response Quality (RQ) - Human-Judged Behavior
5. Efficiency Metrics
6. Behavior Score (Composite)
7. Clarification & Disambiguation Metrics

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
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"    ✗ Error running agent: {e}")
        
        _call_count += 1
        print(f"     Waiting 60 seconds before next test case...")
        time.sleep(60)
        
        return {
            "answer": f"Error: {str(e)}",
            "tool_calls": [],
            "query": query,
            "success": False,
            "latency": latency,
            "token_count": 0,
            "asked_clarification": False,
            "has_business_list": False
        }


# ============================================================================
# AMBIGUITY-AWARE EVALUATORS
# ============================================================================

# ---------------------- 1. Task Success Rate (TSR) - Redefined ----------------------

def task_success_rate_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 1: Task Success Rate (TSR) - Context-Aware
    
    
    For specific queries: success = correct answer
    For ambiguous queries: success = asks clarification OR lists valid options
    """
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    query_type = example.outputs.get("query_type", "specific")
    expected_behavior = example.outputs.get("expected_behavior", "answer_directly")
    valid_business_ids = example.outputs.get("valid_business_ids", [])
    business_id = example.outputs.get("business_id", "")
    answer_gt = example.outputs.get("answer_gt", "")
    
    asked_clarification = run.outputs.get("asked_clarification", False) if run.outputs else False
    has_business_list = run.outputs.get("has_business_list", False) if run.outputs else False
    
    if query_type == "specific":
        # For specific queries: check if answer contains expected content
        actual_norm = actual_answer.lower().strip()
        gt_norm = answer_gt.lower().strip()
        
        # Check key information presence
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
        # For ambiguous queries: reward clarification behavior
        if expected_behavior == "clarify_before_selecting":
            if asked_clarification:
                # Full credit for asking clarification
                score = 1.0
                comment = "Ambiguous query: Asked clarification (full credit)"
            elif has_business_list:
                # Check if valid businesses are listed
                listed_valid = sum(1 for bid in valid_business_ids if bid in actual_answer)
                if listed_valid >= 2:
                    score = 0.5
                    comment = f"Ambiguous query: Listed {listed_valid} valid options (partial credit)"
                else:
                    score = 0.2
                    comment = "Ambiguous query: Listed options but incomplete"
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


# ---------------------- 2. Factual Correctness (FC) - Context-Sensitive ----------------------

def extract_business_ids(text: str) -> List[str]:
    """Extract potential business IDs from text."""
    # Pattern for IDs like "vPK_MN51evy0007W8NeJ6w"
    pattern = r'[A-Za-z0-9_-]{22}'
    return re.findall(pattern, text)


def factual_correctness_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 2: Factual Correctness (FC) - Context-Sensitive

    
    Validates:
    - Listed businesses exist in valid_business_ids
    - Facts remain neutral until selection for ambiguous queries
    - No hallucinated details
    """
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    query_type = example.outputs.get("query_type", "specific")
    valid_business_ids = example.outputs.get("valid_business_ids", [])
    business_id = example.outputs.get("business_id", "")
    answer_gt = example.outputs.get("answer_gt", "")
    
    if not actual_answer or actual_answer.startswith("Error:"):
        return {
            "key": "factual_correctness",
            "score": 0.0,
            "comment": "No valid answer provided"
        }
    
    if query_type == "ambiguous":
        # For ambiguous queries: check neutrality and valid business mentions
        mentioned_ids = extract_business_ids(actual_answer)
        
        if len(mentioned_ids) > 0:
            # Check if mentioned IDs are valid
            valid_mentions = [bid for bid in mentioned_ids if bid in valid_business_ids]
            invalid_mentions = [bid for bid in mentioned_ids if bid not in valid_business_ids]
            
            if len(invalid_mentions) > 0:
                score = 0.0
                comment = f"Mentioned {len(invalid_mentions)} invalid business IDs (fabricated)"
            elif len(valid_mentions) >= 2:
                score = 1.0
                comment = f"Listed {len(valid_mentions)} valid businesses (factual & neutral)"
            else:
                score = 0.5
                comment = "Mentioned businesses but incomplete list"
        else:
            # No specific IDs mentioned, check for neutral language
            assumption_patterns = [
                r'the.*is\s+\d+',  # "the rating is 3.5" (assumes specific business)
                r'located at',  # assumes specific address
                r'their.*hours'  # assumes specific business
            ]
            made_assumption = any(re.search(pat, actual_answer.lower()) for pat in assumption_patterns)
            
            if made_assumption:
                score = 0.5
                comment = "Made assumptions about specific business (partially factual)"
            else:
                score = 0.8
                comment = "Neutral response without assumptions"
    
    else:  # specific query
        # For specific queries: validate against ground truth
        from difflib import SequenceMatcher
        
        # Extract facts
        actual_facts = extract_key_facts(actual_answer)
        gt_facts = extract_key_facts(answer_gt)
        
        if len(gt_facts) == 0:
            similarity = SequenceMatcher(None, 
                                        actual_answer.lower(), 
                                        answer_gt.lower()).ratio()
            score = similarity
            comment = f"Text similarity: {similarity:.2f}"
        else:
            matches = 0
            for gt_fact in gt_facts:
                for actual_fact in actual_facts:
                    sim = SequenceMatcher(None, gt_fact.lower(), actual_fact.lower()).ratio()
                    if sim >= 0.8:
                        matches += 1
                        break
            
            score = matches / len(gt_facts) if gt_facts else 0.0
            comment = f"Fact match: {matches}/{len(gt_facts)} facts correct"
    
    return {
        "key": "factual_correctness",
        "score": score,
        "comment": comment
    }


def extract_key_facts(text: str) -> List[str]:
    """Extract key factual elements from text."""
    facts = []
    
    # Extract times
    time_pattern = r'\d{1,2}:\d{2}[-–]\d{1,2}:\d{2}'
    facts.extend(re.findall(time_pattern, text))
    
    # Extract addresses
    address_pattern = r'\d+\s+[A-Za-z\s]+(?:Ave|St|Rd|Blvd|Dr|Cir|Ct)'
    facts.extend(re.findall(address_pattern, text, re.IGNORECASE))
    
    # Extract ratings
    rating_pattern = r'\b[1-5]\.[0-9]\b'
    facts.extend(re.findall(rating_pattern, text))
    
    # Extract numbers
    number_pattern = r'\b\d+\s*reviews?\b'
    facts.extend(re.findall(number_pattern, text, re.IGNORECASE))
    
    return [f.strip() for f in facts]


# ---------------------- 3. Instruction Adherence (IA) - Expanded ----------------------

def instruction_adherence_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 3: Instruction Adherence (IA) - Expanded
    
    
    Rewards:
    - Asking clarifying questions when ambiguous
    - Following system persona (concise, polite)
    
    Penalizes:
    - Guessing without confirmation
    """
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    query_type = example.outputs.get("query_type", "specific")
    expected_behavior = example.outputs.get("expected_behavior", "answer_directly")
    asked_clarification = run.outputs.get("asked_clarification", False) if run.outputs else False
    response_tone = example.outputs.get("response_tone", "")
    keyword = example.outputs.get("keyword", "")
    topic = example.outputs.get("topic", "")
    
    if not actual_answer or actual_answer.startswith("Error:"):
        return {
            "key": "instruction_adherence",
            "score": 0.0,
            "comment": "No valid response to evaluate"
        }
    
    score = 1.0
    violations = []
    
    # Check ambiguity handling
    if query_type == "ambiguous" and expected_behavior == "clarify_before_selecting":
        if asked_clarification:
            score += 0.2  # Bonus for proper behavior
            violations.append("+clarification_asked")
        elif run.outputs.get("has_business_list", False):
            # Partial credit for listing options
            pass
        else:
            score -= 0.4
            violations.append("guessed_without_clarification")
    
    # Check tone requirements
    if response_tone == "professional":
        casual_patterns = [r'\blol\b', r'\bhaha\b', r'!!!', r'\byeah\b']
        if any(re.search(pat, actual_answer.lower()) for pat in casual_patterns):
            score -= 0.3
            violations.append("unprofessional_tone")
    
    # Check keyword mention for keyword_evidence tasks
    if topic == "keyword_evidence" and keyword:
        if keyword.lower() not in actual_answer.lower():
            score -= 0.4
            violations.append(f"missing_keyword_{keyword}")
    
    # Check for evidence when requested
    query = example.inputs.get("query", "")
    if "evidence" in query.lower() or "example" in query.lower():
        has_evidence = ('"' in actual_answer or 
                       "for example" in actual_answer.lower() or
                       len(actual_answer) > 100)
        if not has_evidence:
            score -= 0.3
            violations.append("no_supporting_evidence")
    
    score = max(0.0, min(score, 1.0))  # Clamp between 0 and 1
    
    comment = f"Score: {score:.2f}"
    if violations:
        comment += f", Actions: {', '.join(violations)}"
    else:
        comment += ", All instructions followed"
    
    return {
        "key": "instruction_adherence",
        "score": score,
        "comment": comment
    }


# ---------------------- 4. Response Quality (RQ) - Human-Judged Behavior ----------------------

def response_quality_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 4: Response Quality (RQ) - Human-Judged Behavior
    
    
    Measures:
    - Clarity and helpfulness
    - Awareness of uncertainty (for ambiguous queries)
    - Well-structured communication
    """
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    query_type = example.outputs.get("query_type", "specific")
    topic = example.outputs.get("topic", "")
    difficulty = example.outputs.get("difficulty", "easy")
    
    if not actual_answer or actual_answer.startswith("Error:"):
        return {
            "key": "response_quality",
            "score": 0.0,
            "comment": "No valid response"
        }
    
    score = 0.0
    quality_notes = []
    
    # 1. Clarity: Check sentence structure
    sentences = re.split(r'[.!?]+', actual_answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) >= 1:
        score += 0.25
    else:
        quality_notes.append("incomplete_sentences")
    
    # 2. Completeness: Check length appropriateness
    word_count = len(actual_answer.split())
    
    if difficulty == "easy" and word_count >= 5:
        score += 0.25
    elif difficulty == "medium" and word_count >= 20:
        score += 0.25
    elif difficulty == "hard" and word_count >= 50:
        score += 0.25
    else:
        quality_notes.append("too_brief")
    
    # 3. Context awareness
    business_name = example.inputs.get("business_name", "")
    if business_name.lower() in actual_answer.lower():
        score += 0.2
    
    # 4. Uncertainty awareness (for ambiguous queries)
    if query_type == "ambiguous":
        uncertainty_indicators = [
            "which", "multiple", "several", "need to know", "could you",
            "please specify", "various", "different locations"
        ]
        shows_uncertainty = any(ind in actual_answer.lower() for ind in uncertainty_indicators)
        
        if shows_uncertainty:
            score += 0.3
            quality_notes.append("+uncertainty_awareness")
        else:
            quality_notes.append("no_uncertainty_awareness")
    else:
        # Bonus for confident, clear answers on specific queries
        score += 0.3
    
    # 5. Professional tone
    informal_patterns = [r'\blol\b', r'\bhaha\b', r'\byeah\b', r'\bnope\b']
    has_informal = any(re.search(pat, actual_answer.lower()) for pat in informal_patterns)
    if not has_informal:
        score += 0.2
    else:
        quality_notes.append("informal_tone")
    
    score = min(score, 1.0)
    
    comment = f"Score: {score:.2f}, Words: {word_count}"
    if quality_notes:
        comment += f", Notes: {', '.join(quality_notes)}"
    
    return {
        "key": "response_quality",
        "score": score,
        "comment": comment
    }


# ---------------------- 5. Efficiency Metrics ----------------------

def efficiency_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 5: Efficiency Metrics
    
    
    Measures latency and token cost with difficulty-based thresholds
    """
    latency = run.outputs.get("latency", 0) if run.outputs else 0
    token_count = run.outputs.get("token_count", 0) if run.outputs else 0
    difficulty = example.outputs.get("difficulty", "easy")
    
    latency_thresholds = {"easy": 10.0, "medium": 20.0, "hard": 30.0}
    token_thresholds = {"easy": 100, "medium": 300, "hard": 500}
    
    max_latency = latency_thresholds.get(difficulty, 20.0)
    max_tokens = token_thresholds.get(difficulty, 300)
    
    latency_score = max(0, 1 - (latency / max_latency))
    token_score = max(0, 1 - (token_count / max_tokens))
    
    efficiency_score = (latency_score * 0.6 + token_score * 0.4)
    
    return {
        "key": "efficiency",
        "score": efficiency_score,
        "comment": f"Latency: {latency:.2f}s/{max_latency}s, Tokens: {token_count}/{max_tokens}"
    }


# ---------------------- 6. Clarification Initiation Rate ----------------------

def clarification_initiation_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 6: Clarification Initiation Rate
    
    From Mohammadi "Behavior under Uncertain Intent"
    
    Measures % of ambiguous queries where agent asks for clarification
    """
    query_type = example.outputs.get("query_type", "specific")
    
    if query_type != "ambiguous":
        return {
            "key": "clarification_initiation",
            "score": 1.0,  # N/A for specific queries
            "comment": "Not applicable (specific query)"
        }
    
    asked_clarification = run.outputs.get("asked_clarification", False) if run.outputs else False
    
    return {
        "key": "clarification_initiation",
        "score": 1.0 if asked_clarification else 0.0,
        "comment": f"Clarification: {'Yes' if asked_clarification else 'No'}"
    }


# ---------------------- 7. Disambiguation Accuracy ----------------------

def disambiguation_accuracy_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 7: Disambiguation Accuracy
    
    From Yehudai "Interactive Decision Benchmarks"
    
    Checks if listed options contain the correct business ID
    """
    query_type = example.outputs.get("query_type", "specific")
    
    if query_type != "ambiguous":
        return {
            "key": "disambiguation_accuracy",
            "score": 1.0,
            "comment": "Not applicable (specific query)"
        }
    
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    valid_business_ids = example.outputs.get("valid_business_ids", [])
    correct_business_id = example.outputs.get("business_id", "")
    
    # Check if correct business is in listed options
    if correct_business_id in actual_answer:
        # Check if other valid businesses are also listed
        valid_mentions = sum(1 for bid in valid_business_ids if bid in actual_answer)
        
        if valid_mentions >= 2:
            score = 1.0
            comment = f"Listed {valid_mentions} valid options including correct one"
        else:
            score = 0.5
            comment = "Mentioned correct business but incomplete list"
    else:
        score = 0.0
        comment = "Correct business not in options"
    
    return {
        "key": "disambiguation_accuracy",
        "score": score,
        "comment": comment
    }


# ---------------------- 8. Over-Confidence Penalty ----------------------

def overconfidence_penalty_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 8: Over-Confidence Penalty
    
    Promotes safe behavior by penalizing decisive responses to ambiguous queries
    without clarification
    """
    query_type = example.outputs.get("query_type", "specific")
    
    if query_type != "ambiguous":
        return {
            "key": "overconfidence_penalty",
            "score": 1.0,
            "comment": "Not applicable (specific query)"
        }
    
    actual_answer = run.outputs.get("answer", "") if run.outputs else ""
    asked_clarification = run.outputs.get("asked_clarification", False) if run.outputs else False
    has_business_list = run.outputs.get("has_business_list", False) if run.outputs else False
    
    # Check for overconfident patterns
    confident_patterns = [
        r'the\s+(?:address|hours|rating)\s+(?:is|are)',
        r'located at\s+\d+',
        r'opens at\s+\d+',
        r'(?:is|has)\s+\d+\.\d+\s+stars'
    ]
    
    is_overconfident = any(re.search(pat, actual_answer.lower()) for pat in confident_patterns)
    
    if is_overconfident and not asked_clarification and not has_business_list:
        score = 0.0
        comment = "Over-confident: Made specific claims without clarification (penalty)"
    elif asked_clarification or has_business_list:
        score = 1.0
        comment = "Safe behavior: Asked clarification or listed options"
    else:
        score = 0.7
        comment = "Neutral response"
    
    return {
        "key": "overconfidence_penalty",
        "score": score,
        "comment": comment
    }


# ---------------------- 9. Behavior Score (Composite Metric) ----------------------

def behavior_score_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator 9: Behavior Score (Composite Metric)
    
    Aggregates all metrics with weighted formula:
    BehaviorScore = 0.35*TSR + 0.25*FC + 0.20*RQ + 0.10*IA + 0.10*Efficiency
    
    Suggested by Yehudai §6 "Metric Calibration"
    """
    # Get individual scores
    tsr = task_success_rate_evaluator(run, example)["score"]
    fc = factual_correctness_evaluator(run, example)["score"]
    rq = response_quality_evaluator(run, example)["score"]
    ia = instruction_adherence_evaluator(run, example)["score"]
    eff = efficiency_evaluator(run, example)["score"]
    
    # Weighted combination
    behavior_score = (
        0.35 * tsr +
        0.25 * fc +
        0.20 * rq +
        0.10 * ia +
        0.10 * eff
    )
    
    return {
        "key": "behavior_score",
        "score": behavior_score,
        "comment": f"Composite: TSR={tsr:.2f} FC={fc:.2f} RQ={rq:.2f} IA={ia:.2f} Eff={eff:.2f}"
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
    
    # All evaluators
    evaluators = [
        task_success_rate_evaluator,
        factual_correctness_evaluator,
        instruction_adherence_evaluator,
        response_quality_evaluator,
        efficiency_evaluator,
        clarification_initiation_evaluator,
        disambiguation_accuracy_evaluator,
        overconfidence_penalty_evaluator,
        behavior_score_evaluator
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
    
    print("\n Metrics Explained:")
    
    print("\n    Core Behavioral Metrics:")
    print("      1. Task Success Rate (TSR)")
    print("         - Specific queries: Correct answer required")
    print("         - Ambiguous queries: Clarification = 1.0, List options = 0.5")
    
    print("\n      2. Factual Correctness (FC)")
    print("         - Validates business IDs against valid set")
    print("         - Penalizes fabricated information")
    print("         - Rewards neutral stance for ambiguous queries")
    
    print("\n      3. Instruction Adherence (IA)")
    print("         - Rewards asking clarification for ambiguous queries")
    print("         - Penalizes guessing without confirmation")
    print("         - Checks tone and format compliance")
    
    print("\n      4. Response Quality (RQ)")
    print("         - Clarity, helpfulness, structure")
    print("         - Awareness of uncertainty")
    print("         - Professional communication")
    
    print("\n      5. Efficiency Metrics")
    print("         - Latency and token cost")
    print("         - Difficulty-based thresholds")
    
    print("\n   🔍 Ambiguity-Specific Metrics:")
    print("      6. Clarification Initiation Rate")
    print("         - % of ambiguous queries where agent asks for clarification")
    
    print("\n      7. Disambiguation Accuracy")
    print("         - Correct business in listed options")
    
    print("\n      8. Over-Confidence Penalty")
    print("         - Penalizes decisive answers without clarification")
    
    print("\n      9. Behavior Score (Composite)")
    print("         - Weighted: 35% TSR + 25% FC + 20% RQ + 10% IA + 10% Eff")
    
    print("\n Aggregate Scores:")
    if hasattr(results, 'aggregate_results'):
        # Group by metric type
        core_metrics = {}
        ambiguity_metrics = {}
        
        for metric_name, metric_value in results.aggregate_results.items():
            if 'clarification' in metric_name or 'disambiguation' in metric_name or 'overconfidence' in metric_name:
                ambiguity_metrics[metric_name] = metric_value
            else:
                core_metrics[metric_name] = metric_value
        
        print("\n   Core Metrics:")
        for name, value in core_metrics.items():
            print(f"      {name}: {value:.3f}")
        
        print("\n   Ambiguity Handling:")
        for name, value in ambiguity_metrics.items():
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