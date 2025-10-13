import json
import re
from typing import List, Dict, Any

# Giả định bạn đã có một lớp để tương tác với LLM (ví dụ: GeminiLLM).
# Nếu chưa, bạn cần import và khởi tạo nó ở đây.
# from gemini_llm import GeminiLLM
# gemini_judge = GeminiLLM()

# --- Phần 1: Các hàm đánh giá dựa trên mã (Code-Based Metrics) ---

def evaluate_tool_usage(expected_tools: List[str], actual_tools: List[str]) -> Dict[str, float]:
    """
    Đánh giá độ chính xác của việc lựa chọn và sử dụng công cụ.
    Sử dụng các chỉ số Precision, Recall, và F1-Score để đo lường.
    """
    if not expected_tools and not actual_tools:
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}

    if not expected_tools or not actual_tools:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    expected_set = set(expected_tools)
    actual_set = set(actual_tools)

    true_positives = len(expected_set.intersection(actual_set))
    
    precision = true_positives / len(actual_set) if actual_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2)
    }

def evaluate_answer_completeness_code_based(expected_summary: List[str], final_answer: str) -> float:
    """
    Đánh giá sự đầy đủ của câu trả lời bằng cách kiểm tra sự hiện diện của các ý chính
    dựa trên từ khóa.
    """
    if not expected_summary:
        return 1.0

    found_elements = 0
    answer_lower = final_answer.lower()

    for element in expected_summary:
        # Tách ý chính thành các từ khóa quan trọng (dài hơn 3 ký tự)
        key_words = [word for word in re.split(r'\s+', element.lower()) if len(word) > 3]
        if not key_words:
            continue
        
        # Chỉ cần một từ khóa xuất hiện là tính ý đó đã được đề cập
        if any(word in answer_lower for word in key_words):
            found_elements += 1
            
    completeness_score = found_elements / len(expected_summary)
    return round(completeness_score, 2)


# --- Phần 2: Hàm đánh giá dùng LLM-as-a-Judge ---

def evaluate_answer_quality_llm_judge(query: str, final_answer: str, expected_summary: List[str]) -> Dict[str, Any]:
 
    prompt_template = f"""
    You are an expert evaluator for a Business Intelligence AI Agent. Your task is to evaluate the agent's response based on a user's query and a set of evaluation criteria.

    **User Query:**
    "{query}"

    **Agent's Final Answer:**
    "{final_answer}"

    **Evaluation Criteria (What a good answer should contain):**
    - {"- ".join(expected_summary)}

    **Instructions:**
    Please evaluate the agent's answer on a scale of 1 to 5 for the following aspects:
    1.  **Relevance:** How relevant is the answer to the user's query? (1 = Not relevant, 5 = Highly relevant)
    2.  **Coherence:** Is the answer well-structured, clear, and easy to understand? (1 = Incoherent, 5 = Very coherent)
    3.  **Correctness & Faithfulness:** Does the answer seem factually correct and faithful to the information it likely analyzed? (1 = Incorrect, 5 = Completely correct)

    Provide a brief justification for each score. Your output MUST be in a valid JSON format like this example:
    {{
      "relevance": {{
        "score": 5,
        "justification": "The answer directly addresses the user's question about dirty tables."
      }},
      "coherence": {{
        "score": 4,
        "justification": "The answer is well-organized with bullet points, but one sentence is a bit long."
      }},
      "correctness": {{
        "score": 5,
        "justification": "The answer provides specific quotes and details that appear to be grounded in real data."
      }}
    }}
    """
    
    try:
        # === BẠN SẼ GỌI LLM CỦA MÌNH TẠI ĐÂY ===
        # response_from_judge = gemini_judge.generate(prompt_template) 
        # return json.loads(response_from_judge)
        
        # Giả lập kết quả trả về để chạy thử
        print("--- [LLM-as-a-Judge] Simulating call to the judge LLM ---")
        mock_response = {
          "relevance": {
            "score": 4,
            "justification": "Mock response: The answer is relevant to the query."
          },
          "coherence": {
            "score": 5,
            "justification": "Mock response: The answer is well-structured."
          },
          "correctness": {
            "score": 4,
            "justification": "Mock response: The information seems correct based on the query."
          }
        }
        return mock_response

    except Exception as e:
        print(f"Error calling LLM-as-a-Judge: {e}")
        return {
          "relevance": {"score": 0, "justification": str(e)},
          "coherence": {"score": 0, "justification": str(e)},
          "correctness": {"score": 0, "justification": str(e)}
        }

# --- Phần 3: Hàm tổng hợp tính toán chỉ số ---

def evaluate_test_result(test_case: Dict[str, Any], agent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hàm chính để tính toán tất cả các chỉ số cho một test case.
    
    Args:
        test_case: Một mục từ golden_dataset_v2.json
        agent_result: Kết quả trả về từ việc chạy agent
        
    Returns:
        Một dictionary chứa tất cả các điểm số đã được tính toán.
    """
    # Lấy các thông tin cần thiết
    expected_tools = test_case.get('expected_tool_chain', [])
    expected_summary = test_case.get('expected_answer_summary', [])
    query = test_case.get('query', '')
    
    actual_tools = agent_result.get('actual_tools', [])
    final_answer = agent_result.get('final_answer', '')
    
    # 1. Tính toán các chỉ số dựa trên mã
    tool_usage_scores = evaluate_tool_usage(expected_tools, actual_tools)
    completeness_score = evaluate_answer_completeness_code_based(expected_summary, final_answer)
    
    # 2. Tính toán các chỉ số chất lượng bằng LLM-as-a-Judge
    quality_scores_llm = evaluate_answer_quality_llm_judge(query, final_answer, expected_summary)
    
    # 3. Tổng hợp tất cả các điểm số
    all_metrics = {
        "code_based_metrics": {
            "tool_usage": tool_usage_scores,
            "answer_completeness": completeness_score
        },
        "llm_as_judge_metrics": quality_scores_llm
    }
    
    return all_metrics

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # Giả lập một test case từ bộ dữ liệu vàng
    mock_test_case = {
      "test_id": "test_001",
      "category": "Evidence Search",
      "query": "Find three recent reviews that mention 'dirty tables' for Vietnamese Food Truck",
      "expected_tool_chain": ["business_fuzzy_search", "hybrid_retrieve"],
      "expected_answer_summary": [
        "Returns specific reviews mentioning dirty tables",
        "Includes review dates",
        "Provides direct quotes as evidence"
      ]
    }
    
    # Giả lập kết quả trả về từ tác nhân
    mock_agent_result = {
      "actual_tools": ["business_fuzzy_search", "hybrid_retrieve"],
      "final_answer": "I found three reviews about 'dirty tables'. The first one from 2023-10-25 says 'the tables were sticky and dirty'. The second..."
    }
    
    # Chạy hàm tính toán chỉ số
    final_scores = evaluate_test_result(mock_test_case, mock_agent_result)
    
    # In kết quả
    print("--- 📊 METRICS COMPUTATION RESULTS 📊 ---")
    print(json.dumps(final_scores, indent=2))

    # Ví dụ trường hợp thất bại
    print("\n--- Example of a failed Tool Usage case ---")
    mock_agent_result_fail = {
      "actual_tools": ["business_pulse"],
      "final_answer": "The business has an average rating of 4.5 stars."
    }
    final_scores_fail = evaluate_test_result(mock_test_case, mock_agent_result_fail)
    print(json.dumps(final_scores_fail, indent=2))