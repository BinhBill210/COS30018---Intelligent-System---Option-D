"""
Retrieval Quality Evaluators
Đánh giá chất lượng retrieval từ ChromaDB/Vector DB
"""

from typing import List, Dict, Any, Set
from langsmith.schemas import Run, Example
import numpy as np


def calculate_precision_at_k(retrieved_ids: List[str], 
                             relevant_ids: Set[str], 
                             k: int) -> float:
    """
    Precision@K = (# relevant documents in top K) / K
    """
    if k == 0 or not retrieved_ids:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = len([doc_id for doc_id in top_k if doc_id in relevant_ids])
    
    return relevant_in_top_k / k


def calculate_recall_at_k(retrieved_ids: List[str], 
                          relevant_ids: Set[str], 
                          k: int) -> float:
    """
    Recall@K = (# relevant documents in top K) / (total # relevant documents)
    """
    if not relevant_ids:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = len([doc_id for doc_id in top_k if doc_id in relevant_ids])
    
    return relevant_in_top_k / len(relevant_ids)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Mean Reciprocal Rank = 1 / (position of first relevant document)
    
    Example: 
        If first relevant doc is at position 3, MRR = 1/3 = 0.333
    """
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], 
                        relevant_ids: Set[str], 
                        k: int) -> float:
    """
    Normalized Discounted Cumulative Gain@K
    
    Measures ranking quality with position discount
    """
    if k == 0 or not retrieved_ids:
        return 0.0
    
    # DCG: sum of (relevance / log2(position + 1))
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        relevance = 1.0 if doc_id in relevant_ids else 0.0
        dcg += relevance / np.log2(i + 1)
    
    # IDCG: DCG of ideal ranking (all relevant docs first)
    num_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant + 1))
    
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# LangSmith Evaluators
# ============================================================================

def retrieval_precision_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: Retrieval Precision@5
    
    Measures what fraction of retrieved documents are relevant.
    """
    # Extract retrieved document IDs from tool calls
    tool_params = run.outputs.get("tool_params", []) if run.outputs else []
    retrieved_ids = []
    
    for tool_call in tool_params:
        if tool_call.get("tool") in ["search_reviews", "hybrid_retrieve"]:
            # Assume tool returned documents with IDs in result
            # This depends on your tool implementation
            result = tool_call.get("result", {})
            if isinstance(result, dict) and "document_ids" in result:
                retrieved_ids = result["document_ids"]
                break
    
    # Get relevant document IDs from example citations
    citations = example.outputs.get("citations_gt", [])
    relevant_ids = set()
    for citation in citations:
        # citation format: "business_id::source::detail"
        relevant_ids.add(citation.split("::")[0])
    
    k = 5
    precision = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
    
    return {
        "key": "retrieval_precision_at_5",
        "score": precision,
        "comment": f"Precision@{k}: {precision:.3f}, Retrieved: {len(retrieved_ids[:k])}, Relevant: {len(relevant_ids)}"
    }


def retrieval_recall_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: Retrieval Recall@10
    
    Measures what fraction of relevant documents are retrieved.
    """
    tool_params = run.outputs.get("tool_params", []) if run.outputs else []
    retrieved_ids = []
    
    for tool_call in tool_params:
        if tool_call.get("tool") in ["search_reviews", "hybrid_retrieve"]:
            result = tool_call.get("result", {})
            if isinstance(result, dict) and "document_ids" in result:
                retrieved_ids = result["document_ids"]
                break
    
    citations = example.outputs.get("citations_gt", [])
    relevant_ids = set([c.split("::")[0] for c in citations])
    
    k = 10
    recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
    
    return {
        "key": "retrieval_recall_at_10",
        "score": recall,
        "comment": f"Recall@{k}: {recall:.3f}, Found: {len([rid for rid in retrieved_ids[:k] if rid in relevant_ids])}/{len(relevant_ids)}"
    }


def retrieval_mrr_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: Mean Reciprocal Rank (MRR)
    
    Measures position of first relevant document.
    """
    tool_params = run.outputs.get("tool_params", []) if run.outputs else []
    retrieved_ids = []
    
    for tool_call in tool_params:
        if tool_call.get("tool") in ["search_reviews", "hybrid_retrieve"]:
            result = tool_call.get("result", {})
            if isinstance(result, dict) and "document_ids" in result:
                retrieved_ids = result["document_ids"]
                break
    
    citations = example.outputs.get("citations_gt", [])
    relevant_ids = set([c.split("::")[0] for c in citations])
    
    mrr = calculate_mrr(retrieved_ids, relevant_ids)
    
    # Find position of first relevant
    first_pos = next((i+1 for i, rid in enumerate(retrieved_ids) if rid in relevant_ids), -1)
    
    return {
        "key": "retrieval_mrr",
        "score": mrr,
        "comment": f"MRR: {mrr:.3f}, First relevant at position: {first_pos if first_pos > 0 else 'Not found'}"
    }


def retrieval_ndcg_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: NDCG@10
    
    Measures ranking quality with position discount.
    """
    tool_params = run.outputs.get("tool_params", []) if run.outputs else []
    retrieved_ids = []
    
    for tool_call in tool_params:
        if tool_call.get("tool") in ["search_reviews", "hybrid_retrieve"]:
            result = tool_call.get("result", {})
            if isinstance(result, dict) and "document_ids" in result:
                retrieved_ids = result["document_ids"]
                break
    
    citations = example.outputs.get("citations_gt", [])
    relevant_ids = set([c.split("::")[0] for c in citations])
    
    k = 10
    ndcg = calculate_ndcg_at_k(retrieved_ids, relevant_ids, k)
    
    return {
        "key": "retrieval_ndcg_at_10",
        "score": ndcg,
        "comment": f"NDCG@{k}: {ndcg:.3f}"
    }