"""
Utility functions for code-based evaluation metrics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _clamp_unit(value: float | None) -> float:
    """Clamp a value into the [0, 1] range."""
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def normalize_text(text: Any) -> str:
    """Lowercase, strip, and collapse whitespace for basic string comparisons."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())


def continuous_from_error(value: float, reference: float) -> float:
    """
    Convert absolute error into a similarity score (1 - relative error).
    When reference is zero, demands exact match.
    """
    if reference == 0:
        return 1.0 if value == 0 else 0.0
    rel_error = abs(value - reference) / (abs(reference) + 1e-8)
    return _clamp_unit(1.0 - rel_error)


def _clean_tool_name(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip()


def tool_use_metrics(expected_chain: Sequence[str], tool_calls: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compare expected tool sequence with actual tool calls and compute granular scores.
    Returns raw floats (not rounded) plus a summary score.
    """
    expected = [_clean_tool_name(t) for t in (expected_chain or []) if _clean_tool_name(t)]
    actual = []
    for call in tool_calls or []:
        if isinstance(call, dict):
            actual.append(_clean_tool_name(call.get("name") or call.get("tool")))
        else:
            actual.append(_clean_tool_name(call))
    actual = [name for name in actual if name]

    intent_acc = 1.0 if expected and actual else 0.0

    if expected:
        matched = sum(1 for tool in expected if tool in actual)
        selection_acc = matched / len(expected)
    else:
        selection_acc = 1.0 if not actual else 0.0

    if len(expected) <= 1:
        order_acc = 1.0
    else:
        total_pairs = len(expected) - 1
        hits = 0
        for idx in range(total_pairs):
            first = expected[idx]
            second = expected[idx + 1]
            first_positions = [i for i, name in enumerate(actual) if name == first]
            success = False
            for pos in first_positions:
                if any(actual[j] == second for j in range(pos + 1, len(actual))):
                    success = True
                    break
            if success:
                hits += 1
        order_acc = hits / total_pairs if total_pairs else 1.0

    param_f1 = 1.0  # Placeholder until parameter GT is provided
    exec_pass = 1.0
    for call in tool_calls or []:
        if isinstance(call, dict) and call.get("ok") is False:
            exec_pass = 0.0
            break

    score = 0.25 * (intent_acc + selection_acc + order_acc + param_f1)

    return {
        "intent_acc": intent_acc,
        "selection_acc": selection_acc,
        "order_acc": order_acc,
        "param_f1": param_f1,
        "exec_pass": exec_pass,
        "score": _clamp_unit(score),
        "score_raw": score,
    }


def facts_exact_or_tolerance(test: Dict[str, Any], final_answer: str) -> float:
    """Exact text match against answer_gt if provided."""
    answer_gt = test.get("answer_gt")
    if not isinstance(answer_gt, str) or not answer_gt.strip():
        return 0.0
    return 1.0 if normalize_text(answer_gt) == normalize_text(final_answer) else 0.0


def citation_f1(model_citations: Iterable[str], gt_citations: Iterable[str]) -> float:
    """Compute F1 overlap between model and ground truth citations."""
    gt = {str(c) for c in gt_citations or [] if str(c)}
    if not gt:
        return 1.0
    model = {str(c) for c in model_citations or [] if str(c)}
    if not model:
        return 0.0
    true_pos = len(gt & model)
    precision = true_pos / len(model) if model else 0.0
    recall = true_pos / len(gt) if gt else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract_reference_text(test: Dict[str, Any]) -> str:
    if isinstance(test.get("answer_gt"), str) and test["answer_gt"].strip():
        return test["answer_gt"]
    summary = test.get("expected_answer_summary")
    if isinstance(summary, list):
        return " ".join(str(item) for item in summary if item)
    if isinstance(summary, str):
        return summary
    return ""


def sentiment_code_score(test: Dict[str, Any], final_answer: str) -> float:
    """
    Heuristic sentiment check used in older judge logic.
    Kept for backward compatibility; returns 1 only for sentiment-focused tasks.
    """
    category = (test.get("category") or "").lower()
    if category not in {"recent_sentiment", "business_pulse"}:
        return 0.0

    reference = _extract_reference_text(test).lower()
    expected_label = None
    for label in ("positive", "negative", "neutral"):
        if label in reference:
            expected_label = label
            break
    if not expected_label:
        return 0.0
    return 1.0 if expected_label in (final_answer or "").lower() else 0.0


def aspect_f1_from_answer(test: Dict[str, Any], final_answer: str) -> Tuple[float, float, float]:
    """Placeholder for aspect-level metrics."""
    return 0.0, 0.0, 0.0


def aggregate_final(
    category: str,
    code_inputs: Dict[str, Any],
    judge_scores: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine code-based metrics with judge outputs and return aggregate scores.
    Values are left as raw floats (0..1).
    """
    tool_use_code = _clamp_unit(code_inputs["tool_use"]["score_raw"])
    tool_use_judge = _clamp_unit(judge_scores.get("tool_use", 0.0))
    tool_use_final = max(tool_use_code, tool_use_judge)

    facts_exact = _clamp_unit(code_inputs.get("facts_exact", 0.0))
    citation = _clamp_unit(code_inputs.get("citation_f1", 0.0))
    factual_code = _clamp_unit(0.7 * facts_exact + 0.3 * citation)
    factual_judge = _clamp_unit(judge_scores.get("facts", 0.0))
    factual_final = max(factual_code, factual_judge)

    sentiment_code = _clamp_unit(code_inputs.get("sentiment_code", 0.0))
    sentiment_judge = _clamp_unit(judge_scores.get("sentiment", 0.0))
    sentiment_mix = _clamp_unit(0.7 * sentiment_code + 0.3 * sentiment_judge)
    sentiment_final = max(sentiment_mix, sentiment_judge)

    aspect_code = _clamp_unit(code_inputs["aspect"].get("f1", 0.0))
    aspect_judge = _clamp_unit(judge_scores.get("aspect_f1", 0.0))
    aspect_final = max(aspect_code, aspect_judge)

    answer_quality = _clamp_unit(judge_scores.get("overall", 0.0))

    overall_score = _clamp_unit(
        0.35 * tool_use_final
        + 0.35 * factual_final
        + 0.20 * sentiment_final
        + 0.10 * answer_quality
    )

    return {
        "tool_use_final": tool_use_final,
        "facts_final": factual_final,
        "sentiment_final": sentiment_final,
        "aspect_final": aspect_final,
        "answer_quality": answer_quality,
        "overall_score": overall_score,
        "narrative_final": _clamp_unit(judge_scores.get("narrative", 0.0)),
        "plan_final": _clamp_unit(judge_scores.get("plan", 0.0)),
        "intermediate": {
            "tool_use_code": tool_use_code,
            "tool_use_judge": tool_use_judge,
            "factual_code": factual_code,
            "factual_judge": factual_judge,
            "sentiment_code": sentiment_code,
            "sentiment_judge": sentiment_judge,
            "aspect_code": aspect_code,
            "aspect_judge": aspect_judge,
        },
    }
