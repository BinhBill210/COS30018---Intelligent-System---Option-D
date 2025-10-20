"""
LLM-as-a-Judge Implementation
Đánh giá subjective metrics bằng LLM
"""

from typing import Dict, Any, Optional, List
from gemini_llm import GeminiLLM, GeminiConfig
from langsmith.run_helpers import traceable
import re
import json

from metrics import tool_use_metrics


class LLMJudge:
    """
    LLM-as-a-Judge evaluator using Gemini
    
    Usage:
        judge = LLMJudge()
        result = judge.evaluate_relevance(query, answer, expected)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM Judge with Gemini"""
        config = GeminiConfig(
            temperature=0.1,  # Deterministic for consistent evaluation
            max_output_tokens=512
        )
        self.llm = GeminiLLM(api_key=api_key, config=config)
    
    @traceable(name="LLM_Judge_Relevance")
    def evaluate_relevance(self, query: str, answer: str, expected: str) -> Dict[str, Any]:
        """
        Đánh giá mức độ relevant của answer
        
        Returns:
            {
                "score": float (0-1),
                "justification": str,
                "raw_response": str
            }
        """
        prompt = f"""You are an expert evaluator. Rate the relevance of the answer to the query.

Query: {query}

Expected Answer: {expected}

Actual Answer: {answer}

Rate from 0-10 where:
- 0: Completely irrelevant or wrong
- 5: Partially relevant, missing key information
- 10: Perfectly relevant and complete

Provide your evaluation in this format:
Score: <number 0-10>
Justification: <one sentence explaining your score>
"""
        
        response = self.llm._call(prompt)
        score = self._parse_score(response)
        justification = self._parse_justification(response)
        
        return {
            "score": score / 10.0,  # Normalize to 0-1
            "justification": justification,
            "raw_response": response
        }
    
    @traceable(name="LLM_Judge_Helpfulness")
    def evaluate_helpfulness(self, query: str, answer: str) -> Dict[str, Any]:
        """Đánh giá mức độ helpful của answer"""
        prompt = f"""Rate how helpful this answer is to the user's query.

Query: {query}

Answer: {answer}

Rate from 0-10 where:
- 0: Not helpful at all, confusing or wrong
- 5: Somewhat helpful but incomplete
- 10: Extremely helpful and actionable

Format:
Score: <number>
Justification: <explanation>
"""
        
        response = self.llm._call(prompt)
        return {
            "score": self._parse_score(response) / 10.0,
            "justification": self._parse_justification(response),
            "raw_response": response
        }
    
    @traceable(name="LLM_Judge_Explanation_Quality")
    def evaluate_explanation_quality(self, answer: str) -> Dict[str, Any]:
        """Đánh giá chất lượng explanation/reasoning"""
        prompt = f"""Rate the quality of explanation and reasoning in this answer.

Answer: {answer}

Criteria:
- Clarity: Easy to understand?
- Logic: Reasoning makes sense?
- Evidence: Backed by facts/data?

Rate from 0-10 where:
- 0: No explanation or very poor
- 5: Basic explanation, lacks depth
- 10: Excellent, clear, logical explanation

Format:
Score: <number>
Justification: <explanation>
"""
        
        response = self.llm._call(prompt)
        return {
            "score": self._parse_score(response) / 10.0,
            "justification": self._parse_justification(response),
            "raw_response": response
        }

    def _extract_named_score(self, response: str, label: str) -> float:
        match = re.search(rf"{label}:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return min(max(value, 0.0), 10.0)
        return 0.0

    @traceable(name="LLM_Judge_Sentiment_Alignment")
    def evaluate_sentiment_alignment(self, reference: str, answer: str) -> Dict[str, Any]:
        """Đánh giá mức độ phù hợp về sentiment và tính chuyên nghiệp."""
        prompt = f"""You are assessing both sentiment alignment and professionalism.

Reference text (captures expected sentiment or factual tone):
{reference}

Actual answer:
{answer}

Please provide TWO separate ratings from 0-10:
- SentimentScore: how well the emotional tone (positive / neutral / negative) matches the reference.
- ProfessionalismScore: how appropriate and business-professional the answer sounds versus the reference.

Guidance:
- 0 means completely mismatched or inappropriate.
- 5 means partially aligned with noticeable issues.
- 10 means perfectly aligned and professional.

Format exactly as:
SentimentScore: <number>
ProfessionalismScore: <number>
Justification: <one-sentence explanation>
"""

        response = self.llm._call(prompt)
        sentiment_raw = self._extract_named_score(response, "SentimentScore")
        professionalism_raw = self._extract_named_score(response, "ProfessionalismScore")
        combined = (sentiment_raw + professionalism_raw) / 20.0  # normalise to 0-1
        return {
            "score": combined,
            "sentiment_component": sentiment_raw / 10.0,
            "professionalism_component": professionalism_raw / 10.0,
            "justification": self._parse_justification(response),
            "raw_response": response
        }
    
    def _parse_score(self, response: str) -> float:
        """Extract numeric score from LLM response"""
        match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 10.0)  # Clamp to [0, 10]
        return 0.0
    
    def _parse_justification(self, response: str) -> str:
        """Extract justification text from LLM response"""
        match = re.search(r"Justification:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if match:
            justification = match.group(1).strip()
            # Take only first sentence/line
            return justification.split('\n')[0][:200]
        return "No justification provided"

    def _expected_summary_text(self, ground_truth: Dict[str, Any]) -> str:
        summary = ground_truth.get("expected_answer_summary")
        if isinstance(summary, list):
            text = " ".join(str(item) for item in summary if item is not None)
        elif isinstance(summary, str):
            text = summary
        else:
            text = ""
        if not text:
            answer_gt = ground_truth.get("answer_gt")
            if isinstance(answer_gt, str):
                text = answer_gt
        return text or "No reference answer provided."

    def _normalise_tool_trace(self, tool_trace: Any) -> List[Dict[str, Any]]:
        if not isinstance(tool_trace, list):
            return []
        normalised = []
        for item in tool_trace:
            if isinstance(item, dict):
                name = item.get("name") or item.get("tool")
                normalised.append({
                    "name": str(name) if name else "",
                    "args": item.get("args", {}),
                    "ok": item.get("ok", True),
                })
            elif isinstance(item, str):
                normalised.append({"name": item, "args": {}, "ok": True})
            else:
                normalised.append({"name": str(item), "args": {}, "ok": True})
        return normalised

    def judge(self, context: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate LLM-based judgements for tool-use, factuality, sentiment, etc.
        """
        ground_truth = context.get("ground_truth") or {}
        query = context.get("user_query", "")
        final_answer = candidate.get("answer", "") or ""
        tool_trace = candidate.get("tool_trace") or candidate.get("tool_calls") or []
        tool_trace = self._normalise_tool_trace(tool_trace)

        expected_tools = ground_truth.get("expected_tool_chain") or []
        try:
            tool_metrics = tool_use_metrics(expected_tools, tool_trace)
            tool_use_score = tool_metrics.get("score_raw", 0.0)
        except Exception:
            tool_use_score = 0.0

        expected_text = self._expected_summary_text(ground_truth)
        relevance_score = 0.0
        helpfulness_score = 0.0
        explanation_score = 0.0
        justifications: List[str] = []

        try:
            relevance_result = self.evaluate_relevance(query, final_answer, expected_text)
            relevance_score = relevance_result.get("score", 0.0)
            justifications.append(relevance_result.get("justification") or "")
        except Exception as exc:
            justifications.append(f"Relevance eval failed: {exc}")

        try:
            helpfulness_result = self.evaluate_helpfulness(query, final_answer)
            helpfulness_score = helpfulness_result.get("score", 0.0)
            justifications.append(helpfulness_result.get("justification") or "")
        except Exception as exc:
            justifications.append(f"Helpfulness eval failed: {exc}")

        try:
            explanation_result = self.evaluate_explanation_quality(final_answer)
            explanation_score = explanation_result.get("score", 0.0)
            justifications.append(explanation_result.get("justification") or "")
        except Exception as exc:
            justifications.append(f"Explanation eval failed: {exc}")

        sentiment_score = 0.0
        sentiment_details: Dict[str, Any] = {}
        try:
            sentiment_result = self.evaluate_sentiment_alignment(expected_text, final_answer)
            sentiment_score = sentiment_result.get("score", 0.0)
            justifications.append(sentiment_result.get("justification") or "")
            sentiment_details = {
                "sentiment_alignment": sentiment_result.get("sentiment_component", sentiment_score),
                "professionalism": sentiment_result.get("professionalism_component", sentiment_score),
                "raw_response": sentiment_result.get("raw_response"),
            }
        except Exception as exc:
            justifications.append(f"Sentiment eval failed: {exc}")

        aspect_precision = 0.0
        aspect_recall = 0.0
        aspect_f1 = 0.0

        overall_components = [
            relevance_score,
            helpfulness_score,
            explanation_score,
            tool_use_score,
        ]
        overall_score = sum(overall_components) / len(overall_components) if overall_components else 0.0
        overall_score = max(0.0, min(1.0, overall_score))

        return {
            "overall_score": overall_score,
            "tool_use_score": tool_use_score,
            "facts_score": relevance_score,
            "sentiment_score": sentiment_score,
            "sentiment_breakdown": sentiment_details,
            "aspect_precision": aspect_precision,
            "aspect_recall": aspect_recall,
            "aspect_f1": aspect_f1,
            "reply_narrative_score": explanation_score,
            "plan_alignment_score": tool_use_score,
            "violations": [],
            "justification": " | ".join(j for j in justifications if j),
        }


# Singleton instance
_judge_instance = None

def get_llm_judge() -> LLMJudge:
    """Get singleton LLM Judge instance"""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = LLMJudge()
    return _judge_instance
