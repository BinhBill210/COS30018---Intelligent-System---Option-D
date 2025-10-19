"""
LLM-as-a-Judge Implementation
Đánh giá subjective metrics bằng LLM
"""

from typing import Dict, Any, Optional
from gemini_llm import GeminiLLM, GeminiConfig
from langsmith.run_helpers import traceable
import re
import json


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
            temperature=0.0,  # Deterministic for consistent evaluation
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


# Singleton instance
_judge_instance = None

def get_llm_judge() -> LLMJudge:
    """Get singleton LLM Judge instance"""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = LLMJudge()
    return _judge_instance