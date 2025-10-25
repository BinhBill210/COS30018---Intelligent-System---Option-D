"""
Unified LLM-as-a-Judge Implementation for Agent Answer Quality Evaluation
Consolidates all answer quality metrics with improved prompts
"""

from typing import Dict, Any, Optional, List
from gemini_llm import GeminiLLM, GeminiConfig
from langsmith.run_helpers import traceable
import re
import json


class UnifiedLLMJudge:
    """
    Unified LLM-as-a-Judge evaluator with comprehensive answer quality assessment
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM Judge with Gemini"""
        config = GeminiConfig(
            temperature=0.3,  # Deterministic for consistent evaluation
            max_output_tokens=1024
        )
        self.llm = GeminiLLM(api_key=api_key, config=config)
    
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
    
    # ============================================================================
    # CORE EVALUATION METHODS
    # ============================================================================
    
    @traceable(name="Judge_Factual_Correctness")
    def evaluate_factual_correctness(
        self, 
        query: str, 
        answer: str, 
        expected: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate factual accuracy with context awareness
        
        Context can include:
        - business_id: For verifying correct business selection
        - citations_gt: Expected citations
        - query_type: specific vs ambiguous
        """
        query_type = context.get("query_type", "specific") if context else "specific"
        
        if query_type == "ambiguous":
            prompt = f"""You are evaluating a business query assistant's response to an AMBIGUOUS query.

USER QUERY: {query}
EXPECTED BEHAVIOR: For ambiguous queries (multiple businesses with same name), the agent should:
1. List multiple matching businesses with distinguishing details (location, address)
2. Ask user to clarify which one they mean
3. NOT make assumptions about which business the user wants

AGENT'S ACTUAL RESPONSE:
{answer}

REFERENCE ANSWER (for comparison):
{expected}

Evaluate factual correctness on a scale of 0-10:
- 10: Lists multiple businesses with clear distinguishing details, asks for clarification
- 7-9: Lists businesses but missing some details or clarity
- 4-6: Mentions multiple options but unclear or incomplete
- 1-3: Makes assumptions or provides info for only one business without asking
- 0: Factually incorrect or hallucinates business information

Score: <number 0-10>
Justification: <one sentence>"""
        else:
            prompt = f"""You are evaluating factual correctness of a business information response.

USER QUERY: {query}

AGENT'S ANSWER:
{answer}

REFERENCE ANSWER:
{expected}

Evaluate factual accuracy on a scale of 0-10:
- 10: All facts correct and match reference
- 7-9: Most facts correct, minor discrepancies
- 4-6: Some correct facts, but missing key information
- 1-3: Major factual errors or missing critical information
- 0: Completely incorrect or fabricated information

Focus on:
- Accuracy of business hours, addresses, ratings
- Correct business identification
- No hallucinated details

Score: <number 0-10>
Justification: <one sentence>"""
        
        response = self.llm._call(prompt)
        score = self._parse_score(response)
        justification = self._parse_justification(response)
        
        return {
            "score": score / 10.0,
            "justification": justification,
            "raw_response": response
        }
    
    @traceable(name="Judge_Completeness")
    def evaluate_completeness(
        self, 
        query: str, 
        answer: str,
        expected_points: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate if answer covers all expected key points
        """
        prompt = f"""You are evaluating whether a response covers all necessary information.

USER QUERY: {query}

AGENT'S ANSWER:
{answer}

EXPECTED KEY POINTS TO COVER:
{json.dumps(expected_points, indent=2)}

Rate completeness on a scale of 0-10:
- 10: Covers all expected points thoroughly
- 7-9: Covers most points, minor omissions
- 4-6: Covers some points but missing several
- 1-3: Minimal coverage, major gaps
- 0: Misses all key points

Consider:
- Are all expected points addressed?
- Is coverage sufficient for each point?
- Are there critical omissions?

Score: <number 0-10>
Justification: <one sentence>"""
        
        response = self.llm._call(prompt)
        score = self._parse_score(response)
        justification = self._parse_justification(response)
        
        return {
            "score": score / 10.0,
            "justification": justification,
            "raw_response": response
        }
    
    @traceable(name="Judge_Relevance")
    def evaluate_relevance(
        self, 
        query: str, 
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate how relevant the answer is to the query
        """
        prompt = f"""You are evaluating the relevance of a response to a user's query.

USER QUERY: {query}

AGENT'S ANSWER:
{answer}

Rate relevance on a scale of 0-10:
- 10: Perfectly addresses the exact question asked
- 7-9: Mostly relevant with minor tangents
- 4-6: Partially relevant but includes irrelevant info
- 1-3: Barely relevant, mostly off-topic
- 0: Completely irrelevant

Consider:
- Does it directly answer what was asked?
- Is information focused on the query?
- Are there unnecessary tangents?

Score: <number 0-10>
Justification: <one sentence>"""
        
        response = self.llm._call(prompt)
        score = self._parse_score(response)
        justification = self._parse_justification(response)
        
        return {
            "score": score / 10.0,
            "justification": justification,
            "raw_response": response
        }
    
    @traceable(name="Judge_Clarity")
    def evaluate_clarity(
        self, 
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate clarity and readability of the answer
        """
        prompt = f"""You are evaluating the clarity and readability of a response.

RESPONSE TO EVALUATE:
{answer}

Rate clarity on a scale of 0-10:
- 10: Crystal clear, well-structured, easy to understand
- 7-9: Clear with minor issues in structure or wording
- 4-6: Understandable but could be clearer
- 1-3: Confusing or poorly structured
- 0: Incomprehensible

Consider:
- Clear sentence structure
- Logical organization
- Easy to follow
- No ambiguous language

Score: <number 0-10>
Justification: <one sentence>"""
        
        response = self.llm._call(prompt)
        score = self._parse_score(response)
        justification = self._parse_justification(response)
        
        return {
            "score": score / 10.0,
            "justification": justification,
            "raw_response": response
        }
    
    @traceable(name="Judge_Helpfulness")
    def evaluate_helpfulness(
        self, 
        query: str, 
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate how helpful and actionable the answer is
        """
        prompt = f"""You are evaluating how helpful a response is to the user.

USER QUERY: {query}

AGENT'S ANSWER:
{answer}

Rate helpfulness on a scale of 0-10:
- 10: Extremely helpful, actionable, provides what user needs
- 7-9: Very helpful with minor room for improvement
- 4-6: Somewhat helpful but lacking actionable details
- 1-3: Minimally helpful
- 0: Not helpful at all

Consider:
- Does it solve the user's problem?
- Are details actionable?
- Is context provided when needed?
- Would user need to ask follow-ups?

Score: <number 0-10>
Justification: <one sentence>"""
        
        response = self.llm._call(prompt)
        score = self._parse_score(response)
        justification = self._parse_justification(response)
        
        return {
            "score": score / 10.0,
            "justification": justification,
            "raw_response": response
        }
    
    @traceable(name="Judge_Professionalism")
    def evaluate_professionalism(
        self, 
        answer: str,
        expected_tone: str = "professional"
    ) -> Dict[str, Any]:
        """
        Evaluate tone and professionalism
        """
        prompt = f"""You are evaluating the professionalism of a business assistant's response.

EXPECTED TONE: {expected_tone}

RESPONSE TO EVALUATE:
{answer}

Rate professionalism on a scale of 0-10:
- 10: Perfect professional tone, polite, appropriate
- 7-9: Professional with minor tone issues
- 4-6: Somewhat professional but has casual elements
- 1-3: Unprofessional or inappropriate tone
- 0: Completely unprofessional

Consider:
- Appropriate formality level
- Polite and respectful language
- No slang or overly casual expressions
- Professional courtesy phrases

Score: <number 0-10>
Justification: <one sentence>"""
        
        response = self.llm._call(prompt)
        score = self._parse_score(response)
        justification = self._parse_justification(response)
        
        return {
            "score": score / 10.0,
            "justification": justification,
            "raw_response": response
        }
    
    @traceable(name="Judge_Citation_Quality")
    def evaluate_citation_quality(
        self, 
        answer: str,
        expected_citations: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate if answer properly cites sources
        """
        prompt = f"""You are evaluating citation quality in a business information response.

RESPONSE:
{answer}

EXPECTED CITATIONS (sources that should be referenced):
{json.dumps(expected_citations, indent=2)}

Rate citation quality on a scale of 0-10:
- 10: All necessary sources cited appropriately
- 7-9: Most sources cited, minor omissions
- 4-6: Some citations but incomplete
- 1-3: Very few or poor citations
- 0: No citations or completely incorrect

Consider:
- Are business IDs or sources mentioned?
- Is the source of information clear?
- Are citations accurate?

Score: <number 0-10>
Justification: <one sentence>"""
        
        response = self.llm._call(prompt)
        score = self._parse_score(response)
        justification = self._parse_justification(response)
        
        return {
            "score": score / 10.0,
            "justification": justification,
            "raw_response": response
        }
    
    # ============================================================================
    # AGGREGATE EVALUATION
    # ============================================================================
    
    def judge_comprehensive(
        self, 
        context: Dict[str, Any], 
        candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation combining all quality dimensions
        
        Args:
            context: {
                "user_query": str,
                "ground_truth": {
                    "answer_gt": str,
                    "expected_answer_summary": List[str],
                    "citations_gt": List[str],
                    "query_type": str,
                    "response_tone": str
                }
            }
            candidate: {
                "answer": str,
                "tool_trace": List[Dict],
                ...
            }
        """
        query = context.get("user_query", "")
        ground_truth = context.get("ground_truth", {})
        answer = candidate.get("answer", "")
        
        # Extract ground truth elements
        # Use expected_answer_summary as the reference answer
        expected_points = ground_truth.get("expected_answer_summary", [])
        
        # Convert expected_answer_summary to string for comparison
        if isinstance(expected_points, list):
            expected_answer = " | ".join(str(p) for p in expected_points if p)
        else:
            expected_answer = str(expected_points) if expected_points else ""
        
        expected_citations = ground_truth.get("citations_gt", [])
        query_type = ground_truth.get("query_type", "specific")
        expected_tone = ground_truth.get("response_tone", "professional")
        
        results = {}
        justifications = []
        
        # 1. Factual Correctness (30%)
        try:
            factual_result = self.evaluate_factual_correctness(
                query, answer, expected_answer,
                context={"query_type": query_type}
            )
            results["factual_correctness"] = factual_result["score"]
            justifications.append(f"Factual: {factual_result['justification']}")
        except Exception as e:
            results["factual_correctness"] = 0.0
            justifications.append(f"Factual eval failed: {e}")
        
        # 2. Completeness (25%)
        if expected_points:
            try:
                completeness_result = self.evaluate_completeness(
                    query, answer, expected_points
                )
                results["completeness"] = completeness_result["score"]
                justifications.append(f"Complete: {completeness_result['justification']}")
            except Exception as e:
                results["completeness"] = 0.0
                justifications.append(f"Completeness eval failed: {e}")
        else:
            results["completeness"] = 1.0  # N/A
        
        # 3. Relevance (15%)
        try:
            relevance_result = self.evaluate_relevance(query, answer)
            results["relevance"] = relevance_result["score"]
            justifications.append(f"Relevant: {relevance_result['justification']}")
        except Exception as e:
            results["relevance"] = 0.0
            justifications.append(f"Relevance eval failed: {e}")
        
        # 4. Clarity (10%)
        try:
            clarity_result = self.evaluate_clarity(answer)
            results["clarity"] = clarity_result["score"]
            justifications.append(f"Clarity: {clarity_result['justification']}")
        except Exception as e:
            results["clarity"] = 0.0
            justifications.append(f"Clarity eval failed: {e}")
        
        # 5. Helpfulness (10%)
        try:
            helpfulness_result = self.evaluate_helpfulness(query, answer)
            results["helpfulness"] = helpfulness_result["score"]
            justifications.append(f"Helpful: {helpfulness_result['justification']}")
        except Exception as e:
            results["helpfulness"] = 0.0
            justifications.append(f"Helpfulness eval failed: {e}")
        
        # 6. Professionalism (5%)
        try:
            professionalism_result = self.evaluate_professionalism(answer, expected_tone)
            results["professionalism"] = professionalism_result["score"]
            justifications.append(f"Professional: {professionalism_result['justification']}")
        except Exception as e:
            results["professionalism"] = 0.0
            justifications.append(f"Professionalism eval failed: {e}")
        
        # 7. Citation Quality (5%)
        if expected_citations:
            try:
                citation_result = self.evaluate_citation_quality(answer, expected_citations)
                results["citation_quality"] = citation_result["score"]
                justifications.append(f"Citations: {citation_result['justification']}")
            except Exception as e:
                results["citation_quality"] = 0.0
                justifications.append(f"Citation eval failed: {e}")
        else:
            results["citation_quality"] = 1.0  # N/A
        
        # Calculate weighted overall score
        overall_score = (
            0.30 * results.get("factual_correctness", 0.0) +
            0.25 * results.get("completeness", 0.0) +
            0.15 * results.get("relevance", 0.0) +
            0.10 * results.get("clarity", 0.0) +
            0.10 * results.get("helpfulness", 0.0) +
            0.05 * results.get("professionalism", 0.0) +
            0.05 * results.get("citation_quality", 0.0)
        )
        
        return {
            "overall_score": max(0.0, min(1.0, overall_score)),
            "factual_correctness": results.get("factual_correctness", 0.0),
            "completeness": results.get("completeness", 0.0),
            "relevance": results.get("relevance", 0.0),
            "clarity": results.get("clarity", 0.0),
            "helpfulness": results.get("helpfulness", 0.0),
            "professionalism": results.get("professionalism", 0.0),
            "citation_quality": results.get("citation_quality", 0.0),
            "justification": " | ".join(justifications),
            "breakdown": results
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_unified_judge_instance = None

def get_unified_llm_judge() -> UnifiedLLMJudge:
    """Get singleton Unified LLM Judge instance"""
    global _unified_judge_instance
    if _unified_judge_instance is None:
        _unified_judge_instance = UnifiedLLMJudge()
    return _unified_judge_instance


# ============================================================================
# LANGSMITH EVALUATOR WRAPPERS
# ============================================================================

def answer_quality_evaluator(run, example) -> dict:
    """
    Comprehensive answer quality evaluator for LangSmith
    """
    judge = get_unified_llm_judge()
    
    # Prepare context
    context = {
        "user_query": example.inputs.get("query", ""),
        "ground_truth": example.outputs
    }
    
    # Prepare candidate
    candidate = {
        "answer": run.outputs.get("answer", "") if run.outputs else "",
        "tool_trace": run.outputs.get("tool_calls", []) if run.outputs else []
    }
    
    try:
        result = judge.judge_comprehensive(context, candidate)
        return {
            "key": "answer_quality",
            "score": result["overall_score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "answer_quality",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }


def factual_correctness_evaluator(run, example) -> dict:
    """LangSmith evaluator for factual correctness only"""
    judge = get_unified_llm_judge()
    
    query = example.inputs.get("query", "")
    answer = run.outputs.get("answer", "") if run.outputs else ""
    expected = example.outputs.get("answer_gt", "")
    query_type = example.outputs.get("query_type", "specific")
    
    try:
        result = judge.evaluate_factual_correctness(
            query, answer, expected,
            context={"query_type": query_type}
        )
        return {
            "key": "factual_correctness",
            "score": result["score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "factual_correctness",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }


def completeness_evaluator(run, example) -> dict:
    """LangSmith evaluator for answer completeness"""
    judge = get_unified_llm_judge()
    
    query = example.inputs.get("query", "")
    answer = run.outputs.get("answer", "") if run.outputs else ""
    expected_points = example.outputs.get("expected_answer_summary", [])
    
    if not expected_points:
        return {
            "key": "completeness",
            "score": 1.0,
            "comment": "No expected points defined"
        }
    
    try:
        result = judge.evaluate_completeness(query, answer, expected_points)
        return {
            "key": "completeness",
            "score": result["score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "completeness",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }


def relevance_evaluator(run, example) -> dict:
    """LangSmith evaluator for answer relevance"""
    judge = get_unified_llm_judge()
    
    query = example.inputs.get("query", "")
    answer = run.outputs.get("answer", "") if run.outputs else ""
    
    try:
        result = judge.evaluate_relevance(query, answer)
        return {
            "key": "relevance",
            "score": result["score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "relevance",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }


def clarity_evaluator(run, example) -> dict:
    """LangSmith evaluator for answer clarity"""
    judge = get_unified_llm_judge()
    
    answer = run.outputs.get("answer", "") if run.outputs else ""
    
    try:
        result = judge.evaluate_clarity(answer)
        return {
            "key": "clarity",
            "score": result["score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "clarity",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }


def helpfulness_evaluator(run, example) -> dict:
    """LangSmith evaluator for answer helpfulness"""
    judge = get_unified_llm_judge()
    
    query = example.inputs.get("query", "")
    answer = run.outputs.get("answer", "") if run.outputs else ""
    
    try:
        result = judge.evaluate_helpfulness(query, answer)
        return {
            "key": "helpfulness",
            "score": result["score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "helpfulness",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }


def professionalism_evaluator(run, example) -> dict:
    """LangSmith evaluator for professionalism"""
    judge = get_unified_llm_judge()
    
    answer = run.outputs.get("answer", "") if run.outputs else ""
    expected_tone = example.outputs.get("response_tone", "professional")
    
    try:
        result = judge.evaluate_professionalism(answer, expected_tone)
        return {
            "key": "professionalism",
            "score": result["score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "professionalism",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }


def citation_quality_evaluator(run, example) -> dict:
    """LangSmith evaluator for citation quality"""
    judge = get_unified_llm_judge()
    
    answer = run.outputs.get("answer", "") if run.outputs else ""
    expected_citations = example.outputs.get("citations_gt", [])
    
    if not expected_citations:
        return {
            "key": "citation_quality",
            "score": 1.0,
            "comment": "No citations expected"
        }
    
    try:
        result = judge.evaluate_citation_quality(answer, expected_citations)
        return {
            "key": "citation_quality",
            "score": result["score"],
            "comment": result["justification"]
        }
    except Exception as e:
        return {
            "key": "citation_quality",
            "score": 0.0,
            "comment": f"Evaluation failed: {str(e)}"
        }


# ============================================================================
# BACKWARD COMPATIBILITY WITH OLD LLMJudge
# ============================================================================

class LLMJudge(UnifiedLLMJudge):
    """
    Backward compatibility wrapper for old code that uses LLMJudge
    Maps old method names to new UnifiedLLMJudge methods
    """
    
    def evaluate_relevance(self, query: str, answer: str, expected: str) -> Dict[str, Any]:
        """Old relevance method - maps to new evaluate_relevance"""
        return super().evaluate_relevance(query, answer)
    
    def evaluate_helpfulness(self, query: str, answer: str) -> Dict[str, Any]:
        """Old helpfulness method - maps to new evaluate_helpfulness"""
        return super().evaluate_helpfulness(query, answer)
    
    def evaluate_explanation_quality(self, answer: str) -> Dict[str, Any]:
        """Old explanation quality method - maps to clarity"""
        return super().evaluate_clarity(answer)
    
    def evaluate_sentiment_alignment(self, reference: str, answer: str) -> Dict[str, Any]:
        """Old sentiment alignment method - maps to professionalism"""
        result = super().evaluate_professionalism(answer)
        # Return in old format with sentiment and professionalism components
        return {
            "score": result["score"],
            "sentiment_component": result["score"],
            "professionalism_component": result["score"],
            "justification": result["justification"],
            "raw_response": result.get("raw_response", "")
        }
    
    def _expected_summary_text(self, ground_truth: Dict[str, Any]) -> str:
        """Helper method for old judge format"""
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
        """Helper method to normalize tool trace"""
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
        OLD FORMAT judge method for backward compatibility
        Maps to new judge_comprehensive but returns old format
        """
        # Try new format first
        try:
            result = self.judge_comprehensive(context, candidate)
            
            # Map new format to old format
            return {
                "overall_score": result.get("overall_score", 0.0),
                "tool_use_score": 0.0,  # Not in new judge, set to 0
                "facts_score": result.get("factual_correctness", 0.0),
                "sentiment_score": result.get("professionalism", 0.0),
                "sentiment_breakdown": {
                    "sentiment_alignment": result.get("professionalism", 0.0),
                    "professionalism": result.get("professionalism", 0.0),
                },
                "aspect_precision": 0.0,
                "aspect_recall": 0.0,
                "aspect_f1": 0.0,
                "reply_narrative_score": result.get("clarity", 0.0),
                "plan_alignment_score": 0.0,  # Not in new judge
                "violations": [],
                "justification": result.get("justification", ""),
                # Include all new metrics for reference
                "completeness": result.get("completeness", 0.0),
                "relevance": result.get("relevance", 0.0),
                "helpfulness": result.get("helpfulness", 0.0),
                "citation_quality": result.get("citation_quality", 0.0),
            }
        except Exception as e:
            # Fallback to safe default
            return {
                "overall_score": 0.0,
                "tool_use_score": 0.0,
                "facts_score": 0.0,
                "sentiment_score": 0.0,
                "sentiment_breakdown": {},
                "aspect_precision": 0.0,
                "aspect_recall": 0.0,
                "aspect_f1": 0.0,
                "reply_narrative_score": 0.0,
                "plan_alignment_score": 0.0,
                "violations": [],
                "justification": f"Evaluation failed: {str(e)}",
            }


# Singleton instance for backward compatibility
_judge_instance = None

def get_llm_judge() -> LLMJudge:
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = LLMJudge()
    return _judge_instance