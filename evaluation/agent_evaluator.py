#!/usr/bin/env python3
"""
Business Intelligence Agent Evaluation Framework
=============================================

This module provides a comprehensive evaluation framework for testing the 
Business Intelligence Agent against the golden test dataset.

Key Metrics:
1. Tool Selection Accuracy - Does the agent choose the right tools?
2. Response Quality - Does the final answer meet expectations?
3. Response Completeness - Are all expected elements present?
4. Tool Chain Efficiency - Does the agent use optimal tool sequences?
"""

import json
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
import re
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Single test case evaluation result"""
    test_id: str
    query: str
    category: str
    
    # Tool Selection Metrics
    expected_tools: List[str]
    actual_tools: List[str]
    tool_selection_accuracy: float
    tool_chain_efficiency: float
    
    # Response Quality Metrics
    expected_elements: List[str]
    response_completeness: float
    response_relevance: float
    
    # Performance Metrics
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None
    
    # Raw Data
    agent_response: str = ""
    expected_summary: List[str] = None

@dataclass
class EvaluationSummary:
    """Overall evaluation summary"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    
    # Aggregate Metrics
    avg_tool_selection_accuracy: float
    avg_response_completeness: float
    avg_response_relevance: float
    avg_execution_time_ms: float
    
    # By Category
    category_results: Dict[str, Dict[str, float]]
    
    # Top Issues
    common_failures: List[str]
    recommendations: List[str]

class BusinessAgentEvaluator:
    """Main evaluation framework for the Business Intelligence Agent"""
    
    def __init__(self, golden_dataset_path: str = "evaluation/golden_test_dataset.json"):
        """Initialize evaluator with golden dataset"""
        self.dataset_path = golden_dataset_path
        self.test_cases = self._load_dataset()
        self.results: List[EvaluationResult] = []
        
        # Agent instance will be set when running evaluation
        self.agent_executor = None
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the golden test dataset"""
        try:
            with open(self.dataset_path, 'r') as f:
                data = json.load(f)
            return data['test_cases']
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []
    
    def _extract_tool_calls_from_response(self, agent_output: str) -> List[str]:
        """Extract tool calls from agent response by parsing the output"""
        
        # Improved patterns for LangChain AgentExecutor responses
        patterns = [
            # Pattern 1: AgentAction(tool='tool_name' format
            r"AgentAction\(tool='([^']+)'",
            # Pattern 2: Action: tool_name in ReAct logs
            r'Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)',
            # Pattern 3: tool='tool_name' format
            r"tool='([^']+)'",
            # Pattern 4: Legacy patterns (kept for compatibility)
            r'\[TOOL CALLED\]\s+(\w+)',
            r'Tool:\s+(\w+)',
            r'Using tool:\s+(\w+)'
        ]
        
        all_tools = []
        for pattern in patterns:
            matches = re.findall(pattern, agent_output, re.IGNORECASE)
            all_tools.extend(matches)
        
        # Deduplicate while preserving order
        seen = set()
        unique_tools = []
        for tool in all_tools:
            tool_clean = tool.strip()
            if tool_clean and tool_clean not in seen:
                seen.add(tool_clean)
                unique_tools.append(tool_clean)
        
        return unique_tools
    
    def evaluate_tool_usage(self, expected_tools: List[str], actual_tools: List[str]) -> float:
        """Simple tool usage accuracy check as specified in Phase 2"""
        expected_normalized = [self._normalize_tool_name(tool) for tool in expected_tools]
        actual_normalized = [self._normalize_tool_name(tool) for tool in actual_tools]
        
        expected_set = set(expected_normalized)
        actual_set = set(actual_normalized)
        
        if expected_set == actual_set:
            return 1.0  # Perfect match
        elif any(tool in actual_set for tool in expected_set):
            return 0.5  # Partial match  
        else:
            return 0.0  # No match
    
    def _calculate_tool_selection_accuracy(self, expected: List[str], actual: List[str]) -> float:
        """Calculate tool selection accuracy score (enhanced version)"""
        if not expected:
            return 1.0 if not actual else 0.0
        
        # Use the simpler evaluation function for consistency
        return self.evaluate_tool_usage(expected, actual)
    
    def _normalize_tool_name(self, tool_name: str) -> str:
        """Normalize tool names for comparison"""
        # Map different tool name formats to standard names matching LangChain agent
        mapping = {
            # Primary LangChain tool names (exact matches)
            'search_reviews': 'search_reviews',
            'analyze_sentiment': 'analyze_sentiment', 
            'get_data_summary': 'get_data_summary',
            'get_business_id': 'get_business_id',
            'business_fuzzy_search': 'business_fuzzy_search',
            'search_businesses': 'search_businesses',
            'get_business_info': 'get_business_info',
            'analyze_aspects': 'analyze_aspects',
            'create_action_plan': 'create_action_plan',
            'generate_review_response': 'generate_review_response',
            'hybrid_retrieve': 'hybrid_retrieve',
            'business_pulse': 'business_pulse',
            
            # Alternative names and variations that might appear in test cases
            'fuzzy_search': 'business_fuzzy_search',
            'search_review': 'search_reviews',  # singular form
            'review_search': 'search_reviews',
            'business_search': 'search_businesses',
            'sentiment_analysis': 'analyze_sentiment',
            'data_summary': 'get_data_summary',
            'business_info': 'get_business_info',
            'aspect_analysis': 'analyze_aspects',
            'action_planner': 'create_action_plan',
            'review_response': 'generate_review_response',
            'hybrid_search': 'hybrid_retrieve',
            'pulse': 'business_pulse'
        }
        
        normalized = tool_name.lower().strip()
        return mapping.get(normalized, normalized)
    
    def _calculate_response_completeness(self, expected_elements: List[str], response: str) -> float:
        """Calculate how many expected elements are present in the response"""
        if not expected_elements:
            return 1.0
        
        response_lower = response.lower()
        found_elements = 0
        
        for element in expected_elements:
            # Check if key concepts from expected element are in response
            element_words = element.lower().split()
            key_words = [w for w in element_words if len(w) > 3]  # Focus on meaningful words
            
            if any(word in response_lower for word in key_words):
                found_elements += 1
        
        return found_elements / len(expected_elements)
    
    def evaluate_response_quality(self, response: str, criteria: List[str], interactive: bool = False) -> Dict[str, float]:
        """Manual scoring on 1-5 scale as specified in Phase 2"""
        scores = {}
        
        if interactive:
            print(f"\nResponse: {response}")
            print(f"Criteria: {', '.join(criteria)}")
            scores['accuracy'] = float(input("Accuracy (1-5): "))
            scores['relevance'] = float(input("Relevance (1-5): "))
            scores['completeness'] = float(input("Completeness (1-5): "))
        else:
            # Automated approximation when not in interactive mode
            scores['accuracy'] = self._estimate_accuracy(response, criteria)
            scores['relevance'] = self._estimate_relevance(response)
            scores['completeness'] = self._estimate_completeness(response, criteria)
        
        return scores
    
    def _estimate_accuracy(self, response: str, criteria: List[str]) -> float:
        """Estimate accuracy based on presence of expected criteria"""
        if not criteria:
            return 5.0
        
        response_lower = response.lower()
        matches = sum(1 for criterion in criteria if any(word.lower() in response_lower for word in criterion.split()))
        return min(5.0, 1.0 + (matches / len(criteria)) * 4.0)
    
    def _estimate_relevance(self, response: str) -> float:
        """Estimate relevance based on response length and structure"""
        if len(response.strip()) < 10:
            return 1.0
        elif "I don't" in response or "cannot" in response:
            return 2.0
        elif len(response.split()) > 50:
            return 4.0
        else:
            return 3.0
    
    def _estimate_completeness(self, response: str, criteria: List[str]) -> float:
        """Estimate completeness based on criteria coverage"""
        if not criteria:
            return 5.0
        
        response_lower = response.lower()
        covered = sum(1 for criterion in criteria if any(word.lower() in response_lower for word in criterion.split()))
        return min(5.0, 1.0 + (covered / len(criteria)) * 4.0)
    
    def _calculate_response_relevance(self, query: str, response: str) -> float:
        """Calculate relevance score based on query-response alignment"""
        # Simple relevance scoring based on keyword overlap
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Filter out common words
        common_words = {'the', 'a', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        query_words = query_words - common_words
        response_words = response_words - common_words
        
        if not query_words:
            return 1.0
        
        overlap = len(query_words.intersection(response_words))
        return min(overlap / len(query_words), 1.0)
    
    def evaluate_single_test(self, test_case: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single test case"""
        
        start_time = time.time()
        success = False
        error_message = None
        agent_response = ""
        actual_tools = []
        
        try:
            # Run the agent
            if not self.agent_executor:
                raise ValueError("Agent executor not set. Call set_agent() first.")
            
            query = test_case['query']
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": ""
            })
            
            agent_response = response.get('output', '')
            actual_tools = self._extract_tool_calls_from_response(str(response))
            success = True
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Test {test_case['test_id']} failed: {e}")
        
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Calculate metrics
        expected_tools = test_case['expected_tool_chain']
        tool_accuracy = self._calculate_tool_selection_accuracy(expected_tools, actual_tools)
        
        expected_elements = test_case['expected_answer_summary']
        completeness = self._calculate_response_completeness(expected_elements, agent_response)
        relevance = self._calculate_response_relevance(test_case['query'], agent_response)
        
        # Calculate efficiency (simpler tool chains are better)
        expected_chain_length = len(expected_tools)
        actual_chain_length = len(actual_tools)
        efficiency = 1.0 if expected_chain_length == 0 else min(expected_chain_length / max(actual_chain_length, 1), 1.0)
        
        return EvaluationResult(
            test_id=test_case['test_id'],
            query=test_case['query'],
            category=test_case['category'],
            expected_tools=expected_tools,
            actual_tools=actual_tools,
            tool_selection_accuracy=tool_accuracy,
            tool_chain_efficiency=efficiency,
            expected_elements=expected_elements,
            response_completeness=completeness,
            response_relevance=relevance,
            execution_time_ms=execution_time,
            success=success,
            error_message=error_message,
            agent_response=agent_response,
            expected_summary=expected_elements
        )
    
    def run_evaluation(self, agent_executor, max_tests: Optional[int] = None) -> EvaluationSummary:
        """Run complete evaluation against the golden dataset"""
        
        self.agent_executor = agent_executor
        self.results = []
        
        test_cases = self.test_cases[:max_tests] if max_tests else self.test_cases
        
        logger.info(f"Starting evaluation with {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test {i+1}/{len(test_cases)}: {test_case['test_id']}")
            
            result = self.evaluate_single_test(test_case)
            self.results.append(result)
            
            # Print progress
            if (i + 1) % 5 == 0:
                logger.info(f"Completed {i+1}/{len(test_cases)} tests")
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _generate_summary(self) -> EvaluationSummary:
        """Generate evaluation summary from results"""
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        if total_tests == 0:
            return EvaluationSummary(
                total_tests=0, passed_tests=0, failed_tests=0,
                avg_tool_selection_accuracy=0.0, avg_response_completeness=0.0,
                avg_response_relevance=0.0, avg_execution_time_ms=0.0,
                category_results={}, common_failures=[], recommendations=[]
            )
        
        # Calculate averages
        avg_tool_accuracy = sum(r.tool_selection_accuracy for r in self.results) / total_tests
        avg_completeness = sum(r.response_completeness for r in self.results) / total_tests
        avg_relevance = sum(r.response_relevance for r in self.results) / total_tests
        avg_execution_time = sum(r.execution_time_ms for r in self.results) / total_tests
        
        # Calculate by category
        categories = {}
        for result in self.results:
            cat = result.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        category_results = {}
        for cat, results in categories.items():
            category_results[cat] = {
                'count': len(results),
                'success_rate': sum(1 for r in results if r.success) / len(results),
                'avg_tool_accuracy': sum(r.tool_selection_accuracy for r in results) / len(results),
                'avg_completeness': sum(r.response_completeness for r in results) / len(results),
                'avg_relevance': sum(r.response_relevance for r in results) / len(results)
            }
        
        # Identify common failures
        failed_results = [r for r in self.results if not r.success]
        common_failures = [r.error_message for r in failed_results if r.error_message][:5]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(avg_tool_accuracy, avg_completeness, avg_relevance)
        
        return EvaluationSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            avg_tool_selection_accuracy=avg_tool_accuracy,
            avg_response_completeness=avg_completeness,
            avg_response_relevance=avg_relevance,
            avg_execution_time_ms=avg_execution_time,
            category_results=category_results,
            common_failures=common_failures,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, tool_accuracy: float, completeness: float, relevance: float) -> List[str]:
        """Generate improvement recommendations based on results"""
        recommendations = []
        
        if tool_accuracy < 0.7:
            recommendations.append("Improve tool selection logic - accuracy is below 70%")
        
        if completeness < 0.6:
            recommendations.append("Enhance response completeness - missing key information")
        
        if relevance < 0.7:
            recommendations.append("Improve response relevance to user queries")
        
        return recommendations
    
    def _save_results(self, summary: EvaluationSummary):
        """Save evaluation results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("evaluation/results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        detailed_results = {
            "metadata": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "dataset_path": self.dataset_path,
                "total_test_cases": len(self.results)
            },
            "summary": asdict(summary),
            "detailed_results": [asdict(r) for r in self.results]
        }
        
        detailed_path = results_dir / f"evaluation_detailed_{timestamp}.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save summary report
        summary_path = results_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("Business Intelligence Agent Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {summary.total_tests}\n")
            f.write(f"Passed: {summary.passed_tests} ({summary.passed_tests/summary.total_tests*100:.1f}%)\n")
            f.write(f"Failed: {summary.failed_tests} ({summary.failed_tests/summary.total_tests*100:.1f}%)\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  Tool Selection Accuracy: {summary.avg_tool_selection_accuracy:.2f}\n")
            f.write(f"  Response Completeness: {summary.avg_response_completeness:.2f}\n")
            f.write(f"  Response Relevance: {summary.avg_response_relevance:.2f}\n")
            f.write(f"  Avg Execution Time: {summary.avg_execution_time_ms:.1f}ms\n\n")
            
            f.write("By Category:\n")
            for cat, metrics in summary.category_results.items():
                f.write(f"  {cat}:\n")
                f.write(f"    Tests: {metrics['count']}\n")
                f.write(f"    Success Rate: {metrics['success_rate']:.2f}\n")
                f.write(f"    Tool Accuracy: {metrics['avg_tool_accuracy']:.2f}\n")
                f.write(f"    Completeness: {metrics['avg_completeness']:.2f}\n\n")
        
        logger.info(f"Results saved to {detailed_path} and {summary_path}")

    def set_agent(self, agent_executor):
        """Set the agent executor for evaluation"""
        self.agent_executor = agent_executor

# Usage example
if __name__ == "__main__":
    print("🧪 Business Intelligence Agent Evaluation Framework")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = BusinessAgentEvaluator()
    
    print(f"📊 Loaded {len(evaluator.test_cases)} test cases from golden dataset")
    print("\n🎯 Test Categories:")
    
    categories = {}
    for test_case in evaluator.test_cases:
        cat = test_case['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in categories.items():
        print(f"  - {cat}: {count} tests")
    
    print(f"\n✅ Evaluation framework ready!")
    print(f"🚀 To run evaluation:")
    print(f"   from langchain_agent_chromadb import create_business_agent_chromadb")
    print(f"   agent = create_business_agent_chromadb()")
    print(f"   evaluator.set_agent(agent)")
    print(f"   summary = evaluator.run_evaluation(agent, max_tests=5)  # Run first 5 tests")
