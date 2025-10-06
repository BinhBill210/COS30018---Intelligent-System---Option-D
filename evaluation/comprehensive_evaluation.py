#!/usr/bin/env python3
"""
Comprehensive Business Intelligence Agent Evaluation Runner
=========================================================

This script implements the complete evaluation pipeline with:
- Phase 2: Basic Evaluation (Tool Usage + Response Quality)
- Phase 3: Automated Metrics (DeepEval Integration)

Usage:
    python evaluation/comprehensive_evaluation.py --phase 2 --tests 10
    python evaluation/comprehensive_evaluation.py --phase 3 --tests 5 --interactive
    python evaluation/comprehensive_evaluation.py --full --deepeval
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.agent_evaluator import BusinessAgentEvaluator
from evaluation.deepeval_integration import DeepEvalIntegration
from langchain_agent_chromadb import create_business_agent_chromadb
from gemini_llm import GeminiConfig

class ComprehensiveEvaluator:
    """Comprehensive evaluation system integrating all phases"""
    
    def __init__(self):
        self.evaluator = BusinessAgentEvaluator()
        self.deepeval = DeepEvalIntegration()
        self.agent_executor = None
        
    def load_agent(self) -> bool:
        """Load the agent with Gemini API"""
        try:
            print("🤖 Loading Business Intelligence Agent with Gemini API...")
            gemini_config = GeminiConfig(temperature=0.1, max_output_tokens=2048)
            
            self.agent_executor = create_business_agent_chromadb(
                model_type="gemini",
                gemini_config=gemini_config,
                max_iterations=15,
                verbose=False
            )
            print("✅ Agent loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load agent: {e}")
            print("💡 Make sure your GEMINI_API_KEY is set in .env file")
            return False
    
    def run_phase_2_evaluation(self, max_tests: int = 10, interactive: bool = False) -> Dict[str, Any]:
        """
        Phase 2: Basic Evaluation Implementation
        - Tool Usage Evaluation
        - Response Quality Evaluation (Manual/Automated)
        """
        
        if not self.agent_executor:
            if not self.load_agent():
                return {}
        
        print(f"\n🔬 Phase 2: Basic Evaluation ({max_tests} tests)")
        print("=" * 60)
        
        test_cases = self.evaluator.test_cases[:max_tests]
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n📋 Test {i+1}/{len(test_cases)}: {test_case['test_id']}")
            print(f"Query: {test_case['query']}")
            print(f"Category: {test_case['category']}")
            
            # Run the test
            start_time = time.time()
            try:
                response = self.agent_executor.invoke({
                    "input": test_case['query'],
                    "chat_history": ""
                })
                agent_response = response.get('output', '')
                actual_tools = self.evaluator._extract_tool_calls_from_response(str(response))
                success = True
                error = None
                
            except Exception as e:
                agent_response = ""
                actual_tools = []
                success = False
                error = str(e)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Phase 2 Evaluations
            expected_tools = test_case['expected_tool_chain']
            
            # Tool Usage Evaluation
            tool_usage_score = self.evaluator.evaluate_tool_usage(expected_tools, actual_tools)
            
            # Response Quality Evaluation
            response_quality_scores = self.evaluator.evaluate_response_quality(
                agent_response, 
                test_case['expected_answer_summary'],
                interactive=interactive
            )
            
            # Store results
            result = {
                'test_id': test_case['test_id'],
                'query': test_case['query'],
                'category': test_case['category'],
                'success': success,
                'error': error,
                'execution_time_ms': execution_time,
                'agent_response': agent_response,
                'expected_tools': expected_tools,
                'actual_tools': actual_tools,
                'tool_usage_score': tool_usage_score,
                'response_quality': response_quality_scores,
                'phase': 2
            }
            results.append(result)
            
            # Display results
            print(f"✅ Success: {success}")
            print(f"🔧 Tool Usage Score: {tool_usage_score:.2f}")
            print(f"📊 Response Quality:")
            for metric, score in response_quality_scores.items():
                print(f"   {metric.title()}: {score:.2f}")
            print(f"⏱️  Execution Time: {execution_time:.1f}ms")
            
            if not interactive and i < len(test_cases) - 1:
                time.sleep(1)  # Rate limiting for API
        
        return self._generate_phase_2_summary(results)
    
    def run_phase_3_evaluation(self, max_tests: int = 5) -> Dict[str, Any]:
        """
        Phase 3: Automated Metrics Implementation
        - DeepEval Integration
        - Comprehensive Automated Scoring
        """
        
        if not self.agent_executor:
            if not self.load_agent():
                return {}
        
        print(f"\n🤖 Phase 3: Automated Metrics Evaluation ({max_tests} tests)")
        print("=" * 60)
        
        test_cases = self.evaluator.test_cases[:max_tests]
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n📋 Test {i+1}/{len(test_cases)}: {test_case['test_id']}")
            print(f"Query: {test_case['query']}")
            
            # Run the test
            start_time = time.time()
            try:
                response = self.agent_executor.invoke({
                    "input": test_case['query'],
                    "chat_history": ""
                })
                agent_response = response.get('output', '')
                success = True
                error = None
                
            except Exception as e:
                agent_response = ""
                success = False
                error = str(e)
            
            execution_time = (time.time() - start_time) * 1000
            
            if success:
                # Extract context from response (simplified)
                context = [agent_response[:200]]  # Use part of response as context
                
                # DeepEval Automated Evaluation
                deepeval_scores = self.deepeval.automated_evaluation(
                    test_case['query'], 
                    agent_response, 
                    context
                )
                
                # Comprehensive Evaluation
                comprehensive_scores = self.deepeval.comprehensive_evaluation(
                    test_case['query'],
                    agent_response,
                    test_case['expected_answer_summary'],
                    context
                )
            else:
                deepeval_scores = {'relevancy': 0.0, 'faithfulness': 0.0, 'source': 'error'}
                comprehensive_scores = {'overall_score': 0.0}
            
            # Store results
            result = {
                'test_id': test_case['test_id'],
                'query': test_case['query'],
                'category': test_case['category'],
                'success': success,
                'error': error,
                'execution_time_ms': execution_time,
                'agent_response': agent_response,
                'deepeval_scores': deepeval_scores,
                'comprehensive_scores': comprehensive_scores,
                'phase': 3
            }
            results.append(result)
            
            # Display results
            print(f"✅ Success: {success}")
            if success:
                print(f"🎯 DeepEval Relevancy: {deepeval_scores['relevancy']:.2f}")
                print(f"🎯 DeepEval Faithfulness: {deepeval_scores['faithfulness']:.2f}")
                print(f"📊 Overall Score: {comprehensive_scores.get('overall_score', 0):.2f}")
            print(f"⏱️  Execution Time: {execution_time:.1f}ms")
            
            if i < len(test_cases) - 1:
                time.sleep(1)  # Rate limiting for API
        
        return self._generate_phase_3_summary(results)
    
    def run_full_evaluation(self, use_deepeval: bool = True) -> Dict[str, Any]:
        """Run complete evaluation across all test cases"""
        
        print("🚀 Full Comprehensive Evaluation")
        print("=" * 60)
        
        # Run basic evaluation first
        phase_2_results = self.run_phase_2_evaluation(max_tests=None, interactive=False)
        
        if use_deepeval:
            # Run automated evaluation on subset
            phase_3_results = self.run_phase_3_evaluation(max_tests=10)
            
            return {
                'phase_2': phase_2_results,
                'phase_3': phase_3_results,
                'evaluation_timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'phase_2': phase_2_results,
                'evaluation_timestamp': datetime.now().isoformat()
            }
    
    def _generate_phase_2_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate Phase 2 evaluation summary"""
        
        if not results:
            return {}
        
        successful_results = [r for r in results if r['success']]
        total_tests = len(results)
        passed_tests = len(successful_results)
        
        # Calculate averages
        avg_tool_usage = sum(r['tool_usage_score'] for r in results) / total_tests
        avg_quality_scores = {}
        
        if successful_results:
            quality_metrics = successful_results[0]['response_quality'].keys()
            for metric in quality_metrics:
                avg_quality_scores[f'avg_{metric}'] = sum(
                    r['response_quality'][metric] for r in successful_results
                ) / len(successful_results)
        
        # By category analysis
        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        category_results = {}
        for cat, cat_results in categories.items():
            successful_cat = [r for r in cat_results if r['success']]
            category_results[cat] = {
                'total': len(cat_results),
                'success_rate': len(successful_cat) / len(cat_results),
                'avg_tool_usage': sum(r['tool_usage_score'] for r in cat_results) / len(cat_results)
            }
        
        summary = {
            'phase': 2,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests,
            'avg_tool_usage_score': avg_tool_usage,
            **avg_quality_scores,
            'category_results': category_results,
            'detailed_results': results
        }
        
        return summary
    
    def _generate_phase_3_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate Phase 3 evaluation summary"""
        
        if not results:
            return {}
        
        successful_results = [r for r in results if r['success']]
        total_tests = len(results)
        passed_tests = len(successful_results)
        
        if successful_results:
            avg_relevancy = sum(r['deepeval_scores']['relevancy'] for r in successful_results) / len(successful_results)
            avg_faithfulness = sum(r['deepeval_scores']['faithfulness'] for r in successful_results) / len(successful_results)
            avg_overall = sum(r['comprehensive_scores'].get('overall_score', 0) for r in successful_results) / len(successful_results)
        else:
            avg_relevancy = avg_faithfulness = avg_overall = 0.0
        
        summary = {
            'phase': 3,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests,
            'avg_relevancy': avg_relevancy,
            'avg_faithfulness': avg_faithfulness,
            'avg_overall_score': avg_overall,
            'deepeval_source': results[0]['deepeval_scores']['source'] if results else 'unknown',
            'detailed_results': results
        }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename_suffix: str = ""):
        """Save evaluation results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("evaluation/results")
        results_dir.mkdir(exist_ok=True)
        
        filename = f"comprehensive_evaluation_{timestamp}{filename_suffix}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        return filepath

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Business Intelligence Agent Evaluation")
    parser.add_argument("--phase", type=int, choices=[2, 3], help="Run specific evaluation phase")
    parser.add_argument("--tests", type=int, default=5, help="Number of tests to run")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive response quality scoring")
    parser.add_argument("--full", action="store_true", help="Run full evaluation")
    parser.add_argument("--deepeval", action="store_true", help="Include DeepEval metrics in full evaluation")
    
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator()
    
    if args.phase == 2:
        print("🔬 Running Phase 2: Basic Evaluation")
        results = evaluator.run_phase_2_evaluation(args.tests, args.interactive)
        evaluator.save_results(results, "_phase2")
        
    elif args.phase == 3:
        print("🤖 Running Phase 3: Automated Metrics")
        results = evaluator.run_phase_3_evaluation(args.tests)
        evaluator.save_results(results, "_phase3")
        
    elif args.full:
        print("🚀 Running Full Evaluation")
        results = evaluator.run_full_evaluation(args.deepeval)
        evaluator.save_results(results, "_full")
        
    else:
        # Default: Quick demonstration
        print("🎯 Quick Evaluation Demo")
        results = evaluator.run_phase_2_evaluation(3, False)
        print(f"\n📊 Quick Summary:")
        print(f"  Success Rate: {results.get('success_rate', 0):.1%}")
        print(f"  Avg Tool Usage: {results.get('avg_tool_usage_score', 0):.2f}")

if __name__ == "__main__":
    main()
