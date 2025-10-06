#!/usr/bin/env python3
"""
Test Runner for Business Intelligence Agent Evaluation
====================================================

Simple script to run evaluation tests against the agent.
This demonstrates how to use the evaluation framework.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.agent_evaluator import BusinessAgentEvaluator
from langchain_agent_chromadb import create_business_agent_chromadb
from gemini_llm import GeminiConfig

def run_sample_evaluation(max_tests: int = 5):
    """Run a sample evaluation with a few test cases"""
    
    print("🚀 Starting Business Intelligence Agent Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = BusinessAgentEvaluator()
    print(f"📊 Loaded {len(evaluator.test_cases)} test cases")
    
    # Create agent (using Gemini API as requested)
    print("🤖 Loading agent with Gemini API...")
    try:
        from gemini_llm import GeminiConfig
        gemini_config = GeminiConfig(temperature=0.1, max_output_tokens=2048)
        
        agent_executor = create_business_agent_chromadb(
            model_type="gemini",
            gemini_config=gemini_config,
            max_iterations=10,
            verbose=False  # Reduce noise during evaluation
        )
        print("✅ Agent with Gemini API loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load agent with Gemini: {e}")
        print("💡 Make sure your GEMINI_API_KEY is set in .env file")
        return
    
    # Run evaluation
    print(f"\n🧪 Running evaluation with {max_tests} test cases...")
    try:
        summary = evaluator.run_evaluation(agent_executor, max_tests=max_tests)
        
        # Print results
        print("\n📈 Evaluation Results:")
        print(f"  Total Tests: {summary.total_tests}")
        print(f"  Passed: {summary.passed_tests} ({summary.passed_tests/summary.total_tests*100:.1f}%)")
        print(f"  Failed: {summary.failed_tests} ({summary.failed_tests/summary.total_tests*100:.1f}%)")
        print(f"\n📊 Performance Metrics:")
        print(f"  Tool Selection Accuracy: {summary.avg_tool_selection_accuracy:.2f}")
        print(f"  Response Completeness: {summary.avg_response_completeness:.2f}")
        print(f"  Response Relevance: {summary.avg_response_relevance:.2f}")
        print(f"  Avg Execution Time: {summary.avg_execution_time_ms:.1f}ms")
        
        print(f"\n🎯 By Category:")
        for category, metrics in summary.category_results.items():
            print(f"  {category}:")
            print(f"    Success Rate: {metrics['success_rate']:.1%}")
            print(f"    Tool Accuracy: {metrics['avg_tool_accuracy']:.2f}")
            print(f"    Completeness: {metrics['avg_completeness']:.2f}")
        
        if summary.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in summary.recommendations:
                print(f"  - {rec}")
                
        print(f"\n✅ Evaluation complete! Check evaluation/results/ for detailed reports.")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def show_test_samples():
    """Show sample test cases from the golden dataset"""
    
    evaluator = BusinessAgentEvaluator()
    
    print("📋 Sample Test Cases from Golden Dataset:")
    print("=" * 50)
    
    # Show one example from each category
    categories_shown = set()
    
    for test_case in evaluator.test_cases:
        category = test_case['category']
        if category not in categories_shown:
            print(f"\n🎯 {category} Example:")
            print(f"  Query: {test_case['query']}")
            print(f"  Expected Tools: {', '.join(test_case['expected_tool_chain'])}")
            print(f"  Expected Elements: {len(test_case['expected_answer_summary'])} key points")
            categories_shown.add(category)
            
            if len(categories_shown) >= 3:  # Show 3 examples
                break
    
    print(f"\n📊 Total: {len(evaluator.test_cases)} test cases across {len(set(t['category'] for t in evaluator.test_cases))} categories")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Business Intelligence Agent Evaluation")
    parser.add_argument("--samples", action="store_true", help="Show sample test cases")
    parser.add_argument("--run", type=int, default=5, help="Run evaluation with N test cases")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (all 50 tests)")
    
    args = parser.parse_args()
    
    if args.samples:
        show_test_samples()
    elif args.full:
        run_sample_evaluation(max_tests=None)
    else:
        run_sample_evaluation(max_tests=args.run)
