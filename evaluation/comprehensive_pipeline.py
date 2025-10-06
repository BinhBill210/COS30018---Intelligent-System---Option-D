#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline Diagnostic and Setup
====================================================

Step-by-step pipeline to diagnose issues and run complete evaluation.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveEvaluationPipeline:
    """Complete evaluation pipeline with diagnostics and systematic testing"""
    
    def __init__(self):
        self.results = {}
        self.issues = []
        self.agent_executor = None
        
    def step_1_environment_validation(self) -> bool:
        """Step 1: Validate environment setup"""
        logger.info("🔍 Step 1: Environment Validation")
        
        try:
            # Check conda environment
            if 'biz-agent-gpu-2' not in os.environ.get('CONDA_DEFAULT_ENV', ''):
                self.issues.append("Not in correct conda environment")
                
            # Check .env file
            env_path = Path('.env')
            if not env_path.exists():
                self.issues.append("Missing .env file")
            
            # Check API keys
            from config.api_keys import APIKeyManager
            api_manager = APIKeyManager()
            gemini_key = api_manager.get_api_key('gemini')
            if not gemini_key:
                self.issues.append("Missing Gemini API key")
            
            logger.info(f"✅ Environment validation complete. Issues found: {len(self.issues)}")
            return len(self.issues) == 0
            
        except Exception as e:
            self.issues.append(f"Environment validation failed: {e}")
            return False
    
    def step_2_data_validation(self) -> bool:
        """Step 2: Validate data availability and database setup"""
        logger.info("🗄️ Step 2: Data Validation")
        
        try:
            # Check processed data files
            data_files = [
                'data/processed/business_cleaned.csv',
                'data/processed/business_cleaned.parquet'
            ]
            
            for file_path in data_files:
                if not Path(file_path).exists():
                    self.issues.append(f"Missing data file: {file_path}")
            
            # Check database availability
            try:
                from database.db_manager import get_db_manager
                db_manager = get_db_manager()
                # Test query
                result = db_manager.execute_query("SELECT COUNT(*) FROM businesses LIMIT 1")
                logger.info(f"✅ Database accessible with {result.iloc[0, 0]} businesses")
            except Exception as e:
                self.issues.append(f"Database not accessible: {e}")
                logger.warning("❌ Database setup needed")
            
            # Check ChromaDB
            try:
                import chromadb
                client = chromadb.HttpClient(host="172.24.104.210", port=8001)
                collection = client.get_collection("yelp_reviews")
                count = collection.count()
                logger.info(f"✅ ChromaDB accessible with {count} reviews")
            except Exception as e:
                self.issues.append(f"ChromaDB not accessible: {e}")
                logger.warning("❌ ChromaDB server not running")
            
            return len([i for i in self.issues if 'data' in i.lower() or 'database' in i.lower()]) == 0
            
        except Exception as e:
            self.issues.append(f"Data validation failed: {e}")
            return False
    
    def step_3_tool_validation(self) -> Dict[str, bool]:
        """Step 3: Test each tool individually"""
        logger.info("🔧 Step 3: Individual Tool Validation")
        
        tool_results = {}
        
        # Test each tool individually
        tools_to_test = [
            ('BusinessSearchTool', 'tools.business_search_tool', 'BusinessSearchTool'),
            ('ReviewSearchTool', 'tools.review_search_tool', 'ReviewSearchTool'),
            ('SentimentSummaryTool', 'tools.sentiment_summary_tool', 'SentimentSummaryTool'),
            ('DataSummaryTool', 'tools.data_summary_tool', 'DataSummaryTool'),
            ('AspectABSA', 'tools.aspect_analysis', 'AspectABSAToolHF'),
            ('ActionPlanner', 'tools.ActionPlanner', 'ActionPlannerTool'),
            ('HybridRetrieve', 'tools.hybrid_retrieval_tool', 'HybridRetrieve'),
            ('BusinessPulse', 'tools.business_pulse', 'BusinessPulse')
        ]
        
        for tool_name, module_path, class_name in tools_to_test:
            try:
                logger.info(f"Testing {tool_name}...")
                
                # Import and instantiate
                module = __import__(module_path, fromlist=[class_name])
                tool_class = getattr(module, class_name)
                
                # Different initialization parameters for different tools
                if tool_name in ['DataSummaryTool', 'HybridRetrieve', 'BusinessPulse']:
                    tool_instance = tool_class("data/processed/review_cleaned.parquet")
                elif tool_name in ['BusinessSearchTool', 'ReviewSearchTool']:
                    tool_instance = tool_class(host="172.24.104.210")
                else:
                    tool_instance = tool_class()
                
                # Test basic functionality
                if tool_name == 'BusinessSearchTool':
                    result = tool_instance.fuzzy_search("Vietnamese Food Truck", top_n=1)
                elif tool_name == 'SentimentSummaryTool':
                    result = tool_instance(["This is a great restaurant"])
                elif tool_name == 'DataSummaryTool':
                    result = tool_instance()
                else:
                    logger.info(f"  {tool_name} instantiated successfully")
                    result = "instantiated"
                
                tool_results[tool_name] = True
                logger.info(f"  ✅ {tool_name} working")
                
            except Exception as e:
                tool_results[tool_name] = False
                self.issues.append(f"{tool_name} failed: {e}")
                logger.error(f"  ❌ {tool_name} failed: {e}")
        
        working_tools = sum(tool_results.values())
        total_tools = len(tool_results)
        logger.info(f"✅ Tool validation complete: {working_tools}/{total_tools} tools working")
        
        return tool_results
    
    def step_4_agent_validation(self) -> bool:
        """Step 4: Test agent loading and basic functionality"""
        logger.info("🤖 Step 4: Agent Validation")
        
        try:
            from langchain_agent_chromadb import create_business_agent_chromadb
            from gemini_llm import GeminiConfig
            
            # Load agent
            gemini_config = GeminiConfig(temperature=0.1, max_output_tokens=1024)
            self.agent_executor = create_business_agent_chromadb(
                model_type="gemini",
                gemini_config=gemini_config,
                max_iterations=5,
                verbose=False
            )
            
            # Test basic agent response
            test_response = self.agent_executor.invoke({
                "input": "Hello, can you help me?",
                "chat_history": ""
            })
            
            if test_response and 'output' in test_response:
                logger.info("✅ Agent loaded and responding")
                return True
            else:
                self.issues.append("Agent not responding properly")
                return False
                
        except Exception as e:
            self.issues.append(f"Agent validation failed: {e}")
            logger.error(f"❌ Agent validation failed: {e}")
            return False
    
    def step_5_tool_call_extraction_test(self) -> bool:
        """Step 5: Test tool call extraction from agent responses"""
        logger.info("🔍 Step 5: Tool Call Extraction Test")
        
        if not self.agent_executor:
            self.issues.append("Agent not loaded for tool call test")
            return False
        
        try:
            # Test a simple query that should trigger tools
            test_query = "Tell me about Vietnamese Food Truck"
            response = self.agent_executor.invoke({
                "input": test_query,
                "chat_history": ""
            })
            
            full_response = str(response)
            logger.info(f"Agent response length: {len(full_response)} characters")
            
            # Test tool extraction patterns
            import re
            tool_pattern = r'\[TOOL CALLED\]\s+(\w+)'
            action_pattern = r'Action:\s+(\w+)'
            
            tools_found = re.findall(tool_pattern, full_response, re.IGNORECASE)
            actions_found = re.findall(action_pattern, full_response, re.IGNORECASE)
            
            all_tools = list(dict.fromkeys(tools_found + actions_found))
            
            logger.info(f"Tools extracted: {all_tools}")
            
            if len(all_tools) > 0:
                logger.info("✅ Tool call extraction working")
                return True
            else:
                self.issues.append("No tools extracted from agent response")
                logger.warning("❌ No tools extracted - may need pattern adjustment")
                return False
                
        except Exception as e:
            self.issues.append(f"Tool call extraction test failed: {e}")
            logger.error(f"❌ Tool call extraction test failed: {e}")
            return False
    
    def step_6_improved_tool_mapping(self) -> bool:
        """Step 6: Create improved tool name mapping"""
        logger.info("🔄 Step 6: Improving Tool Name Mapping")
        
        # Create comprehensive tool mapping
        improved_mapping = """
        def _normalize_tool_name(self, tool_name: str) -> str:
            \"\"\"Enhanced tool name normalization\"\"\"
            mapping = {
                # Search and retrieval tools
                'search_reviews': 'hybrid_retrieve',
                'hybrid_retrieve': 'hybrid_retrieve',
                'fuzzy_search': 'business_search',
                'business_fuzzy_search': 'business_search',
                'search_businesses': 'business_search',
                'get_business_id': 'business_search',
                'get_business_info': 'business_search',
                
                # Analysis tools
                'analyze_sentiment': 'aspect_analysis',
                'analyze_aspects': 'aspect_analysis',
                'aspect_analysis': 'aspect_analysis',
                
                # Business intelligence tools
                'get_data_summary': 'business_pulse',
                'business_pulse': 'business_pulse',
                'data_summary': 'business_pulse',
                
                # Action and planning tools
                'create_action_plan': 'action_planner',
                'action_planner': 'action_planner',
                'generate_review_response': 'review_response',
                'review_response': 'review_response'
            }
            
            normalized = tool_name.lower().strip()
            return mapping.get(normalized, normalized)
        """
        
        # Update the evaluator with improved mapping
        try:
            # Write improved mapping to a patch file
            with open('evaluation/improved_tool_mapping.py', 'w') as f:
                f.write(improved_mapping)
            
            logger.info("✅ Improved tool mapping created")
            return True
            
        except Exception as e:
            self.issues.append(f"Tool mapping improvement failed: {e}")
            return False
    
    def step_7_run_systematic_evaluation(self, max_tests: int = 10) -> Dict[str, Any]:
        """Step 7: Run systematic evaluation with diagnostics"""
        logger.info(f"📊 Step 7: Running Systematic Evaluation ({max_tests} tests)")
        
        if not self.agent_executor:
            return {"error": "Agent not loaded"}
        
        try:
            from evaluation.agent_evaluator import BusinessAgentEvaluator
            
            evaluator = BusinessAgentEvaluator()
            evaluator.agent_executor = self.agent_executor
            
            # Run evaluation with detailed logging
            test_cases = evaluator.test_cases[:max_tests]
            detailed_results = []
            
            for i, test_case in enumerate(test_cases):
                logger.info(f"Running test {i+1}/{len(test_cases)}: {test_case['test_id']}")
                
                start_time = time.time()
                try:
                    response = self.agent_executor.invoke({
                        "input": test_case['query'],
                        "chat_history": ""
                    })
                    
                    full_response = str(response)
                    agent_output = response.get('output', '')
                    
                    # Extract tools with improved patterns for LangChain
                    import re
                    patterns = [
                        # Primary patterns for LangChain AgentExecutor
                        r"AgentAction\(tool='([^']+)'",  # AgentAction(tool='tool_name'
                        r'Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)',  # Action: tool_name
                        r"tool='([^']+)'",  # tool='tool_name'
                        # Legacy patterns (kept for compatibility)
                        r'\[TOOL CALLED\]\s+(\w+)',
                        r'Tool:\s+(\w+)',
                        r'Using tool:\s+(\w+)'
                    ]
                    
                    extracted_tools = []
                    for pattern in patterns:
                        matches = re.findall(pattern, full_response, re.IGNORECASE)
                        extracted_tools.extend(matches)
                    
                    # Clean and deduplicate tools
                    unique_tools = []
                    seen = set()
                    for tool in extracted_tools:
                        tool_clean = tool.strip()
                        if tool_clean and tool_clean not in seen:
                            seen.add(tool_clean)
                            unique_tools.append(tool_clean)
                    
                    # Calculate scores
                    expected_tools = test_case['expected_tool_chain']
                    tool_score = evaluator.evaluate_tool_usage(expected_tools, unique_tools)
                    
                    execution_time = (time.time() - start_time) * 1000
                    
                    result = {
                        'test_id': test_case['test_id'],
                        'query': test_case['query'],
                        'category': test_case['category'],
                        'expected_tools': expected_tools,
                        'extracted_tools': unique_tools,
                        'tool_score': tool_score,
                        'execution_time_ms': execution_time,
                        'response_length': len(agent_output),
                        'success': True
                    }
                    
                    detailed_results.append(result)
                    logger.info(f"  Tool score: {tool_score:.2f}, Tools: {unique_tools}")
                    
                except Exception as e:
                    result = {
                        'test_id': test_case.get('test_id', f'test_{i}'),
                        'error': str(e),
                        'success': False
                    }
                    detailed_results.append(result)
                    logger.error(f"  Test failed: {e}")
                
                # Rate limiting
                time.sleep(1)
            
            # Calculate summary statistics
            successful_tests = [r for r in detailed_results if r.get('success', False)]
            if successful_tests:
                avg_tool_score = sum(r['tool_score'] for r in successful_tests) / len(successful_tests)
                avg_execution_time = sum(r['execution_time_ms'] for r in successful_tests) / len(successful_tests)
            else:
                avg_tool_score = 0.0
                avg_execution_time = 0.0
            
            summary = {
                'total_tests': len(detailed_results),
                'successful_tests': len(successful_tests),
                'success_rate': len(successful_tests) / len(detailed_results),
                'avg_tool_score': avg_tool_score,
                'avg_execution_time_ms': avg_execution_time,
                'detailed_results': detailed_results
            }
            
            logger.info(f"✅ Evaluation complete: {len(successful_tests)}/{len(detailed_results)} successful")
            logger.info(f"   Average tool score: {avg_tool_score:.2f}")
            logger.info(f"   Success rate: {summary['success_rate']:.1%}")
            
            return summary
            
        except Exception as e:
            self.issues.append(f"Systematic evaluation failed: {e}")
            return {"error": str(e)}
    
    def step_8_generate_comprehensive_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Step 8: Generate comprehensive analysis report"""
        logger.info("📋 Step 8: Generating Comprehensive Report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"evaluation/pipeline_report_{timestamp}.md"
        
        report = f"""# Comprehensive Evaluation Pipeline Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Pipeline Validation Results

### Issues Identified
{chr(10).join([f"- {issue}" for issue in self.issues]) if self.issues else "✅ No issues found"}

### Systematic Evaluation Results
- **Total Tests**: {evaluation_results.get('total_tests', 0)}
- **Successful Tests**: {evaluation_results.get('successful_tests', 0)}
- **Success Rate**: {evaluation_results.get('success_rate', 0):.1%}
- **Average Tool Score**: {evaluation_results.get('avg_tool_score', 0):.2f}
- **Average Execution Time**: {evaluation_results.get('avg_execution_time_ms', 0):.1f}ms

### Tool Usage Analysis
"""
        
        if 'detailed_results' in evaluation_results:
            successful_results = [r for r in evaluation_results['detailed_results'] if r.get('success', False)]
            
            # Analyze tool extraction patterns
            all_extracted_tools = []
            all_expected_tools = []
            
            for result in successful_results:
                all_extracted_tools.extend(result.get('extracted_tools', []))
                all_expected_tools.extend(result.get('expected_tools', []))
            
            from collections import Counter
            extracted_counts = Counter(all_extracted_tools)
            expected_counts = Counter(all_expected_tools)
            
            report += f"""
#### Most Extracted Tools
{chr(10).join([f"- {tool}: {count} times" for tool, count in extracted_counts.most_common(5)])}

#### Most Expected Tools  
{chr(10).join([f"- {tool}: {count} times" for tool, count in expected_counts.most_common(5)])}

#### Detailed Test Results
| Test ID | Category | Tool Score | Expected Tools | Extracted Tools |
|---------|----------|------------|----------------|-----------------|
"""
            
            for result in successful_results[:10]:  # Show first 10
                expected = ", ".join(result.get('expected_tools', []))
                extracted = ", ".join(result.get('extracted_tools', []))
                report += f"| {result['test_id']} | {result.get('category', 'N/A')} | {result['tool_score']:.2f} | {expected} | {extracted} |\\n"
        
        report += f"""

## Recommendations

### Immediate Actions Needed
{chr(10).join([f"1. Fix: {issue}" for issue in self.issues[:5]]) if self.issues else "✅ No immediate actions needed"}

### System Improvements
1. **Tool Name Mapping**: Update tool name normalization to handle more variations
2. **Database Setup**: Ensure all database tables are properly initialized
3. **Error Handling**: Improve error handling and recovery in tool calls
4. **Response Parsing**: Enhance tool call extraction patterns
5. **Performance Optimization**: Optimize slow tools and reduce API call latency

### Next Steps
1. Run full evaluation with all 100 test cases
2. Implement recommended fixes
3. Add more sophisticated evaluation metrics
4. Create automated daily evaluation pipeline
5. Add performance benchmarking and regression testing
"""
        
        # Save report
        try:
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"✅ Report saved to {report_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
        
        return report
    
    def run_complete_pipeline(self, max_tests: int = 10) -> Dict[str, Any]:
        """Run the complete evaluation pipeline"""
        logger.info("🚀 Starting Comprehensive Evaluation Pipeline")
        
        pipeline_results = {
            'started_at': datetime.now().isoformat(),
            'steps': {}
        }
        
        # Step 1: Environment Validation
        pipeline_results['steps']['environment'] = self.step_1_environment_validation()
        
        # Step 2: Data Validation  
        pipeline_results['steps']['data'] = self.step_2_data_validation()
        
        # Step 3: Tool Validation
        pipeline_results['steps']['tools'] = self.step_3_tool_validation()
        
        # Step 4: Agent Validation
        pipeline_results['steps']['agent'] = self.step_4_agent_validation()
        
        # Step 5: Tool Call Extraction Test
        pipeline_results['steps']['extraction'] = self.step_5_tool_call_extraction_test()
        
        # Step 6: Improved Tool Mapping
        pipeline_results['steps']['mapping'] = self.step_6_improved_tool_mapping()
        
        # Step 7: Systematic Evaluation
        evaluation_results = self.step_7_run_systematic_evaluation(max_tests)
        pipeline_results['steps']['evaluation'] = evaluation_results
        
        # Step 8: Generate Report
        report = self.step_8_generate_comprehensive_report(evaluation_results)
        pipeline_results['report'] = report
        
        pipeline_results['completed_at'] = datetime.now().isoformat()
        pipeline_results['total_issues'] = len(self.issues)
        pipeline_results['issues'] = self.issues
        
        logger.info("🎉 Comprehensive Evaluation Pipeline Complete!")
        return pipeline_results

def main():
    """Run the comprehensive evaluation pipeline"""
    pipeline = ComprehensiveEvaluationPipeline()
    results = pipeline.run_complete_pipeline(max_tests=5)  # Start with 5 tests
    
    print("\\n" + "="*60)
    print("COMPREHENSIVE EVALUATION PIPELINE RESULTS")
    print("="*60)
    print(f"Issues found: {results['total_issues']}")
    print(f"Steps completed: {len(results['steps'])}")
    
    if 'evaluation' in results['steps'] and isinstance(results['steps']['evaluation'], dict):
        eval_results = results['steps']['evaluation']
        print(f"Tests run: {eval_results.get('total_tests', 0)}")
        print(f"Success rate: {eval_results.get('success_rate', 0):.1%}")
        print(f"Average tool score: {eval_results.get('avg_tool_score', 0):.2f}")
    
    print("\\nCheck evaluation/pipeline_report_*.md for detailed analysis")

if __name__ == "__main__":
    main()
