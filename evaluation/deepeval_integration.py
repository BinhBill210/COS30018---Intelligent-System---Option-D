#!/usr/bin/env python3
"""
DeepEval Integration for Automated Metrics
========================================

This module integrates DeepEval for automated evaluation metrics
as specified in Phase 3 of the evaluation plan.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DeepEvalIntegration:
    """DeepEval integration for automated evaluation metrics"""
    
    def __init__(self):
        """Initialize DeepEval integration"""
        self.deepeval_available = self._check_deepeval_availability()
        
        if self.deepeval_available:
            try:
                from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
                from deepeval.test_case import LLMTestCase
                self.AnswerRelevancyMetric = AnswerRelevancyMetric
                self.FaithfulnessMetric = FaithfulnessMetric
                self.LLMTestCase = LLMTestCase
                logger.info("✅ DeepEval integration initialized successfully")
            except ImportError as e:
                logger.warning(f"❌ DeepEval import failed: {e}")
                self.deepeval_available = False
        else:
            logger.warning("❌ DeepEval not available, using fallback metrics")
    
    def _check_deepeval_availability(self) -> bool:
        """Check if DeepEval is available"""
        try:
            import deepeval
            return True
        except ImportError:
            return False
    
    def automated_evaluation(self, query: str, agent_response: str, context: List[str] = None) -> Dict[str, float]:
        """
        Automated evaluation using DeepEval metrics as specified in Phase 3
        
        Args:
            query: The original user query
            agent_response: The agent's response
            context: Retrieved context/evidence for faithfulness evaluation
            
        Returns:
            Dictionary with relevancy and faithfulness scores
        """
        
        if not self.deepeval_available:
            return self._fallback_evaluation(query, agent_response, context)
        
        try:
            # Create test case
            test_case = self.LLMTestCase(
                input=query,
                actual_output=agent_response,
                retrieval_context=context or []
            )
            
            # Initialize metrics
            relevancy_metric = self.AnswerRelevancyMetric()
            faithfulness_metric = self.FaithfulnessMetric()
            
            # Measure metrics
            relevancy_score = relevancy_metric.measure(test_case)
            faithfulness_score = faithfulness_metric.measure(test_case)
            
            return {
                'relevancy': relevancy_score,
                'faithfulness': faithfulness_score,
                'source': 'deepeval'
            }
            
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            return self._fallback_evaluation(query, agent_response, context)
    
    def _fallback_evaluation(self, query: str, agent_response: str, context: List[str] = None) -> Dict[str, float]:
        """Fallback evaluation when DeepEval is not available"""
        
        # Simple relevancy scoring
        relevancy_score = self._calculate_relevancy_fallback(query, agent_response)
        
        # Simple faithfulness scoring  
        faithfulness_score = self._calculate_faithfulness_fallback(agent_response, context or [])
        
        return {
            'relevancy': relevancy_score,
            'faithfulness': faithfulness_score,
            'source': 'fallback'
        }
    
    def _calculate_relevancy_fallback(self, query: str, response: str) -> float:
        """Fallback relevancy calculation"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Filter out common words
        common_words = {'the', 'a', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = query_words - common_words
        response_words = response_words - common_words
        
        if not query_words:
            return 0.8  # Default score for queries with only common words
        
        overlap = len(query_words.intersection(response_words))
        return min(overlap / len(query_words), 1.0)
    
    def _calculate_faithfulness_fallback(self, response: str, context: List[str]) -> float:
        """Fallback faithfulness calculation"""
        if not context:
            return 0.5  # Neutral score when no context available
        
        # Simple check if response contains information from context
        response_lower = response.lower()
        context_matches = 0
        
        for ctx in context:
            ctx_words = set(ctx.lower().split())
            response_words = set(response_lower.split())
            
            if len(ctx_words.intersection(response_words)) > 0:
                context_matches += 1
        
        return min(context_matches / len(context), 1.0) if context else 0.5
    
    def comprehensive_evaluation(self, query: str, agent_response: str, 
                               expected_elements: List[str], context: List[str] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation combining DeepEval metrics with custom metrics
        
        Args:
            query: Original user query
            agent_response: Agent's response
            expected_elements: Expected elements in the response
            context: Retrieved context for faithfulness
            
        Returns:
            Comprehensive evaluation scores
        """
        
        # Get automated evaluation
        auto_scores = self.automated_evaluation(query, agent_response, context)
        
        # Add custom metrics
        completeness_score = self._calculate_completeness(agent_response, expected_elements)
        coherence_score = self._calculate_coherence(agent_response)
        
        return {
            **auto_scores,
            'completeness': completeness_score,
            'coherence': coherence_score,
            'overall_score': (
                auto_scores['relevancy'] * 0.3 +
                auto_scores['faithfulness'] * 0.3 +
                completeness_score * 0.25 +
                coherence_score * 0.15
            )
        }
    
    def _calculate_completeness(self, response: str, expected_elements: List[str]) -> float:
        """Calculate completeness based on expected elements"""
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
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate response coherence"""
        # Simple coherence metrics
        sentences = response.split('.')
        word_count = len(response.split())
        
        # Basic coherence indicators
        if word_count < 10:
            return 0.3  # Too short
        elif word_count > 500:
            return 0.7  # Very long, potentially repetitive
        elif len(sentences) < 2:
            return 0.5  # Single sentence responses
        else:
            return 0.8  # Good length and structure

def install_deepeval():
    """Helper function to install DeepEval if needed"""
    try:
        import subprocess
        import sys
        
        print("📦 Installing DeepEval...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deepeval"])
        print("✅ DeepEval installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to install DeepEval: {e}")
        return False

# Usage example
if __name__ == "__main__":
    # Initialize DeepEval integration
    deepeval_integration = DeepEvalIntegration()
    
    # Example evaluation
    query = "What are customers saying about the food quality at Vietnamese Food Truck?"
    response = "Based on reviews, customers generally praise the food quality at Vietnamese Food Truck, with many mentioning fresh ingredients and authentic flavors."
    context = ["Vietnamese Food Truck serves authentic Vietnamese cuisine", "Customers frequently mention fresh ingredients"]
    
    # Run automated evaluation
    scores = deepeval_integration.automated_evaluation(query, response, context)
    
    print("📊 Automated Evaluation Results:")
    print(f"  Relevancy: {scores['relevancy']:.2f}")
    print(f"  Faithfulness: {scores['faithfulness']:.2f}")
    print(f"  Source: {scores['source']}")
    
    # Run comprehensive evaluation
    expected_elements = ["food quality assessment", "customer opinions", "specific examples"]
    comprehensive_scores = deepeval_integration.comprehensive_evaluation(query, response, expected_elements, context)
    
    print("\n📈 Comprehensive Evaluation Results:")
    for metric, score in comprehensive_scores.items():
        print(f"  {metric.title()}: {score:.2f}")
