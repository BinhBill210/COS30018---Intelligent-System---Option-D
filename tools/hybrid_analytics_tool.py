#!/usr/bin/env python3
"""
Hybrid ChromaDB + DuckDB Analytics Tool
======================================

This tool combines the best of both worlds:
- ChromaDB: Semantic search for finding relevant reviews
- DuckDB: High-performance analytics for complex queries

Use Cases:
1. "Find reviews about service quality and analyze sentiment trends"
   → ChromaDB finds relevant reviews → DuckDB analyzes trends

2. "What are customers saying about food at high-rated restaurants?"
   → DuckDB filters high-rated restaurants → ChromaDB finds food-related reviews

3. "Analyze sentiment for businesses with delivery complaints"
   → ChromaDB finds delivery complaints → DuckDB analyzes business performance

Author: COS30018 Intelligent Systems Project
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from chromadb_integration import ChromaDBVectorStore
from duckdb_integration import DuckDBAnalytics
from tools.duckdb_analytics_tools import BusinessPerformanceTool, TrendAnalysisTool, SentimentAnalyticsTool
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class HybridAnalyticsTool:
    """
    Hybrid analytics tool combining ChromaDB semantic search with DuckDB analytics.
    
    This tool intelligently routes queries to the appropriate backend:
    - Use ChromaDB for semantic search and content discovery
    - Use DuckDB for aggregations, trends, and complex analytics
    - Combine results for comprehensive insights
    """
    
    def __init__(self, 
                 chroma_db_path: str = "./chroma_db",
                 duckdb_path: str = "business_analytics.duckdb"):
        """Initialize hybrid analytics with both vector search and SQL analytics."""
        
        # Vector search for semantic queries
        self.vector_store = ChromaDBVectorStore(
            collection_name="yelp_reviews",
            persist_directory=chroma_db_path
        )
        
        # SQL analytics for complex aggregations
        self.sql_analytics = DuckDBAnalytics(duckdb_path)
        
        # Specialized analytics tools
        self.business_perf = BusinessPerformanceTool(duckdb_path)
        self.trend_analysis = TrendAnalysisTool(duckdb_path)
        self.sentiment_analytics = SentimentAnalyticsTool(duckdb_path)
        
        logger.info("🔄 Hybrid ChromaDB + DuckDB Analytics Ready!")
    
    def __call__(self, query_type: str, **kwargs) -> Dict[str, Any]:
        """
        Route queries to appropriate analytics backend.
        
        Args:
            query_type: Type of analysis to perform
            **kwargs: Parameters specific to each query type
        
        Query Types:
        - 'semantic_search': Pure semantic search
        - 'business_analysis': Business performance with related reviews
        - 'sentiment_with_examples': Sentiment analysis with example reviews
        - 'competitive_with_reviews': Competitive analysis with review examples
        - 'trend_analysis': Trend analysis with supporting evidence
        """
        
        if query_type == "semantic_search":
            return self._semantic_search_analysis(kwargs)
        elif query_type == "business_analysis":
            return self._business_analysis_with_reviews(kwargs)
        elif query_type == "sentiment_with_examples":
            return self._sentiment_analysis_with_examples(kwargs)
        elif query_type == "competitive_with_reviews":
            return self._competitive_analysis_with_reviews(kwargs)
        elif query_type == "trend_analysis":
            return self._trend_analysis_with_evidence(kwargs)
        else:
            return {"error": f"Unknown query type: {query_type}"}
    
    def _semantic_search_analysis(self, params: Dict) -> Dict[str, Any]:
        """Pure semantic search with analytics summary."""
        
        query = params.get("query", "")
        k = params.get("k", 10)
        business_id = params.get("business_id", None)
        
        # Semantic search
        filter_dict = {"business_id": business_id} if business_id else None
        search_results = self.vector_store.similarity_search(query, k=k, filter_dict=filter_dict)
        
        # Convert results
        reviews = []
        business_ids = set()
        for doc, score in search_results:
            metadata = doc.metadata
            review_data = {
                "review_id": metadata.get("review_id", ""),
                "text": doc.page_content,
                "stars": metadata.get("stars", ""),
                "business_id": metadata.get("business_id", ""),
                "date": metadata.get("date", ""),
                "similarity_score": float(score)
            }
            reviews.append(review_data)
            business_ids.add(metadata.get("business_id", ""))
        
        # Get analytics summary for found businesses
        analytics_summary = {}
        if business_ids:
            for bid in list(business_ids)[:5]:  # Limit to top 5 businesses
                try:
                    biz_analysis = self.business_perf(business_id=bid, analysis_type="overview")
                    if "error" not in biz_analysis:
                        analytics_summary[bid] = {
                            "name": biz_analysis.get("name", "Unknown"),
                            "avg_rating": biz_analysis.get("recent_avg_rating", "N/A"),
                            "total_reviews": biz_analysis.get("recent_review_count", 0)
                        }
                except Exception as e:
                    logger.warning(f"Analytics failed for business {bid}: {e}")
        
        return {
            "query": query,
            "semantic_results": reviews,
            "business_analytics": analytics_summary,
            "total_found": len(reviews)
        }
    
    def _business_analysis_with_reviews(self, params: Dict) -> Dict[str, Any]:
        """Business performance analysis enhanced with relevant review examples."""
        
        business_id = params.get("business_id")
        if not business_id:
            return {"error": "business_id required"}
        
        # Get comprehensive business analytics
        overview = self.business_perf(business_id=business_id, analysis_type="overview")
        trends = self.business_perf(business_id=business_id, analysis_type="trends")
        
        # Get representative review examples
        positive_reviews = self.vector_store.similarity_search(
            "excellent great amazing fantastic", k=3, 
            filter_dict={"business_id": business_id}
        )
        
        negative_reviews = self.vector_store.similarity_search(
            "terrible awful bad disappointing poor", k=3,
            filter_dict={"business_id": business_id}
        )
        
        # Format review examples
        def format_reviews(search_results):
            return [{
                "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "stars": doc.metadata.get("stars", ""),
                "date": doc.metadata.get("date", "")
            } for doc, score in search_results]
        
        return {
            "business_analysis": overview,
            "performance_trends": trends,
            "review_examples": {
                "positive_reviews": format_reviews(positive_reviews),
                "negative_reviews": format_reviews(negative_reviews)
            }
        }
    
    def _sentiment_analysis_with_examples(self, params: Dict) -> Dict[str, Any]:
        """Sentiment analysis with supporting review examples."""
        
        category = params.get("category", None)
        city = params.get("city", None)
        
        # Get sentiment analytics
        sentiment_overview = self.sentiment_analytics(analysis_type="overview")
        
        if category:
            category_sentiment = self.sentiment_analytics(analysis_type="by_category")
        else:
            category_sentiment = None
        
        # Find representative reviews for different sentiment categories
        search_queries = {
            "positive": "excellent amazing wonderful fantastic great love",
            "negative": "terrible awful horrible disappointing hate bad",
            "neutral": "okay average decent fine"
        }
        
        sentiment_examples = {}
        for sentiment, query in search_queries.items():
            try:
                results = self.vector_store.similarity_search(query, k=3)
                sentiment_examples[sentiment] = [{
                    "text": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "stars": doc.metadata.get("stars", ""),
                    "business_id": doc.metadata.get("business_id", "")
                } for doc, score in results[:2]]  # Top 2 examples
            except Exception as e:
                logger.warning(f"Failed to get {sentiment} examples: {e}")
                sentiment_examples[sentiment] = []
        
        return {
            "sentiment_analytics": sentiment_overview,
            "category_breakdown": category_sentiment,
            "representative_examples": sentiment_examples
        }
    
    def _competitive_analysis_with_reviews(self, params: Dict) -> Dict[str, Any]:
        """Competitive analysis enhanced with review insights."""
        
        business_id = params.get("business_id")
        if not business_id:
            return {"error": "business_id required"}
        
        # Get competitive analysis
        competitive_analysis = self.business_perf(business_id=business_id, analysis_type="competitive")
        
        if "error" in competitive_analysis:
            return competitive_analysis
        
        # Get review insights for competitors
        competitors = competitive_analysis.get("competitors", [])
        competitor_insights = {}
        
        for comp in competitors[:3]:  # Top 3 competitors
            comp_id = comp["business_id"]
            
            # Find what customers say about each competitor
            try:
                comp_reviews = self.vector_store.similarity_search(
                    "service quality food experience", k=2,
                    filter_dict={"business_id": comp_id}
                )
                
                competitor_insights[comp_id] = {
                    "name": comp["name"],
                    "market_rank": comp["market_rank"],
                    "avg_rating": comp["avg_review_rating"],
                    "sample_reviews": [{
                        "text": doc.page_content[:100] + "...",
                        "stars": doc.metadata.get("stars", "")
                    } for doc, score in comp_reviews]
                }
            except Exception as e:
                logger.warning(f"Failed to get reviews for competitor {comp_id}: {e}")
        
        return {
            "competitive_analysis": competitive_analysis,
            "competitor_insights": competitor_insights
        }
    
    def _trend_analysis_with_evidence(self, params: Dict) -> Dict[str, Any]:
        """Trend analysis with supporting review evidence."""
        
        time_period = params.get("time_period", "monthly")
        category = params.get("category", None)
        
        # Get trend analysis
        trends = self.trend_analysis(time_period=time_period, category=category)
        
        # Find reviews that support the trends
        evidence_reviews = {}
        
        # Get recent reviews (if trends show recent changes)
        if trends["trend_direction"] in ["improving", "declining"]:
            trend_query = "recent experience lately now" if trends["trend_direction"] == "improving" else "worse declining problems"
            
            try:
                recent_evidence = self.vector_store.similarity_search(trend_query, k=3)
                evidence_reviews["trend_evidence"] = [{
                    "text": doc.page_content[:150] + "...",
                    "stars": doc.metadata.get("stars", ""),
                    "date": doc.metadata.get("date", ""),
                    "business_id": doc.metadata.get("business_id", "")
                } for doc, score in recent_evidence]
            except Exception as e:
                logger.warning(f"Failed to get trend evidence: {e}")
                evidence_reviews["trend_evidence"] = []
        
        return {
            "trend_analysis": trends,
            "supporting_evidence": evidence_reviews
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about both analytics systems."""
        
        # ChromaDB info
        vector_info = self.vector_store.get_collection_info()
        
        # DuckDB info  
        sql_info = self.sql_analytics.get_database_info()
        
        return {
            "hybrid_system": "ChromaDB + DuckDB",
            "vector_search": {
                "collection": vector_info.get("name", ""),
                "document_count": vector_info.get("count", 0),
                "status": "ready" if vector_info.get("count", 0) > 0 else "empty"
            },
            "sql_analytics": {
                "database": sql_info.get("database_path", ""),
                "businesses": sql_info.get("statistics", {}).get("businesses", 0),
                "reviews": sql_info.get("statistics", {}).get("reviews", 0),
                "status": "ready" if sql_info.get("statistics", {}).get("reviews", 0) > 0 else "empty"
            }
        }

# Test function
def test_hybrid_analytics():
    """Test the hybrid analytics system."""
    
    print("🔄 Testing Hybrid ChromaDB + DuckDB Analytics...")
    
    hybrid = HybridAnalyticsTool()
    
    # Test system info
    print("\n📊 System Information:")
    info = hybrid.get_system_info()
    print(f"Vector documents: {info['vector_search']['document_count']:,}")
    print(f"SQL businesses: {info['sql_analytics']['businesses']:,}")
    print(f"SQL reviews: {info['sql_analytics']['reviews']:,}")
    
    # Test semantic search
    print("\n🔍 Testing Semantic Search with Analytics:")
    search_result = hybrid(
        query_type="semantic_search",
        query="great food and service",
        k=5
    )
    print(f"Found {search_result['total_found']} relevant reviews")
    print(f"Analyzed {len(search_result['business_analytics'])} businesses")
    
    # Test sentiment analysis with examples
    print("\n😊 Testing Sentiment Analysis with Examples:")
    sentiment_result = hybrid(
        query_type="sentiment_with_examples"
    )
    sentiment_dist = sentiment_result['sentiment_analytics']['sentiment_distribution']
    print(f"Sentiment: {sentiment_dist['positive_pct']}% positive, {sentiment_dist['negative_pct']}% negative")
    
    # Test trend analysis
    print("\n📈 Testing Trend Analysis:")
    trend_result = hybrid(
        query_type="trend_analysis",
        time_period="monthly"
    )
    print(f"Market trend: {trend_result['trend_analysis']['trend_direction']}")
    
    print("\n✅ Hybrid analytics system tested successfully!")

if __name__ == "__main__":
    test_hybrid_analytics()
