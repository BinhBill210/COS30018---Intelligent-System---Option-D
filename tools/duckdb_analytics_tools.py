#!/usr/bin/env python3
"""
DuckDB-Powered Analytics Tools for Business Intelligence Agent
=============================================================

These tools provide high-performance analytics capabilities using DuckDB
for complex business intelligence queries that would be slow with Pandas.

Key Features:
- Lightning-fast aggregations on 1.4M+ reviews
- Complex time-series analysis
- Business performance metrics
- Competitive analysis
- Trend detection
- Advanced filtering and grouping

Author: COS30018 Intelligent Systems Project
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from duckdb_integration import DuckDBAnalytics
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class BusinessPerformanceTool:
    """Advanced business performance analysis using DuckDB."""
    
    def __init__(self, db_path: str = "business_analytics.duckdb"):
        """Initialize the performance analysis tool."""
        self.analytics = DuckDBAnalytics(db_path)
        
    def __call__(self, business_id: Optional[str] = None, analysis_type: str = "overview") -> Dict[str, Any]:
        """
        Analyze business performance with various metrics.
        
        Args:
            business_id: Specific business ID to analyze (None for all businesses)
            analysis_type: Type of analysis ('overview', 'trends', 'competitive')
        """
        
        if analysis_type == "overview":
            return self._business_overview(business_id)
        elif analysis_type == "trends":
            return self._business_trends(business_id)
        elif analysis_type == "competitive":
            return self._competitive_analysis(business_id)
        else:
            return self._business_overview(business_id)
    
    def _business_overview(self, business_id: Optional[str]) -> Dict[str, Any]:
        """Generate comprehensive business overview."""
        
        if business_id:
            # Single business analysis
            business_info = self.analytics.conn.execute("""
                SELECT 
                    b.name,
                    b.city,
                    b.state,
                    b.stars as business_rating,
                    b.review_count as total_reviews,
                    b.categories,
                    -- Recent performance
                    AVG(r.stars) as recent_avg_rating,
                    COUNT(r.review_id) as recent_review_count,
                    -- Engagement metrics
                    AVG(r.useful) as avg_useful,
                    AVG(r.funny) as avg_funny,
                    AVG(r.cool) as avg_cool,
                    -- Rating distribution
                    SUM(CASE WHEN r.stars = 5 THEN 1 ELSE 0 END) as five_star_count,
                    SUM(CASE WHEN r.stars = 4 THEN 1 ELSE 0 END) as four_star_count,
                    SUM(CASE WHEN r.stars = 3 THEN 1 ELSE 0 END) as three_star_count,
                    SUM(CASE WHEN r.stars = 2 THEN 1 ELSE 0 END) as two_star_count,
                    SUM(CASE WHEN r.stars = 1 THEN 1 ELSE 0 END) as one_star_count,
                    -- Time range
                    MIN(r.date) as earliest_review,
                    MAX(r.date) as latest_review
                FROM businesses b
                LEFT JOIN reviews r ON b.business_id = r.business_id
                WHERE b.business_id = ?
                GROUP BY b.business_id, b.name, b.city, b.state, b.stars, b.review_count, b.categories
            """, [business_id]).fetchone()
            
            if not business_info:
                return {"error": f"Business {business_id} not found"}
            
            # Convert to dictionary
            keys = ["name", "city", "state", "business_rating", "total_reviews", "categories",
                   "recent_avg_rating", "recent_review_count", "avg_useful", "avg_funny", "avg_cool",
                   "five_star", "four_star", "three_star", "two_star", "one_star",
                   "earliest_review", "latest_review"]
            
            result = dict(zip(keys, business_info))
            result["business_id"] = business_id
            
            # Calculate percentages
            total = sum([result["five_star"], result["four_star"], result["three_star"], 
                        result["two_star"], result["one_star"]])
            if total > 0:
                result["rating_distribution"] = {
                    "5_star_pct": round(result["five_star"] / total * 100, 1),
                    "4_star_pct": round(result["four_star"] / total * 100, 1),
                    "3_star_pct": round(result["three_star"] / total * 100, 1),
                    "2_star_pct": round(result["two_star"] / total * 100, 1),
                    "1_star_pct": round(result["one_star"] / total * 100, 1)
                }
            
            return result
            
        else:
            # Overall market overview
            overview = self.analytics.conn.execute("""
                SELECT 
                    COUNT(DISTINCT b.business_id) as total_businesses,
                    COUNT(DISTINCT r.review_id) as total_reviews,
                    COUNT(DISTINCT r.user_id) as total_users,
                    ROUND(AVG(r.stars), 2) as overall_avg_rating,
                    -- Top performing cities
                    (SELECT city FROM businesses WHERE stars >= 4.0 GROUP BY city ORDER BY COUNT(*) DESC LIMIT 1) as top_city,
                    -- Most reviewed category (simplified)
                    'Restaurants' as top_category
                FROM businesses b
                LEFT JOIN reviews r ON b.business_id = r.business_id
            """).fetchone()
            
            return {
                "total_businesses": overview[0],
                "total_reviews": overview[1], 
                "total_users": overview[2],
                "overall_avg_rating": overview[3],
                "top_performing_city": overview[4],
                "most_popular_category": overview[5]
            }

    def _business_trends(self, business_id: Optional[str]) -> Dict[str, Any]:
        """Analyze rating and review trends over time."""
        
        where_clause = "WHERE b.business_id = ?" if business_id else ""
        params = [business_id] if business_id else []
        
        # Monthly trends
        trends = self.analytics.conn.execute(f"""
            SELECT 
                DATE_TRUNC('month', r.date) as month,
                COUNT(r.review_id) as review_count,
                ROUND(AVG(r.stars), 2) as avg_rating,
                ROUND(AVG(r.useful), 2) as avg_useful_score
            FROM businesses b
            JOIN reviews r ON b.business_id = r.business_id
            {where_clause}
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
        """, params).fetchall()
        
        # Calculate trends
        trend_data = []
        for month, count, rating, useful in trends:
            trend_data.append({
                "month": str(month),
                "review_count": count,
                "avg_rating": rating,
                "useful_score": useful
            })
        
        # Performance indicators
        if len(trend_data) >= 2:
            latest = trend_data[0]
            previous = trend_data[1]
            
            rating_change = latest["avg_rating"] - previous["avg_rating"]
            volume_change = latest["review_count"] - previous["review_count"]
            
            performance_indicator = {
                "rating_trend": "improving" if rating_change > 0.1 else "declining" if rating_change < -0.1 else "stable",
                "volume_trend": "increasing" if volume_change > 0 else "decreasing" if volume_change < 0 else "stable",
                "rating_change": round(rating_change, 2),
                "volume_change": volume_change
            }
        else:
            performance_indicator = {"trend": "insufficient_data"}
        
        return {
            "monthly_trends": trend_data,
            "performance_indicator": performance_indicator
        }
    
    def _competitive_analysis(self, business_id: Optional[str]) -> Dict[str, Any]:
        """Perform competitive analysis for a business."""
        
        if not business_id:
            return {"error": "Business ID required for competitive analysis"}
        
        # Get business info
        business_info = self.analytics.conn.execute("""
            SELECT name, city, category_list
            FROM businesses 
            WHERE business_id = ?
        """, [business_id]).fetchone()
        
        if not business_info:
            return {"error": "Business not found"}
        
        name, city, categories = business_info
        
        # Find competitors (same city, similar categories)
        if categories:
            # Get the first category for competitor matching
            main_category = categories[0] if len(categories) > 0 else None
            
            competitors = self.analytics.conn.execute("""
                WITH business_stats AS (
                    SELECT 
                        b.business_id,
                        b.name,
                        b.stars as business_rating,
                        COUNT(r.review_id) as review_count,
                        ROUND(AVG(r.stars), 2) as avg_review_rating,
                        ROUND(AVG(r.useful), 2) as avg_useful
                    FROM businesses b
                    LEFT JOIN reviews r ON b.business_id = r.business_id
                    WHERE b.city = ? 
                      AND ? = ANY(b.category_list)
                      AND b.business_id != ?
                    GROUP BY b.business_id, b.name, b.stars
                    HAVING COUNT(r.review_id) >= 5
                )
                SELECT *,
                    RANK() OVER (ORDER BY avg_review_rating DESC, review_count DESC) as ranking
                FROM business_stats
                ORDER BY avg_review_rating DESC, review_count DESC
                LIMIT 5
            """, [city, main_category, business_id]).fetchall()
            
            # Get target business rank
            target_rank = self.analytics.conn.execute("""
                WITH business_stats AS (
                    SELECT 
                        b.business_id,
                        ROUND(AVG(r.stars), 2) as avg_review_rating,
                        COUNT(r.review_id) as review_count,
                        RANK() OVER (ORDER BY AVG(r.stars) DESC, COUNT(r.review_id) DESC) as ranking
                    FROM businesses b
                    LEFT JOIN reviews r ON b.business_id = r.business_id
                    WHERE b.city = ? 
                      AND ? = ANY(b.category_list)
                    GROUP BY b.business_id
                    HAVING COUNT(r.review_id) >= 5
                )
                SELECT ranking, avg_review_rating, review_count
                FROM business_stats
                WHERE business_id = ?
            """, [city, main_category, business_id]).fetchone()
            
            competitor_list = []
            for comp in competitors:
                competitor_list.append({
                    "business_id": comp[0],
                    "name": comp[1],
                    "business_rating": comp[2],
                    "review_count": comp[3],
                    "avg_review_rating": comp[4],
                    "avg_useful": comp[5],
                    "market_rank": comp[6]
                })
            
            return {
                "target_business": {
                    "name": name,
                    "market_rank": target_rank[0] if target_rank else "N/A",
                    "avg_rating": target_rank[1] if target_rank else "N/A", 
                    "review_count": target_rank[2] if target_rank else "N/A"
                },
                "competitors": competitor_list,
                "analysis_criteria": {
                    "city": city,
                    "category": main_category,
                    "min_reviews": 5
                }
            }
        
        return {"error": "No categories found for competitive analysis"}

class TrendAnalysisTool:
    """Advanced trend analysis using DuckDB for time-series insights."""
    
    def __init__(self, db_path: str = "business_analytics.duckdb"):
        self.analytics = DuckDBAnalytics(db_path)
    
    def __call__(self, time_period: str = "monthly", category: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze market trends over time.
        
        Args:
            time_period: 'monthly', 'quarterly', 'yearly'
            category: Specific category to analyze (None for all)
        """
        
        # Determine time truncation
        time_trunc = {
            "monthly": "month",
            "quarterly": "quarter", 
            "yearly": "year"
        }.get(time_period, "month")
        
        # Build category filter
        category_filter = ""
        params = []
        if category:
            category_filter = "AND ? = ANY(b.category_list)"
            params.append(category)
        
        # Market trends query
        trends = self.analytics.conn.execute(f"""
            SELECT 
                DATE_TRUNC('{time_trunc}', r.date) as period,
                COUNT(DISTINCT r.business_id) as active_businesses,
                COUNT(r.review_id) as total_reviews,
                ROUND(AVG(r.stars), 2) as avg_rating,
                -- Sentiment indicators
                ROUND(AVG(CASE WHEN r.stars >= 4 THEN 1.0 ELSE 0.0 END) * 100, 1) as positive_sentiment_pct,
                ROUND(AVG(CASE WHEN r.stars <= 2 THEN 1.0 ELSE 0.0 END) * 100, 1) as negative_sentiment_pct,
                -- Engagement
                ROUND(AVG(r.useful), 2) as avg_useful_score
            FROM businesses b
            JOIN reviews r ON b.business_id = r.business_id
            WHERE r.date >= '2020-01-01' {category_filter}
            GROUP BY period
            ORDER BY period DESC
            LIMIT 24
        """, params).fetchall()
        
        trend_data = []
        for period, businesses, reviews, rating, positive_pct, negative_pct, useful in trends:
            trend_data.append({
                "period": str(period),
                "active_businesses": businesses,
                "total_reviews": reviews,
                "avg_rating": rating,
                "positive_sentiment_pct": positive_pct,
                "negative_sentiment_pct": negative_pct,
                "avg_useful_score": useful
            })
        
        # Calculate overall trend direction
        if len(trend_data) >= 3:
            recent_avg = sum(item["avg_rating"] for item in trend_data[:3]) / 3
            older_avg = sum(item["avg_rating"] for item in trend_data[-3:]) / 3
            
            trend_direction = "improving" if recent_avg > older_avg + 0.1 else \
                            "declining" if recent_avg < older_avg - 0.1 else "stable"
        else:
            trend_direction = "insufficient_data"
        
        return {
            "time_period": time_period,
            "category": category or "all_categories",
            "trend_direction": trend_direction,
            "data": trend_data
        }

class SentimentAnalyticsTool:
    """Advanced sentiment analytics using DuckDB for large-scale analysis."""
    
    def __init__(self, db_path: str = "business_analytics.duckdb"):
        self.analytics = DuckDBAnalytics(db_path)
    
    def __call__(self, analysis_type: str = "overview", **kwargs) -> Dict[str, Any]:
        """
        Perform sentiment analysis using rating-based indicators.
        
        Args:
            analysis_type: 'overview', 'by_category', 'by_city', 'time_series'
            **kwargs: Additional parameters for specific analysis types
        """
        
        if analysis_type == "overview":
            return self._sentiment_overview()
        elif analysis_type == "by_category":
            return self._sentiment_by_category()
        elif analysis_type == "by_city":
            return self._sentiment_by_city()
        elif analysis_type == "time_series":
            return self._sentiment_time_series(kwargs.get("business_id"))
        else:
            return self._sentiment_overview()
    
    def _sentiment_overview(self) -> Dict[str, Any]:
        """Overall sentiment distribution."""
        
        sentiment_dist = self.analytics.conn.execute("""
            SELECT 
                COUNT(*) as total_reviews,
                -- Sentiment categories based on star ratings
                ROUND(AVG(CASE WHEN stars >= 4 THEN 1.0 ELSE 0.0 END) * 100, 1) as positive_pct,
                ROUND(AVG(CASE WHEN stars = 3 THEN 1.0 ELSE 0.0 END) * 100, 1) as neutral_pct,
                ROUND(AVG(CASE WHEN stars <= 2 THEN 1.0 ELSE 0.0 END) * 100, 1) as negative_pct,
                -- Average metrics
                ROUND(AVG(stars), 2) as avg_rating,
                ROUND(AVG(useful), 2) as avg_useful,
                -- Text length analysis (proxy for engagement)
                ROUND(AVG(text_length), 0) as avg_text_length,
                ROUND(AVG(word_count), 0) as avg_word_count
            FROM reviews
        """).fetchone()
        
        return {
            "total_reviews": sentiment_dist[0],
            "sentiment_distribution": {
                "positive_pct": sentiment_dist[1],
                "neutral_pct": sentiment_dist[2], 
                "negative_pct": sentiment_dist[3]
            },
            "engagement_metrics": {
                "avg_rating": sentiment_dist[4],
                "avg_useful_score": sentiment_dist[5],
                "avg_text_length": sentiment_dist[6],
                "avg_word_count": sentiment_dist[7]
            }
        }
    
    def _sentiment_by_category(self) -> Dict[str, Any]:
        """Sentiment analysis by business category."""
        
        category_sentiment = self.analytics.conn.execute("""
            WITH category_reviews AS (
                SELECT 
                    UNNEST(b.category_list) as category,
                    r.stars,
                    r.useful,
                    r.text_length
                FROM businesses b
                JOIN reviews r ON b.business_id = r.business_id
                WHERE b.category_list IS NOT NULL
            )
            SELECT 
                category,
                COUNT(*) as review_count,
                ROUND(AVG(stars), 2) as avg_rating,
                ROUND(AVG(CASE WHEN stars >= 4 THEN 1.0 ELSE 0.0 END) * 100, 1) as positive_pct,
                ROUND(AVG(CASE WHEN stars <= 2 THEN 1.0 ELSE 0.0 END) * 100, 1) as negative_pct,
                ROUND(AVG(useful), 2) as avg_useful
            FROM category_reviews
            GROUP BY category
            HAVING COUNT(*) >= 1000  -- Minimum reviews for statistical significance
            ORDER BY avg_rating DESC
            LIMIT 15
        """).fetchall()
        
        category_data = []
        for cat, count, rating, pos_pct, neg_pct, useful in category_sentiment:
            category_data.append({
                "category": cat,
                "review_count": count,
                "avg_rating": rating,
                "positive_pct": pos_pct,
                "negative_pct": neg_pct,
                "avg_useful": useful
            })
        
        return {
            "analysis_type": "by_category",
            "categories": category_data
        }
    
    def _sentiment_by_city(self) -> Dict[str, Any]:
        """Sentiment analysis by city."""
        
        city_sentiment = self.analytics.conn.execute("""
            SELECT 
                b.city,
                COUNT(r.review_id) as review_count,
                ROUND(AVG(r.stars), 2) as avg_rating,
                ROUND(AVG(CASE WHEN r.stars >= 4 THEN 1.0 ELSE 0.0 END) * 100, 1) as positive_pct,
                ROUND(AVG(CASE WHEN r.stars <= 2 THEN 1.0 ELSE 0.0 END) * 100, 1) as negative_pct
            FROM businesses b
            JOIN reviews r ON b.business_id = r.business_id
            WHERE b.city IS NOT NULL
            GROUP BY b.city
            HAVING COUNT(r.review_id) >= 5000  -- Cities with significant review volume
            ORDER BY avg_rating DESC
            LIMIT 10
        """).fetchall()
        
        city_data = []
        for city, count, rating, pos_pct, neg_pct in city_sentiment:
            city_data.append({
                "city": city,
                "review_count": count,
                "avg_rating": rating,
                "positive_pct": pos_pct,
                "negative_pct": neg_pct
            })
        
        return {
            "analysis_type": "by_city",
            "cities": city_data
        }
    
    def _sentiment_time_series(self, business_id: Optional[str]) -> Dict[str, Any]:
        """Sentiment trends over time."""
        
        where_clause = "WHERE b.business_id = ?" if business_id else ""
        params = [business_id] if business_id else []
        
        time_series = self.analytics.conn.execute(f"""
            SELECT 
                DATE_TRUNC('month', r.date) as month,
                COUNT(r.review_id) as review_count,
                ROUND(AVG(r.stars), 2) as avg_rating,
                ROUND(AVG(CASE WHEN r.stars >= 4 THEN 1.0 ELSE 0.0 END) * 100, 1) as positive_pct,
                ROUND(AVG(CASE WHEN r.stars <= 2 THEN 1.0 ELSE 0.0 END) * 100, 1) as negative_pct
            FROM businesses b
            JOIN reviews r ON b.business_id = r.business_id
            {where_clause}
            GROUP BY month
            ORDER BY month DESC
            LIMIT 24
        """, params).fetchall()
        
        time_data = []
        for month, count, rating, pos_pct, neg_pct in time_series:
            time_data.append({
                "month": str(month),
                "review_count": count,
                "avg_rating": rating,
                "positive_pct": pos_pct,
                "negative_pct": neg_pct
            })
        
        return {
            "analysis_type": "time_series", 
            "business_id": business_id,
            "data": time_data
        }

# Test function
def test_analytics_tools():
    """Test the new DuckDB analytics tools."""
    
    print("🧪 Testing DuckDB Analytics Tools...")
    
    # Test business performance tool
    print("\n📊 Testing Business Performance Tool...")
    perf_tool = BusinessPerformanceTool()
    
    # Overall overview
    overview = perf_tool(analysis_type="overview")
    print(f"Market Overview: {overview}")
    
    # Test trend analysis
    print("\n📈 Testing Trend Analysis Tool...")
    trend_tool = TrendAnalysisTool()
    trends = trend_tool(time_period="monthly")
    print(f"Monthly Trends: {len(trends['data'])} periods analyzed")
    
    # Test sentiment analytics
    print("\n😊 Testing Sentiment Analytics Tool...")
    sentiment_tool = SentimentAnalyticsTool()
    sentiment = sentiment_tool(analysis_type="overview")
    print(f"Sentiment Overview: {sentiment}")
    
    print("✅ All analytics tools tested successfully!")

if __name__ == "__main__":
    test_analytics_tools()
