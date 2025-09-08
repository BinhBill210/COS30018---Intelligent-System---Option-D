#!/usr/bin/env python3
"""
Data Loading Script for DuckDB Analytics Backend
===============================================

This script loads the processed Yelp data into DuckDB for high-performance analytics.
It handles both business and review data with proper schema validation.
"""

from duckdb_integration import DuckDBAnalytics
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Load data into DuckDB analytics database."""
    
    print("🚀 Loading Yelp Data into DuckDB Analytics Database")
    print("=" * 60)
    
    # Initialize DuckDB analytics
    analytics = DuckDBAnalytics("business_analytics.duckdb")
    
    try:
        # Check if data files exist
        business_file = "data/processed/business_cleaned.parquet"
        review_file = "data/processed/review_cleaned.parquet"
        
        # Fallback to CSV if Parquet not available
        if not Path(business_file).exists():
            business_file = "data/processed/business_cleaned.csv"
        if not Path(review_file).exists():
            review_file = "data/processed/review_cleaned.csv"
        
        print(f"📁 Business data: {business_file}")
        print(f"📁 Review data: {review_file}")
        
        # Load data
        analytics.load_data_from_files(business_file, review_file)
        
        # Get final database info
        info = analytics.get_database_info()
        
        print("\n" + "=" * 60)
        print("📊 LOADING COMPLETE - DATABASE SUMMARY")
        print("=" * 60)
        print(f"📍 Database: {info['database_path']}")
        print(f"🏢 Businesses: {info['statistics']['businesses']:,}")
        print(f"📝 Reviews: {info['statistics']['reviews']:,}")
        print(f"📅 Date Range: {info['statistics']['earliest_review']} to {info['statistics']['latest_review']}")
        print(f"🔧 DuckDB Version: {info['duckdb_version']}")
        
        # Test some basic analytics
        print("\n🧪 Testing Analytics Capabilities...")
        
        # Top cities by business count
        top_cities = analytics.conn.execute("""
            SELECT city, COUNT(*) as business_count 
            FROM businesses 
            WHERE city IS NOT NULL 
            GROUP BY city 
            ORDER BY business_count DESC 
            LIMIT 5
        """).fetchall()
        
        print("\n🏙️ Top 5 Cities by Business Count:")
        for city, count in top_cities:
            print(f"   {city}: {count:,} businesses")
        
        # Average rating by category (using explode)
        top_categories = analytics.conn.execute("""
            WITH category_exploded AS (
                SELECT 
                    business_id,
                    stars,
                    UNNEST(category_list) as category
                FROM businesses 
                WHERE category_list IS NOT NULL AND len(category_list) > 0
            )
            SELECT 
                category,
                COUNT(*) as business_count,
                ROUND(AVG(stars), 2) as avg_rating
            FROM category_exploded
            GROUP BY category 
            HAVING COUNT(*) >= 100
            ORDER BY business_count DESC 
            LIMIT 5
        """).fetchall()
        
        print("\n📂 Top 5 Categories:")
        for category, count, rating in top_categories:
            print(f"   {category}: {count:,} businesses (⭐ {rating})")
        
        # Recent review trends
        recent_trends = analytics.conn.execute("""
            SELECT 
                DATE_TRUNC('month', date) as month,
                COUNT(*) as review_count,
                ROUND(AVG(stars), 2) as avg_rating
            FROM reviews 
            WHERE date >= '2023-01-01'
            GROUP BY month 
            ORDER BY month DESC 
            LIMIT 6
        """).fetchall()
        
        print("\n📈 Recent Review Trends (Last 6 Months):")
        for month, count, rating in recent_trends:
            print(f"   {month}: {count:,} reviews (⭐ {rating})")
        
        print("\n✅ DuckDB Analytics Database Ready!")
        print("🔥 You can now run complex analytical queries at lightning speed!")
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise
    finally:
        analytics.close()

if __name__ == "__main__":
    main()
