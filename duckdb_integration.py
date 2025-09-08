
#!/usr/bin/env python3
"""
DuckDB Integration for Business Intelligence Agent
=================================================

This module provides a high-performance analytical backend using DuckDB
for the LLM-powered Business Improvement Agent project.

Key Features:
- Fast analytical queries on large datasets
- Seamless integration with existing ChromaDB vector search
- Business intelligence and trend analysis
- Optimized for Yelp review and business data

Architecture:
- ChromaDB: Semantic search and similarity matching  
- DuckDB: Analytics, aggregations, and complex queries
- Hybrid approach: Best of both worlds

Author: COS30018 Intelligent Systems Project
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuckDBAnalytics:
    """
    High-performance analytics engine using DuckDB for business intelligence.
    
    This class provides:
    1. Fast data loading and storage
    2. Complex analytical queries
    3. Business intelligence functions
    4. Integration with existing data pipeline
    """
    
    def __init__(self, db_path: str = "business_analytics.duckdb"):
        """
        Initialize DuckDB connection and setup analytics database.
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
        # Configure DuckDB for optimal performance
        self.conn.execute("SET memory_limit='2GB'")
        self.conn.execute("SET threads=4")
        
        logger.info(f"✅ DuckDB Analytics initialized: {db_path}")
        logger.info(f"   Memory limit: 2GB, Threads: 4")
        
        # Track table status
        self.tables_created = False
        
    def create_tables(self):
        """Create optimized table schemas for business and review data."""
        
        logger.info("🔧 Creating optimized table schemas...")
        
        # Business table with optimized data types
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS businesses (
                business_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                address VARCHAR,
                city VARCHAR,
                state VARCHAR,
                postal_code VARCHAR,
                latitude DOUBLE,
                longitude DOUBLE,
                stars DECIMAL(2,1),
                review_count INTEGER,
                is_open BOOLEAN,
                categories VARCHAR,
                attributes VARCHAR,
                hours VARCHAR,
                -- Derived fields for analytics
                category_list VARCHAR[],  -- Parsed categories
                price_range INTEGER,      -- Extracted from attributes
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Reviews table with analytics optimizations
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                review_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                business_id VARCHAR NOT NULL,
                stars DECIMAL(2,1) NOT NULL,
                useful INTEGER DEFAULT 0,
                funny INTEGER DEFAULT 0,
                cool INTEGER DEFAULT 0,
                text TEXT NOT NULL,
                date DATE,
                -- Derived analytics fields
                text_length INTEGER,
                word_count INTEGER,
                sentiment_score DOUBLE,     -- Will be populated by ML
                aspect_scores JSON,         -- Aspect-based sentiment
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Foreign key constraint
                FOREIGN KEY (business_id) REFERENCES businesses(business_id)
            )
        """)
        
        # Create indexes for performance
        logger.info("📊 Creating performance indexes...")
        
        # Business indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_business_city ON businesses(city)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_business_stars ON businesses(stars)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_business_categories ON businesses(categories)")
        
        # Review indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_review_business_id ON reviews(business_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_review_date ON reviews(date)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_review_stars ON reviews(stars)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_review_user_id ON reviews(user_id)")
        
        # Composite indexes for common queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_review_business_date ON reviews(business_id, date)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_review_business_stars ON reviews(business_id, stars)")
        
        self.tables_created = True
        logger.info("✅ Database schema created successfully!")
        
    def load_data_from_files(self, 
                           business_file: str = "data/processed/business_cleaned.parquet",
                           review_file: str = "data/processed/review_cleaned.parquet"):
        """
        Load data from existing processed files into DuckDB.
        
        Args:
            business_file: Path to business data file
            review_file: Path to review data file
        """
        
        # Always recreate tables for fresh data load
        logger.info("🔄 Recreating tables for fresh data load...")
        self.conn.execute("DROP TABLE IF EXISTS reviews")
        self.conn.execute("DROP TABLE IF EXISTS businesses")
        self.create_tables()
        
        logger.info("📥 Loading data into DuckDB...")
        
        # Load business data
        business_path = Path(business_file)
        if business_path.exists():
            if business_path.suffix == '.parquet':
                logger.info(f"Loading businesses from Parquet: {business_file}")
                # DuckDB can read Parquet directly - very efficient!
                self.conn.execute(f"""
                    INSERT INTO businesses 
                    SELECT 
                        business_id,
                        name,
                        address,
                        city,
                        state,
                        postal_code,
                        latitude,
                        longitude,
                        stars,
                        review_count,
                        CASE WHEN is_open = 1 THEN true ELSE false END as is_open,
                        categories,
                        attributes,
                        hours,
                        -- Parse categories into array
                        CASE 
                            WHEN categories IS NOT NULL THEN string_split(REPLACE(categories, ', ', ','), ',')
                            ELSE NULL 
                        END as category_list,
                        -- Extract price range from attributes (struct format)
                        TRY_CAST(attributes.RestaurantsPriceRange2 AS INTEGER) as price_range,
                        CURRENT_TIMESTAMP as created_at
                    FROM read_parquet('{business_file}')
                """)
            else:
                # CSV fallback
                self.conn.execute(f"""
                    INSERT INTO businesses 
                    SELECT 
                        business_id,
                        name,
                        address,
                        city,
                        state,
                        postal_code,
                        latitude,
                        longitude,
                        stars,
                        review_count,
                        CASE WHEN is_open = 1 THEN true ELSE false END as is_open,
                        categories,
                        attributes,
                        hours,
                        CASE 
                            WHEN categories IS NOT NULL THEN string_split(REPLACE(categories, ', ', ','), ',')
                            ELSE NULL 
                        END as category_list,
                        CASE 
                            WHEN attributes LIKE '%RestaurantsPriceRange2%' THEN 
                                CAST(regexp_extract(attributes, 'RestaurantsPriceRange2.*?([1-4])', 1) AS INTEGER)
                            ELSE NULL 
                        END as price_range,
                        CURRENT_TIMESTAMP as created_at
                    FROM read_csv_auto('{business_file}')
                """)
            
            business_count = self.conn.execute("SELECT COUNT(*) FROM businesses").fetchone()[0]
            logger.info(f"✅ Loaded {business_count:,} businesses")
        
        # Load review data
        review_path = Path(review_file)
        if review_path.exists():
            if review_path.suffix == '.parquet':
                logger.info(f"Loading reviews from Parquet: {review_file}")
                self.conn.execute(f"""
                    INSERT INTO reviews 
                    SELECT 
                        review_id,
                        user_id,
                        business_id,
                        stars,
                        COALESCE(useful, 0) as useful,
                        COALESCE(funny, 0) as funny,
                        COALESCE(cool, 0) as cool,
                        text,
                        CAST(date AS DATE) as date,
                        -- Calculate derived fields
                        length(text) as text_length,
                        length(text) - length(replace(text, ' ', '')) + 1 as word_count,
                        NULL as sentiment_score,  -- Will be populated later
                        NULL as aspect_scores,    -- Will be populated later
                        CURRENT_TIMESTAMP as created_at
                    FROM read_parquet('{review_file}')
                    WHERE text IS NOT NULL AND text != ''
                """)
            else:
                # CSV fallback with proper handling of empty reviews
                self.conn.execute(f"""
                    INSERT INTO reviews 
                    SELECT 
                        review_id,
                        user_id,
                        business_id,
                        stars,
                        COALESCE(useful, 0) as useful,
                        COALESCE(funny, 0) as funny,
                        COALESCE(cool, 0) as cool,
                        text,
                        CAST(date AS DATE) as date,
                        length(text) as text_length,
                        length(text) - length(replace(text, ' ', '')) + 1 as word_count,
                        NULL as sentiment_score,
                        NULL as aspect_scores,
                        CURRENT_TIMESTAMP as created_at
                    FROM read_csv_auto('{review_file}')
                    WHERE text IS NOT NULL AND text != ''
                """)
            
            review_count = self.conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
            logger.info(f"✅ Loaded {review_count:,} reviews")
        
        # Create summary statistics
        self._create_summary_stats()
        
    def _create_summary_stats(self):
        """Create summary statistics for quick access."""
        logger.info("📊 Generating summary statistics...")
        
        stats = self.conn.execute("""
            SELECT 
                'Overall' as category,
                COUNT(DISTINCT b.business_id) as total_businesses,
                COUNT(DISTINCT r.review_id) as total_reviews,
                COUNT(DISTINCT r.user_id) as total_users,
                AVG(r.stars) as avg_rating,
                MIN(r.date) as earliest_review,
                MAX(r.date) as latest_review
            FROM businesses b
            LEFT JOIN reviews r ON b.business_id = r.business_id
        """).fetchone()
        
        logger.info(f"📈 Summary Statistics:")
        logger.info(f"   Businesses: {stats[1]:,}")
        logger.info(f"   Reviews: {stats[2]:,}")
        logger.info(f"   Users: {stats[3]:,}")
        logger.info(f"   Avg Rating: {stats[4]:.2f}")
        logger.info(f"   Date Range: {stats[5]} to {stats[6]}")

    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the database."""
        
        # Table information
        tables = self.conn.execute("""
            SELECT table_name, 
                   estimated_size as row_count
            FROM duckdb_tables() 
            WHERE database_name = 'main'
        """).fetchall()
        
        # Basic statistics
        stats = {}
        if self.tables_created:
            business_count = self.conn.execute("SELECT COUNT(*) FROM businesses").fetchone()[0]
            review_count = self.conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
            
            if review_count > 0:
                date_range = self.conn.execute("""
                    SELECT MIN(date) as earliest, MAX(date) as latest 
                    FROM reviews 
                    WHERE date IS NOT NULL
                """).fetchone()
                
                stats = {
                    "businesses": business_count,
                    "reviews": review_count,
                    "earliest_review": str(date_range[0]) if date_range[0] else None,
                    "latest_review": str(date_range[1]) if date_range[1] else None
                }
        
        return {
            "database_path": self.db_path,
            "tables": dict(tables),
            "statistics": stats,
            "duckdb_version": duckdb.__version__
        }

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("🔒 DuckDB connection closed")

# Quick test function
def test_duckdb_setup():
    """Test the DuckDB setup with sample data."""
    
    print("🧪 Testing DuckDB Integration...")
    
    # Initialize
    analytics = DuckDBAnalytics("test_analytics.duckdb")
    
    # Create tables
    analytics.create_tables()
    
    # Test basic functionality
    analytics.conn.execute("""
        INSERT INTO businesses VALUES 
        ('test-123', 'Test Restaurant', '123 Main St', 'TestCity', 'TS', '12345', 
         40.7128, -74.0060, 4.5, 100, true, 'Restaurants, Italian', '{}', '{}', 
         ARRAY['Restaurants', 'Italian'], 2, CURRENT_TIMESTAMP)
    """)
    
    analytics.conn.execute("""
        INSERT INTO reviews VALUES 
        ('rev-123', 'user-123', 'test-123', 5.0, 1, 0, 1, 'Great food and service!', 
         '2024-01-15', 23, 4, NULL, NULL, CURRENT_TIMESTAMP)
    """)
    
    # Test query
    result = analytics.conn.execute("""
        SELECT b.name, AVG(r.stars) as avg_rating, COUNT(r.review_id) as review_count
        FROM businesses b
        JOIN reviews r ON b.business_id = r.business_id
        GROUP BY b.name
    """).fetchall()
    
    print(f"✅ Test query result: {result}")
    
    # Get info
    info = analytics.get_database_info()
    print(f"📊 Database info: {info}")
    
    # Cleanup
    analytics.close()
    Path("test_analytics.duckdb").unlink(missing_ok=True)
    
    print("🎉 DuckDB integration test completed successfully!")

if __name__ == "__main__":
    test_duckdb_setup()
