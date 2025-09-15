#!/usr/bin/env python3
"""
Database Setup Script
Creates the foundation DuckDB database that any tool can use
"""

import pandas as pd
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import get_db_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_schema():
    """Create database schema from SQL file"""
    logger.info("Creating database schema...")
    
    schema_file = Path(__file__).parent.parent / "schema" / "base_schema.sql"
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    
    # Read and execute schema
    with open(schema_file, 'r') as f:
        schema_sql = f.read()
    
    db_manager = get_db_manager()
    
    # Split by semicolon and execute each statement
    statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip() and not stmt.strip().startswith('--')]
    
    for stmt in statements:
        try:
            db_manager.execute_update(stmt)
        except Exception as e:
            logger.warning(f"Schema statement failed (may be normal): {e}")
    
    logger.info("Database schema created successfully")

def load_business_data():
    """Load business data into the database"""
    logger.info("Loading business data...")
    
    db_manager = get_db_manager()
    
    # Check if data already exists
    existing_count = db_manager.execute_query("SELECT COUNT(*) as count FROM businesses")
    if not existing_count.empty and existing_count.iloc[0, 0] > 0:
        logger.info(f"Business data already exists ({existing_count.iloc[0, 0]:,} records)")
        return True
    
    # Find business data file
    business_file = None
    for possible_path in [
        "data/processed/business_cleaned.parquet",
        "data/processed/business_cleaned.csv"
    ]:
        if Path(possible_path).exists():
            business_file = possible_path
            break
    
    if not business_file:
        logger.warning("No business data file found. Database created but empty.")
        logger.info("Expected: data/processed/business_cleaned.parquet or .csv")
        return False
    
    logger.info(f"Loading business data from {business_file}")
    
    # Load data
    if business_file.endswith('.parquet'):
        df = pd.read_parquet(business_file)
    else:
        df = pd.read_csv(business_file)
    
    # Clean data for database
    df = df.fillna({
        'address': '',
        'city': '',
        'state': '',
        'postal_code': '',
        'latitude': 0.0,
        'longitude': 0.0,
        'stars': 0.0,
        'review_count': 0,
        'is_open': True,
        'attributes': '{}',
        'categories': '',
        'hours': '{}'
    })
    
    # Add metadata columns
    from datetime import datetime
    df['created_at'] = datetime.now()
    df['updated_at'] = datetime.now()
    
    # Insert data
    logger.info(f"Inserting {len(df):,} business records...")
    db_manager.batch_insert("businesses", df)
    
    # Verify insertion
    final_count = db_manager.execute_query("SELECT COUNT(*) as count FROM businesses")
    logger.info(f"Successfully loaded {final_count.iloc[0, 0]:,} business records")
    
    return True

def load_review_data():
    """Load review data into the database (optional)"""
    logger.info("Checking for review data...")
    
    # Find review data file
    review_file = None
    for possible_path in [
        "data/processed/review_cleaned.parquet",
        "data/processed/review_cleaned.csv"
    ]:
        if Path(possible_path).exists():
            review_file = possible_path
            break
    
    if not review_file:
        logger.info("No review data file found. Skipping review data loading.")
        return True
    
    db_manager = get_db_manager()
    
    # Check if review data already exists
    existing_count = db_manager.execute_query("SELECT COUNT(*) as count FROM reviews")
    if not existing_count.empty and existing_count.iloc[0, 0] > 0:
        logger.info(f"Review data already exists ({existing_count.iloc[0, 0]:,} records)")
        return True
    
    logger.info(f"Loading review data from {review_file}")
    
    # Load data in chunks for large files
    chunk_size = 10000
    total_inserted = 0
    
    try:
        if review_file.endswith('.parquet'):
            df = pd.read_parquet(review_file)
            # Process in chunks
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size].copy()
                chunk = prepare_review_chunk(chunk)
                db_manager.batch_insert("reviews", chunk)
                total_inserted += len(chunk)
                logger.info(f"Inserted {total_inserted:,} review records...")
        else:
            # For CSV, read in chunks
            for chunk in pd.read_csv(review_file, chunksize=chunk_size):
                chunk = prepare_review_chunk(chunk)
                db_manager.batch_insert("reviews", chunk)
                total_inserted += len(chunk)
                logger.info(f"Inserted {total_inserted:,} review records...")
    
        logger.info(f"Successfully loaded {total_inserted:,} review records")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load review data: {e}")
        return False

def prepare_review_chunk(df):
    """Prepare review chunk for database insertion"""
    # Clean data
    df = df.fillna({
        'user_id': '',
        'stars': 0.0,
        'useful': 0,
        'funny': 0,
        'cool': 0,
        'text': ''
    })
    
    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Add metadata
    from datetime import datetime
    df['created_at'] = datetime.now()
    df['updated_at'] = datetime.now()
    
    return df

def run_setup():
    """Run complete database setup"""
    logger.info("=" * 60)
    logger.info("🚀 Setting up DuckDB Database Foundation")
    logger.info("=" * 60)
    
    try:
        # Step 1: Create schema
        create_schema()
        
        # Step 2: Load business data
        business_success = load_business_data()
        
        # Step 3: Load review data (optional)
        review_success = load_review_data()
        
        # Step 4: Test the setup
        test_setup()
        
        logger.info("=" * 60)
        logger.info("✅ Database setup completed successfully!")
        logger.info("=" * 60)
        
        if business_success:
            logger.info("🏢 Business data loaded - BusinessSearchTool ready")
        if review_success:
            logger.info("📝 Review data loaded - Ready for future sentiment/review tools")
        
        logger.info("\nNext steps:")
        logger.info("1. Test BusinessSearchTool: python -c \"from tools.business_search_tool import BusinessSearchTool; tool = BusinessSearchTool(); print(tool.get_business_id('Starbucks'))\"")
        logger.info("2. Run the agent: python run_langchain_chat.py")
        logger.info("3. Try Streamlit: streamlit run streamlit_agent.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

def test_setup():
    """Test the database setup"""
    logger.info("Testing database setup...")
    
    db_manager = get_db_manager()
    
    # Test businesses table
    business_count = db_manager.execute_query("SELECT COUNT(*) as count FROM businesses")
    logger.info(f"Businesses table: {business_count.iloc[0, 0]:,} records")
    
    # Test reviews table
    try:
        review_count = db_manager.execute_query("SELECT COUNT(*) as count FROM reviews")
        logger.info(f"Reviews table: {review_count.iloc[0, 0]:,} records")
    except:
        logger.info("Reviews table: not populated (optional)")
    
    # Test sample queries
    try:
        sample_business = db_manager.execute_query("SELECT name, city, stars FROM businesses LIMIT 1")
        if not sample_business.empty:
            logger.info(f"Sample business: {sample_business.iloc[0]['name']} in {sample_business.iloc[0]['city']}")
        
        # Test BusinessSearchTool if available
        try:
            from tools.business_search_tool import BusinessSearchTool
            tool = BusinessSearchTool()
            if tool.db_available:
                logger.info("✅ BusinessSearchTool can access DuckDB")
            else:
                logger.warning("⚠️ BusinessSearchTool cannot access DuckDB")
        except Exception as e:
            logger.warning(f"BusinessSearchTool test failed: {e}")
            
    except Exception as e:
        logger.error(f"Setup test failed: {e}")

def main():
    """Main setup function"""
    return run_setup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
