#!/usr/bin/env python3
"""
migrate_to_chromadb.py

Migration script to convert existing Yelp review data to ChromaDB vector store.
This script replaces the custom FAISS implementation with ChromaDB + LangChain.

Usage:
    python migrate_to_chromadb.py [--data-path PATH] [--chroma-path PATH] [--chunk-size SIZE]

Features:
- Migrates from CSV/Parquet to ChromaDB
- Maintains metadata compatibility
- Progress tracking
- Error handling and recovery
- Validates migration success
"""

import argparse
import sys
import traceback
from pathlib import Path
import pandas as pd
from chromadb_integration import ChromaDBVectorStore

def validate_data_file(data_path: str) -> bool:
    """Validate that the data file exists and has the expected format"""
    
    if not Path(data_path).exists():
        print(f"❌ Error: Data file not found: {data_path}")
        return False
    
    try:
        # Try loading a few rows to validate format
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path, nrows=5)
        else:
            df = pd.read_csv(data_path, nrows=5)
        
        # Check for required columns
        required_columns = ['text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        print(f"✅ Data validation passed: {len(df.columns)} columns, sample loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error validating data file: {e}")
        return False

def migrate_to_chromadb(data_path: str, 
                       chroma_path: str = "./chroma_db",
                       chunk_size: int = 500,
                       chunk_overlap: int = 50,
                       collection_name: str = "yelp_reviews") -> bool:
    """
    Main migration function
    
    Args:
        data_path: Path to the source data (CSV or Parquet)
        chroma_path: Directory for ChromaDB storage
        chunk_size: Size of text chunks for vector storage
        chunk_overlap: Overlap between chunks
        collection_name: Name of the ChromaDB collection
        
    Returns:
        bool: True if migration successful, False otherwise
    """
    
    print("🚀 Starting migration to ChromaDB...")
    print(f"   Source: {data_path}")
    print(f"   Target: {chroma_path}")
    print(f"   Collection: {collection_name}")
    print(f"   Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    print("-" * 60)
    
    try:
        # Step 1: Validate input data
        print("📋 Step 1: Validating data file...")
        if not validate_data_file(data_path):
            return False
        
        # Step 2: Initialize ChromaDB vector store
        print("🔧 Step 2: Initializing ChromaDB vector store...")
        vector_store = ChromaDBVectorStore(
            collection_name=collection_name,
            persist_directory=chroma_path,
            embedding_model="all-MiniLM-L6-v2"  # Same as your existing setup
        )
        
        # Step 3: Check if collection already exists
        info = vector_store.get_collection_info()
        if "error" not in info and info.get("count", 0) > 0:
            print(f"⚠️  Collection '{collection_name}' already exists with {info['count']} documents")
            response = input("Do you want to continue and add more data? (y/N): ").strip().lower()
            if response != 'y':
                print("Migration cancelled by user")
                return False
        
        # Step 4: Load and index the data
        print("📚 Step 3: Loading and indexing reviews...")
        total_chunks = vector_store.load_and_index_reviews(
            data_path=data_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Step 5: Validate migration
        print("✅ Step 4: Validating migration...")
        final_info = vector_store.get_collection_info()
        
        if "error" in final_info:
            print(f"❌ Migration validation failed: {final_info['error']}")
            return False
        
        print(f"✅ Migration successful!")
        print(f"   📊 Total chunks indexed: {final_info['count']}")
        print(f"   🗂️  Collection name: {final_info['name']}")
        print(f"   📁 Storage location: {chroma_path}")
        
        # Step 6: Test search functionality
        print("🔍 Step 5: Testing search functionality...")
        test_results = vector_store.similarity_search("great food", k=3)
        
        if test_results:
            print(f"   ✅ Search test passed: Found {len(test_results)} results")
            sample_result = test_results[0]
            print(f"   📝 Sample result: {sample_result[0].page_content[:100]}...")
        else:
            print("   ⚠️  Search test returned no results (this might be normal for small datasets)")
        
        print("\n" + "="*60)
        print("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📋 Next steps:")
        print("1. Test the migration:")
        print(f"   python -c \"from tools_chromadb import ReviewSearchTool; tool=ReviewSearchTool('{chroma_path}'); print(tool('great food'))\"")
        print("\n2. Run the ChromaDB demo:")
        print("   python demo2_chromadb.py")
        print("\n3. Use your updated tools:")
        print("   python test_basic_components.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed with error: {e}")
        print("\n🔍 Full error traceback:")
        traceback.print_exc()
        return False

def main():
    """Main entry point with command line interface"""
    
    parser = argparse.ArgumentParser(
        description="Migrate Yelp review data to ChromaDB vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic migration with default settings
  python migrate_to_chromadb.py
  
  # Custom data path
  python migrate_to_chromadb.py --data-path data/processed/review_cleaned.csv
  
  # Custom ChromaDB location and chunk size
  python migrate_to_chromadb.py --chroma-path ./my_chroma_db --chunk-size 300
        """
    )
    
    parser.add_argument(
        "--data-path",
        default="data/processed/review_cleaned.parquet",
        help="Path to the review data file (CSV or Parquet). Default: data/processed/review_cleaned.parquet"
    )
    
    parser.add_argument(
        "--chroma-path", 
        default="./chroma_db",
        help="Directory for ChromaDB persistent storage. Default: ./chroma_db"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Size of text chunks for vector indexing. Default: 500"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between text chunks. Default: 50"
    )
    
    parser.add_argument(
        "--collection-name",
        default="yelp_reviews",
        help="Name of the ChromaDB collection. Default: yelp_reviews"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force migration without confirmation prompts"
    )
    
    args = parser.parse_args()
    
    # Display banner
    print("=" * 60)
    print("🔄 YELP REVIEW DATA MIGRATION TO CHROMADB")
    print("=" * 60)
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"❌ Error: Data file not found: {args.data_path}")
        print("\n💡 Available data files:")
        
        # Look for common data files
        data_dir = Path("data/processed")
        if data_dir.exists():
            for file in data_dir.glob("*.parquet"):
                print(f"   - {file}")
            for file in data_dir.glob("*.csv"):
                print(f"   - {file}")
        else:
            print("   No data/processed directory found")
            
        print(f"\n📝 Usage: python {sys.argv[0]} --data-path YOUR_DATA_FILE")
        sys.exit(1)
    
    # Confirm migration if not forced
    if not args.force:
        print(f"\n📋 Migration Configuration:")
        print(f"   Data source: {args.data_path}")
        print(f"   ChromaDB path: {args.chroma_path}")
        print(f"   Chunk size: {args.chunk_size}")
        print(f"   Collection: {args.collection_name}")
        
        response = input(f"\n❓ Proceed with migration? (y/N): ").strip().lower()
        if response != 'y':
            print("Migration cancelled by user")
            sys.exit(0)
    
    # Run migration
    success = migrate_to_chromadb(
        data_path=args.data_path,
        chroma_path=args.chroma_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        collection_name=args.collection_name
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
