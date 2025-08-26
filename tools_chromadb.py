# tools_chromadb.py - Updated to use ChromaDB instead of FAISS
from typing import List, Dict, Any
import pandas as pd
from transformers import pipeline
import numpy as np
from pathlib import Path
from chromadb_integration import ChromaDBVectorStore

class ReviewSearchTool:
    """Updated ReviewSearchTool that uses ChromaDB instead of FAISS"""
    
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        """
        Initialize with ChromaDB instead of FAISS index
        
        Args:
            chroma_db_path: Path to ChromaDB persistent storage
        """
        self.vector_store = ChromaDBVectorStore(
            collection_name="yelp_reviews",
            persist_directory=chroma_db_path
        )
        
        # Check if data exists
        info = self.vector_store.get_collection_info()
        if "error" in info or info.get("count", 0) == 0:
            print("Warning: No data found in ChromaDB. Run migration first: python migrate_to_chromadb.py")
    
    def __call__(self, query: str, k: int = 5):
        """
        Search for similar reviews - same interface as before
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dictionaries with review data (same format as FAISS version)
        """
        try:
            # Use ChromaDB vector search
            search_results = self.vector_store.similarity_search(query, k=k)
            
            # Format results to match original FAISS format
            results = []
            for doc, score in search_results:
                metadata = doc.metadata
                results.append({
                    "review_id": metadata.get("review_id", ""),
                    "text": doc.page_content,  # ChromaDB stores text in page_content
                    "stars": metadata.get("stars", ""),
                    "business_id": metadata.get("business_id", ""),
                    "date": metadata.get("date", ""),
                    "score": float(score)
                })
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []

class SentimentSummaryTool:
    """Same as before - no changes needed"""
    
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    def __call__(self, reviews: List[str]):
        # Analyze sentiment for each review
        sentiments = self.sentiment_analyzer(reviews)
        
        # Count sentiments
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0}
        for sentiment in sentiments:
            label = sentiment["label"]
            sentiment_counts[label] += 1
        
        # Calculate percentages
        total = len(sentiments)
        positive_pct = (sentiment_counts["POSITIVE"] / total) * 100 if total > 0 else 0
        negative_pct = (sentiment_counts["NEGATIVE"] / total) * 100 if total > 0 else 0
        
        return {
            "total_reviews": total,
            "positive_percentage": round(positive_pct, 2),
            "negative_percentage": round(negative_pct, 2),
            "sample_sentiments": sentiments[:3]  # Return first 3 as examples
        }

class DataSummaryTool:
    """Same as before - no changes needed"""
    
    def __init__(self, data_path: str):
        self.df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    
    def __call__(self, business_id: str = None):
        if business_id:
            filtered_df = self.df[self.df["business_id"] == business_id]
        else:
            filtered_df = self.df
        
        # Calculate basic statistics
        avg_stars = filtered_df["stars"].mean()
        total_reviews = len(filtered_df)
        useful_avg = filtered_df["useful"].mean()
        funny_avg = filtered_df["funny"].mean()
        cool_avg = filtered_df["cool"].mean()
        
        return {
            "business_id": business_id or "all businesses",
            "total_reviews": total_reviews,
            "average_stars": round(avg_stars, 2),
            "average_useful": round(useful_avg, 2),
            "average_funny": round(funny_avg, 2),
            "average_cool": round(cool_avg, 2)
        }

# Backward compatibility: create aliases so existing code doesn't break
ChromaReviewSearchTool = ReviewSearchTool

def test_chromadb_tools():
    """Test the ChromaDB tools to ensure they work correctly"""
    print("Testing ChromaDB Tools...")
    
    try:
        # Test ReviewSearchTool
        print("\n1. Testing ReviewSearchTool...")
        search_tool = ReviewSearchTool()
        results = search_tool("great food", k=3)
        print(f"   Found {len(results)} results")
        if results:
            print(f"   Sample result: {results[0]['text'][:100]}...")
        
        # Test SentimentSummaryTool
        print("\n2. Testing SentimentSummaryTool...")
        sentiment_tool = SentimentSummaryTool()
        test_reviews = ["Great food!", "Terrible service", "Amazing experience"]
        sentiment_result = sentiment_tool(test_reviews)
        print(f"   Sentiment analysis: {sentiment_result}")
        
        # Test DataSummaryTool
        print("\n3. Testing DataSummaryTool...")
        data_tool = DataSummaryTool("data/processed/review_cleaned.parquet")
        summary = data_tool()
        print(f"   Data summary: {summary}")
        
        print("\n✅ All tools working correctly!")
        
    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chromadb_tools()
