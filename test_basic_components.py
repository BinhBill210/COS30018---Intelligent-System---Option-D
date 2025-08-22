# test_basic_components.py
from tools import ReviewSearchTool, SentimentSummaryTool, DataSummaryTool

def test_basic_functionality():
    print("Testing basic components...")
    
    # Test search tool
    try:
        search_tool = ReviewSearchTool("index_demo")
        results = search_tool("service quality", k=3)
        print("✓ Search tool works")
        print(f"Found {len(results)} results")
    except Exception as e:
        print(f"✗ Search tool failed: {e}")
        return False
    
    # Test sentiment tool
    try:
        sentiment_tool = SentimentSummaryTool()
        sample_reviews = ["Great service!", "Terrible experience", "It was okay"]
        sentiment = sentiment_tool(sample_reviews)
        print("✓ Sentiment tool works")
        print(f"Sentiment: {sentiment}")
    except Exception as e:
        print(f"✗ Sentiment tool failed: {e}")
        return False
    
    # Test data tool
    try:
        data_tool = DataSummaryTool("data/processed/review_cleaned.csv")
        summary = data_tool()
        print("✓ Data tool works")
        print(f"Summary: {summary}")
    except Exception as e:
        print(f"✗ Data tool failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\nAll basic components work! The issue is likely with the LLM.")
    else:
        print("\nSome components failed. Let's fix those first.")