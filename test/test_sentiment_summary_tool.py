from dotenv import load_dotenv
load_dotenv()
from tools.sentiment_summary_tool import SentimentSummaryTool

def test_sentiment_summary_tool():
    print("Testing SentimentSummaryTool...")
    sentiment_tool = SentimentSummaryTool()
    sample_reviews = ["Great service!", "Terrible experience", "It was okay"]
    result = sentiment_tool(sample_reviews)
    print(f"Input reviews: {sample_reviews}")
    print(f"Output: {result}")
    return result is not None and 'total_reviews' in result and result['total_reviews'] > 0

if __name__ == "__main__":
    test_sentiment_summary_tool()
