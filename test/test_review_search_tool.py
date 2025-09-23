from dotenv import load_dotenv
load_dotenv()
from tools.review_search_tool import ReviewSearchTool
import os

def test_review_search_tool():
    print("Testing ReviewSearchTool...")
    host = os.environ.get("CHROMA_HOST", "localhost")
    print(f"Using CHROMA_HOST: {host}")
    search_tool = ReviewSearchTool(host=host)
    query = "Terrible host service, 45 minute wait, seated other parties of the same size that arrived after us. Someone needs to manage this location, been a customer for a long time, likely done after this visit. Poor management."
    business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"
    results = search_tool(query, k=10, business_id=business_id)
    print(f"Query: {query}")
    print(f"Found {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Review ID: {result.get('review_id', '')}")
        print(f"Business ID: {result.get('business_id', '')}")
        print(f"Stars: {result.get('stars', '')}")
        print(f"Date: {result.get('date', '')}")
        print(f"Score: {result.get('score', '')}")
        print(f"Text: {result.get('text', '')}")
        print()
    return len(results) > 0

if __name__ == "__main__":
    test_review_search_tool()
