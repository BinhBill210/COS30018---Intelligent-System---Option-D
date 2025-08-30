from tools.review_search_tool import ReviewSearchTool

def test_review_search_tool():
    print("Testing ReviewSearchTool...")
    search_tool = ReviewSearchTool("./chroma_db")
    query = "service quality"
    results = search_tool(query, k=10)
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
