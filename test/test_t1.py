import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path[:3]}")  # Check first 3 paths

from tools.hybrid_retrieval_tool import HybridRetrieve


def demo_t1_hybrid_retrieve():
    """Demonstrate T1 HybridRetrieve tool usage with various query patterns."""
    
    # Initialize the tool
    hybrid_retrieve = HybridRetrieve(
        data_path="data/processed/review_cleaned.csv",
        chroma_path="./chroma_db"
    )
    
    # Example 1: Basic query for service issues
    print("=== Example 1: Service Quality Issues ===")
    results1 = hybrid_retrieve(
        business_id="XQfwVwDr-v0ZS3_CbbE5Xw",
        query="slow service wait time",
        top_k=5,
        filters={"stars": [1, 2]}  # Focus on negative reviews
    )
    
    print(f"Found {len(results1['hits'])} hits")
    for i, evidence in enumerate(results1['evidence'][:3]):
        print(f"Evidence {i+1}: {evidence['quote']} (★{evidence['stars']})")
    
    # Example 2: Date-filtered refund policy search
    print("\n=== Example 2: Recent Refund Policy Issues ===")
    results2 = hybrid_retrieve(
        business_id="XQfwVwDr-v0ZS3_CbbE5Xw",
        query="refund policy return money",
        top_k=10,
        filters={
            "date_from": "2022-01-01",
            "date_to": "2023-12-31",
            "stars": [1, 5]  # All ratings
        }
    )
    
    print(f"Found {len(results2['hits'])} hits")
    print(f"Processing time: {results2['meta']['elapsed_ms']}ms")
    
    # Example 3: High-helpfulness evidence gathering
    print("\n=== Example 3: Food Quality Evidence ===")
    results3 = hybrid_retrieve(
        business_id="XQfwVwDr-v0ZS3_CbbE5Xw",
        query="food quality taste fresh",
        top_k=15
    )
    
    # Sort evidence by helpfulness
    high_help_evidence = sorted(
        results3['evidence'], 
        key=lambda x: x['helpfulness'], 
        reverse=True
    )[:5]
    
    for evidence in high_help_evidence:
        print(f"Helpful evidence (score: {evidence['helpfulness']:.2f}): {evidence['quote']}")

if __name__ == "__main__":
    demo_t1_hybrid_retrieve()
