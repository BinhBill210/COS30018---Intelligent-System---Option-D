import time
import os
from chromadb_integration import ChromaDBVectorStore

def measure_load_time(path, collection_name):
    print(f"Loading ChromaDB from {path}, collection: {collection_name}")
    start_time = time.time()
    
    vector_store = ChromaDBVectorStore(
        collection_name=collection_name,
        persist_directory=path
    )
    
    info = vector_store.get_collection_info()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"ChromaDB loaded in {elapsed:.2f} seconds")
    print(f"Collection info: {info}")
    return elapsed

def test_all_chroma_loads():
    # Test business ChromaDB load time
    business_time = measure_load_time("./business_chroma_db", "yelp_businesses")
    
    # Test review ChromaDB load time
    review_time = measure_load_time("./chroma_db", "yelp_reviews")
    
    print("\nSummary:")
    print(f"Business ChromaDB load time: {business_time:.2f} seconds")
    print(f"Review ChromaDB load time: {review_time:.2f} seconds")
    print(f"Total ChromaDB load time: {business_time + review_time:.2f} seconds")

if __name__ == "__main__":
    test_all_chroma_loads()
