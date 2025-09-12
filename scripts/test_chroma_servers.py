"""
Test ChromaDB client connections to the running servers.
"""
import chromadb
import time

def test_business_connection():
    """Test connection to business database server."""
    print("\nTesting connection to business database (port 8000)...")
    start_time = time.time()
    
    try:
        # Connect to the ChromaDB server
        client = chromadb.HttpClient(host="localhost", port=8000)
        collection = client.get_collection("yelp_businesses")
        
        # Get collection info
        count = collection.count()
        
        # Calculate time
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"✅ Successfully connected to business database")
        print(f"Collection 'yelp_businesses' has {count} items")
        print(f"Connection time: {elapsed:.2f} seconds")
        
        # Test a simple query
        results = collection.query(
            query_texts=["coffee shop in downtown"],
            n_results=2
        )
        
        print("\nSample query results:")
        for i, doc_id in enumerate(results['ids'][0]):
            print(f"Result {i+1} ID: {doc_id}")
            if 'metadatas' in results and results['metadatas'][0][i]:
                print(f"   Business name: {results['metadatas'][0][i].get('name', 'N/A')}")
                print(f"   Categories: {results['metadatas'][0][i].get('categories', 'N/A')}")
                print(f"   Address: {results['metadatas'][0][i].get('address', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error connecting to business database: {e}")
        return False

def test_reviews_connection():
    """Test connection to reviews database server."""
    print("\nTesting connection to reviews database (port 8001)...")
    start_time = time.time()
    
    try:
        # Connect to the ChromaDB server
        client = chromadb.HttpClient(host="localhost", port=8001)
        collection = client.get_collection("yelp_reviews")
        
        # Get collection info
        count = collection.count()
        
        # Calculate time
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"✅ Successfully connected to reviews database")
        print(f"Collection 'yelp_reviews' has {count} items")
        print(f"Connection time: {elapsed:.2f} seconds")
        
        # Test a simple query
        results = collection.query(
            query_texts=["amazing food and great service"],
            n_results=2
        )
        
        print("\nSample query results:")
        for i, doc_id in enumerate(results['ids'][0]):
            print(f"Result {i+1} ID: {doc_id}")
            if 'metadatas' in results and results['metadatas'][0][i]:
                print(f"   Business ID: {results['metadatas'][0][i].get('business_id', 'N/A')}")
                print(f"   Rating: {results['metadatas'][0][i].get('stars', 'N/A')}")
                print(f"   Date: {results['metadatas'][0][i].get('date', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error connecting to reviews database: {e}")
        return False

if __name__ == "__main__":
    print("Testing ChromaDB server connections...")
    
    business_ok = test_business_connection()
    reviews_ok = test_reviews_connection()
    
    if business_ok and reviews_ok:
        print("\n✅ All ChromaDB servers are running correctly!")
    else:
        print("\n⚠️ There were issues connecting to the ChromaDB servers.")
        print("Make sure you've started the servers with scripts/start_chroma_servers.py")