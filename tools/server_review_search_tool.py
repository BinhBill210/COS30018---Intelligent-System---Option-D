import chromadb

class ReviewSearchTool:
    """Updated ReviewSearchTool that uses ChromaDB server and supports business_id filtering"""
    def __init__(self, host="localhost", port=8001):
        # Connect to ChromaDB server instead of loading locally
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_collection("yelp_reviews")
        
        # Get collection info to check if data exists
        try:
            info = self.collection.count()
            if info == 0:
                print("Warning: No data found in ChromaDB. Run migration first: python migrate_to_chromadb.py")
        except Exception as e:
            print(f"Warning: Error connecting to ChromaDB server: {e}")

    def __call__(self, query: str, k: int = 5, business_id: str = None):
        try:
            # Set up filter for business_id if provided
            where_filter = {"business_id": business_id} if business_id else None
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter if where_filter else None
            )
            
            # Process results
            processed_results = []
            if results and 'ids' in results and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    # Get metadata and calculate similarity score
                    metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                    text = results['documents'][0][i] if 'documents' in results else ""
                    distance = results['distances'][0][i] if 'distances' in results else 0
                    
                    # Convert distance to similarity score
                    similarity_score = 1.0 - distance
                    
                    processed_results.append({
                        "review_id": metadata.get("review_id", ""),
                        "text": text,
                        "stars": metadata.get("stars", ""),
                        "business_id": metadata.get("business_id", ""),
                        "date": metadata.get("date", ""),
                        "score": float(similarity_score)
                    })
            
            return processed_results
        except Exception as e:
            print(f"Search error: {e}")
            return []