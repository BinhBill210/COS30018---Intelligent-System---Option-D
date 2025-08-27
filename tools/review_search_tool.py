from chromadb_integration import ChromaDBVectorStore



class ReviewSearchTool:
    """Updated ReviewSearchTool that uses ChromaDB instead of FAISS"""
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        self.vector_store = ChromaDBVectorStore(
            collection_name="yelp_reviews",
            persist_directory=chroma_db_path
        )
        info = self.vector_store.get_collection_info()
        if "error" in info or info.get("count", 0) == 0:
            print("Warning: No data found in ChromaDB. Run migration first: python migrate_to_chromadb.py")

    def __call__(self, query: str, k: int = 5):
        try:
            search_results = self.vector_store.similarity_search(query, k=k)
            results = []
            for doc, score in search_results:
                metadata = doc.metadata
                results.append({
                    "review_id": metadata.get("review_id", ""),
                    "text": doc.page_content,
                    "stars": metadata.get("stars", ""),
                    "business_id": metadata.get("business_id", ""),
                    "date": metadata.get("date", ""),
                    "score": float(score)
                })
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []





