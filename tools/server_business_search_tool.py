import pandas as pd
import chromadb
from rapidfuzz import process

class BusinessSearchTool:
    """Tool for business name/id linking, semantic search, and info retrieval"""
    def __init__(self, business_data_path="data/processed/business_cleaned.parquet", host="localhost", port=8000):
        if business_data_path.endswith('.parquet'):
            self.df = pd.read_parquet(business_data_path)
        else:
            self.df = pd.read_csv(business_data_path)
        
        # Change: Replace ChromaDBVectorStore with ChromaDB HTTP client
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_collection("yelp_businesses")
        
        self.name_to_id = {name.lower(): bid for bid, name in zip(self.df['business_id'], self.df['name'])}

    def get_business_id(self, name: str):
        """Exact name lookup"""
        return self.name_to_id.get(name.lower())

    def search_businesses(self, query: str, k: int = 3):
        """Semantic search using ChromaDB embeddings"""
        # Change: Use collection.query instead of vector_store.similarity_search
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        output = []
        # Change: Process results from collection.query format
        if results and 'ids' in results and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                # Get metadata from results
                meta = results['metadatas'][0][i] if 'metadatas' in results else {}
                
                # Distance in ChromaDB (lower is more similar)
                distance = results['distances'][0][i] if 'distances' in results else 0
                
                # Convert to similarity score (higher is more similar)
                # Note: Using same conversion as original tool for consistency
                similarity_score = 1.0 - distance
                
                output.append({
                    "business_id": meta.get("business_id", ""),
                    "name": meta.get("name", ""),
                    "address": meta.get("address", ""),
                    "stars": meta.get("stars", ""),
                    "categories": meta.get("categories", ""),
                    "score": similarity_score
                })
                
        return output

    def fuzzy_search(self, query: str, top_n: int = 5):
        """Fuzzy/partial name search using rapidfuzz"""
        names = self.df['name'].tolist()
        matches = process.extract(query, names, limit=top_n)
        results = []
        for match_name, score, idx in matches:
            row = self.df.iloc[idx]
            results.append({
                'business_id': row['business_id'],
                'name': row['name'],
                'address': row['address'],
                'stars': row['stars'],
                'categories': row['categories'],
                'score': score
            })
        return results

    def get_business_info(self, business_id: str):
        """Return general info for a business_id"""
        row = self.df[self.df['business_id'] == business_id]
        if not row.empty:
            return row.iloc[0].to_dict()
        return {}