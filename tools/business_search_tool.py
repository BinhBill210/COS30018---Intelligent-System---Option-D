import pandas as pd
from chromadb_integration import ChromaDBVectorStore
from rapidfuzz import process

class BusinessSearchTool:
    """Tool for business name/id linking, semantic search, and info retrieval"""
    def __init__(self, business_data_path="data/processed/business_cleaned.parquet", chroma_db_path="./business_chroma_db"):
        if business_data_path.endswith('.parquet'):
            self.df = pd.read_parquet(business_data_path)
        else:
            self.df = pd.read_csv(business_data_path)
        self.vector_store = ChromaDBVectorStore(
            collection_name="yelp_businesses",
            persist_directory=chroma_db_path
        )
        self.name_to_id = {name.lower(): bid for bid, name in zip(self.df['business_id'], self.df['name'])}

    def get_business_id(self, name: str):
        """Exact name lookup"""
        return self.name_to_id.get(name.lower())

    def search_businesses(self, query: str, k: int = 3):
        """Semantic search using ChromaDB embeddings"""
        results = self.vector_store.similarity_search(query, k=k)
        output = []
        for doc, score in results:
            meta = doc.metadata
            output.append({
                "business_id": meta.get("business_id", ""),
                "name": meta.get("name", ""),
                "address": meta.get("address", ""),
                "stars": meta.get("stars", ""),
                "categories": meta.get("categories", ""),
                "score": score
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
