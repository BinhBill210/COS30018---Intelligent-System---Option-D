import pandas as pd
import chromadb
import sys
from pathlib import Path
from rapidfuzz import process
from typing import Optional

# Add parent directory to path for database imports  
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import get_db_manager


class BusinessSearchTool:
    """Business search tool using DuckDB instead of loading parquet files"""
    def __init__(self, business_data_path="data/processed/business_cleaned.parquet", host="localhost", port=8000):
        # business_data_path parameter kept for compatibility but not used
        try:
            self.db_manager = get_db_manager()
            self.db_available = True
        except:
            self.db_available = False
        
        # ChromaDB client for semantic search (keep existing functionality)
        try:
            self.client = chromadb.HttpClient(host=host, port=port)
            self.collection = self.client.get_collection("yelp_businesses")
            self.chroma_available = True
        except Exception as e:
            self.chroma_available = False
        
        # For compatibility, create name_to_id mapping on demand
        self._name_to_id = None

    def get_business_id(self, name: str):
        """Exact name lookup using DuckDB query"""
        if not self.db_available or not name:
            return None
            
        query = """
        SELECT business_id 
        FROM businesses 
        WHERE LOWER(name) = LOWER(?) 
        LIMIT 1
        """
        
        result = self.db_manager.execute_query(query, params=[name])
        return result.iloc[0, 0] if not result.empty else None

    def search_businesses(self, query: str, k: int = 3):
        """Semantic search using ChromaDB embeddings"""
        if self.chroma_available:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k
                )
                
                output = []
                if results and 'ids' in results and len(results['ids']) > 0:
                    for i in range(len(results['ids'][0])):
                        meta = results['metadatas'][0][i] if 'metadatas' in results else {}
                        distance = results['distances'][0][i] if 'distances' in results else 0
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
            except Exception as e:
                print(f"ChromaDB search failed: {e}")
        
        # Fallback to DuckDB text search if ChromaDB unavailable
        if self.db_available:
            query_sql = """
            SELECT business_id, name, address, stars, categories
            FROM businesses 
            WHERE LOWER(name) LIKE LOWER(?) OR LOWER(categories) LIKE LOWER(?)
            ORDER BY stars DESC, review_count DESC
            LIMIT ?
            """
            
            search_term = f"%{query}%"
            result = self.db_manager.execute_query(query_sql, params=[search_term, search_term, k])
            
            output = []
            for _, row in result.iterrows():
                output.append({
                    "business_id": row['business_id'],
                    "name": row['name'],
                    "address": row['address'],
                    "stars": row['stars'],
                    "categories": row['categories'],
                    "score": 1.0  # Default score for SQL search
                })
            return output
        
        return []

    def fuzzy_search(self, query: str, top_n: int = 5):
        """Fuzzy search for businesses by name. Input can be a string (query) or a dict with 'query' and optional 'top_n'. The input query is used to search the business record with the business name most similar to the input query. Returns a list of similar business records."""
        if not self.db_available:
            return []
        
        # SQL-based fuzzy matching
        query_sql = """
        SELECT business_id, name, address, stars, categories,
               CASE 
                   WHEN LOWER(name) LIKE LOWER(?) THEN 95
                   WHEN LOWER(name) LIKE LOWER(?) THEN 80
                   ELSE 60
               END as score
        FROM businesses 
        WHERE LOWER(name) LIKE LOWER(?) OR LOWER(name) LIKE LOWER(?)
        ORDER BY score DESC, stars DESC, review_count DESC
        LIMIT ?
        """
        
        exact_match = f"%{query}%"
        partial_match = f"%{query.split()[0] if query.split() else query}%"
        
        result = self.db_manager.execute_query(
            query_sql, 
            params=[exact_match, partial_match, exact_match, partial_match, top_n]
        )
        
        results = []
        for _, row in result.iterrows():
            results.append({
                'business_id': row['business_id'],
                'name': row['name'],
                'address': row['address'],
                'stars': row['stars'],
                'categories': row['categories'],
                'score': row['score']
            })
        
        return results

    def get_business_info(self, business_id: str):
        """Return general info for a business_id using DuckDB"""
        if not self.db_available or not business_id:
            return {}
            
        query = "SELECT * FROM businesses WHERE business_id = ?"
        result = self.db_manager.execute_query(query, params=[business_id])
        
        if not result.empty:
            return result.iloc[0].to_dict()
        return {}