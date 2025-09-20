# tools/hybrid_retrieval_tool.py
import chromadb
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

class HybridRetrieve:
    """
    T1 HybridRetrieve - Simplified hybrid semantic retrieval with evidence.
    
    Purpose: Efficient semantic search with business filtering and evidence generation.
    Uses direct ChromaDB client for reliability and simplicity.
    """
    
    def __init__(self, data_path: str = "data/processed/review_cleaned.parquet", 
                 chroma_path: str = "./chroma_db", host: str = "localhost", port: int = 8001):
        """
        Initialize the HybridRetrieve tool.
        
        Args:
            data_path: Path to source data (for compatibility)
            chroma_path: Path to ChromaDB storage (for local client)
            host: ChromaDB server host
            port: ChromaDB server port
        """
        self.data_path = data_path
        self.chroma_path = chroma_path
        
        # Try server connection first, fallback to local
        try:
            self.client = chromadb.HttpClient(host=host, port=port)
            self.collection = self.client.get_collection("yelp_reviews")
            # Test connection
            _ = self.collection.count()
            self.connection_mode = "server"
            print(f"✅ Connected to ChromaDB server at {host}:{port}")
        except Exception as e:
            try:
                # Fallback to local client
                self.client = chromadb.PersistentClient(path=chroma_path)
                self.collection = self.client.get_collection("yelp_reviews")
                self.connection_mode = "local"
                print(f"✅ Connected to local ChromaDB at {chroma_path}")
            except Exception as e2:
                raise ConnectionError(f"Failed to connect to ChromaDB. Server error: {e}, Local error: {e2}")

    def __call__(self, business_id: str, query: str, top_k: int = 10, 
                 filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute hybrid retrieval with business filtering and evidence generation.
        
        Args:
            business_id: Target business identifier
            query: Search query string
            top_k: Maximum number of results to return
            filters: Optional filters for date range and star ratings
                    Format: {"date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD", "stars": [min, max]}
        
        Returns:
            Dictionary with search results and evidence
        """
        start_time = time.time()
        
        # Input validation
        if not business_id or not isinstance(business_id, str):
            return self._create_error_response("Invalid business_id", start_time, business_id, query)
        
        if not query or not isinstance(query, str):
            return self._create_error_response("Invalid query", start_time, business_id, query)
        
        top_k = max(1, min(top_k, 50))  # Reasonable limits
        
        try:
            # Build filter for ChromaDB
            where_filter = {"business_id": business_id}
            
            # Add additional filters if provided (simplified for ChromaDB compatibility)
            # Most filtering will be done post-search for reliability
            
            # Perform semantic search
            search_limit = min(top_k * 3, 50)  # Get extra for filtering
            results = self.collection.query(
                query_texts=[query],
                n_results=search_limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results or not results.get("ids") or not results["ids"][0]:
                return self._create_empty_response(business_id, query, start_time)
            
            # Process results
            hits = []
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                document = results["documents"][0][i] if results.get("documents") else ""
                distance = results["distances"][0][i] if results.get("distances") else 1.0
                
                # Convert distance to similarity score
                similarity_score = max(0, 1.0 - distance)
                
                hit = {
                    "review_id": metadata.get("review_id", ""),
                    "text": document,
                    "stars": metadata.get("stars", 0),
                    "date": metadata.get("date", ""),
                    "useful": metadata.get("useful", 0),
                    "score": round(similarity_score, 3),
                    "business_id": metadata.get("business_id", business_id)
                }
                
                # Apply post-search filtering
                include_hit = True
                if filters:
                    # Star rating post-filtering
                    if "stars" in filters:
                        stars_filter = filters["stars"]
                        if isinstance(stars_filter, list) and len(stars_filter) == 2:
                            hit_stars = metadata.get("stars", 0)
                            if hit_stars < stars_filter[0] or hit_stars > stars_filter[1]:
                                include_hit = False
                    
                    # Date post-filtering
                    if "date_from" in filters or "date_to" in filters:
                        hit_date = metadata.get("date", "")
                        if "date_from" in filters and hit_date < filters["date_from"]:
                            include_hit = False
                        if "date_to" in filters and hit_date > filters["date_to"]:
                            include_hit = False
                
                if include_hit:
                    hits.append(hit)
            
            # Apply simple diversity filtering
            diverse_hits = self._apply_diversity_filter(hits, top_k)
            
            # Generate evidence quotes
            evidence_quotes = self._generate_evidence_quotes(diverse_hits[:5])
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            return {
                "business_id": business_id,
                "query": query,
                "total_hits": len(diverse_hits),
                "hits": diverse_hits,
                "evidence_quotes": evidence_quotes,
                "summary": f"Found {len(diverse_hits)} relevant reviews for '{query}' in business {business_id}",
                    "elapsed_ms": elapsed_ms,
                "connection_mode": self.connection_mode
            }
            
        except Exception as e:
            logging.error(f"Error in hybrid retrieval: {e}")
            return self._create_error_response(str(e), start_time, business_id, query)
    
    def _apply_diversity_filter(self, hits: List[Dict], target_count: int) -> List[Dict]:
        """
        Apply simple diversity filtering to avoid too similar results.
        """
        if len(hits) <= target_count:
            return hits
        
        # Sort by score first
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)
        
        # Simple diversity: avoid very similar text
        diverse_hits = [hits[0]]  # Always include the best match
        
        for hit in hits[1:]:
            if len(diverse_hits) >= target_count:
                break
            
            # Check if too similar to existing results
            is_diverse = True
            for existing in diverse_hits:
                if self._text_similarity(hit["text"], existing["text"]) > 0.8:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_hits.append(hit)
        
        # Fill remaining slots with highest scoring hits if needed
        while len(diverse_hits) < target_count and len(diverse_hits) < len(hits):
            for hit in hits:
                if hit not in diverse_hits:
                    diverse_hits.append(hit)
                    break
        
        return diverse_hits
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Simple text similarity based on word overlap.
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_evidence_quotes(self, hits: List[Dict]) -> List[str]:
        """
        Generate evidence quotes from top hits.
        """
        quotes = []
        for hit in hits:
            text = hit.get("text", "")
            if not text:
                continue
            
            # Extract first meaningful sentence or truncate
            sentences = text.split(".")
            if sentences and len(sentences[0].strip()) > 10:
                quote = sentences[0].strip()
            else:
                quote = text
            
            # Truncate if too long
            if len(quote) > 150:
                quote = quote[:147] + "..."
            
            if quote and len(quote.strip()) > 10:
                quotes.append(quote)
        
        return quotes
    
    def _create_empty_response(self, business_id: str, query: str, start_time: float) -> Dict[str, Any]:
        """Create response for no results found."""
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "business_id": business_id,
            "query": query,
            "total_hits": 0,
            "hits": [],
            "evidence_quotes": [],
            "summary": f"No reviews found for '{query}' in business {business_id}",
            "elapsed_ms": elapsed_ms,
            "connection_mode": self.connection_mode
        }
    
    def _create_error_response(self, error_msg: str, start_time: float, 
                              business_id: str = "", query: str = "") -> Dict[str, Any]:
        """Create response for errors."""
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "business_id": business_id,
            "query": query,
            "error": error_msg,
            "total_hits": 0,
            "hits": [],
            "evidence_quotes": [],
            "summary": f"Error: {error_msg}",
            "elapsed_ms": elapsed_ms
        }

# Simple alias for backward compatibility
HybridRetrievalTool = HybridRetrieve