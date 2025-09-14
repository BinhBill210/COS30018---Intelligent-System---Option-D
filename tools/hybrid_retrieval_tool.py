# tools/hybrid_retrieval_tool.py
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chromadb_integration import ChromaDBVectorStore

class HybridRetrieve:
    """
    T1 HybridRetrieve - High recall lexical+semantic retrieval with evidence.
    
    Purpose: High-recall lexical+semantic retrieval with diverse, helpfulness-weighted 
    quotes to ground any claim using hybrid semantic (ChromaDB) + lexical filtering.
    """
    
    def __init__(self, data_path: str, chroma_path: str = "./chroma_db"):
        """
        Initialize the HybridRetrieve tool with ChromaDB vector store.
        
        Args:
            data_path: Path to the source data file (for compatibility)
            chroma_path: Path to ChromaDB persistent storage
        """
        self.data_path = data_path
        self.vector_store = ChromaDBVectorStore(
            collection_name="yelp_reviews", 
            persist_directory=chroma_path,
            embedding_model="all-MiniLM-L6-v2"
        )
    
    def __call__(self, business_id: str, query: str, top_k: int = 50, 
                 filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute hybrid retrieval with MMR deduplication and helpfulness weighting.
        
        Args:
            business_id: Target business identifier
            query: Search query string
            top_k: Maximum number of results to return
            filters: Optional filters for date range and star ratings
                    Format: {"date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD", "stars": [1,5]}
        
        Returns:
            Dictionary with hits, evidence, and metadata following T1 spec
        """
        start_time = time.time()
        
        if filters is None:
            filters = {}
        
        # Build ChromaDB compatible filter dictionary
        chroma_filter = {"business_id": business_id}  # Filter by business
        
        # Date range filtering
        if "date_from" in filters or "date_to" in filters:
            date_conditions = {}
            if "date_from" in filters:
                date_conditions["$gte"] = filters["date_from"]
            if "date_to" in filters:
                date_conditions["$lte"] = filters["date_to"]
            chroma_filter["date"] = date_conditions
        
        # Star rating filtering
        if "stars" in filters:
            stars = filters["stars"]
            if isinstance(stars, list) and len(stars) == 2:
                # Range format [min, max]
                chroma_filter["stars"] = {"$gte": stars[0], "$lte": stars[1]}
            else:
                # Specific values
                chroma_filter["stars"] = {"$in": stars}
        
        try:
            # Perform semantic search with ChromaDB
            hits_with_scores = self.vector_store.similarity_search(
                query=query,
                k=min(top_k * 2, 100),  # Get more results for MMR filtering
                filter_dict=chroma_filter
            )
            
            # Apply MMR (Maximal Marginal Relevance) for diversification
            hits = self._apply_mmr_deduplication(hits_with_scores, top_k)
            
            # Sort by helpfulness weight (higher is better)
            hits = sorted(hits, key=lambda x: x.get('helpfulness', 0), reverse=True)
            
            # Generate evidence quotes from top hits
            evidence = self._generate_evidence(hits[:min(10, len(hits))])
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            return {
                "hits": hits[:top_k],
                "evidence": evidence,
                "meta": {
                    "generated_at": datetime.now().isoformat(),
                    "elapsed_ms": elapsed_ms,
                    "params": {"top_k": top_k, "business_id": business_id, "query": query}
                }
            }
            
        except Exception as e:
            return {
                "hits": [],
                "evidence": [],
                "meta": {
                    "generated_at": datetime.now().isoformat(),
                    "elapsed_ms": int((time.time() - start_time) * 1000),
                    "error": str(e),
                    "params": {"top_k": top_k, "business_id": business_id, "query": query}
                }
            }
    
    def _apply_mmr_deduplication(self, hits_with_scores: List, top_k: int, 
                                lambda_param: float = 0.7) -> List[Dict]:
        """
        Apply Maximal Marginal Relevance for result diversification.
        
        Args:
            hits_with_scores: List of (document, score) tuples from vector search
            top_k: Number of diverse results to return
            lambda_param: Balance between relevance and diversity (0.7 = 70% relevance)
        
        Returns:
            List of diversified hit dictionaries
        """
        if not hits_with_scores:
            return []
        
        # Convert to standardized format
        candidates = []
        for doc, score in hits_with_scores:
            metadata = doc.metadata
            hit = {
                "review_id": metadata.get("review_id", ""),
                "score": float(score),
                "text": doc.page_content,
                "stars": metadata.get("stars", 0),
                "date": metadata.get("date", ""),
                "helpfulness": float(metadata.get("helpfulness", 0))
            }
            candidates.append(hit)
        
        if len(candidates) <= top_k:
            return candidates
        
        # Simple MMR approximation: select diverse results based on text similarity
        selected = [candidates[0]]  # Start with highest scoring
        candidates = candidates[1:]
        
        while len(selected) < top_k and candidates:
            best_mmr_score = -float('inf')
            best_idx = 0
            
            for i, candidate in enumerate(candidates):
                # Relevance score (already computed)
                relevance = candidate['score']
                
                # Diversity score (simple text overlap approximation)
                max_similarity = 0
                for selected_hit in selected:
                    # Simple word overlap similarity
                    words1 = set(candidate['text'].lower().split())
                    words2 = set(selected_hit['text'].lower().split())
                    if words1 and words2:
                        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        max_similarity = max(max_similarity, similarity)
                
                # MMR score: balance relevance and diversity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i
            
            selected.append(candidates.pop(best_idx))
        
        return selected
    
    def _generate_evidence(self, hits: List[Dict]) -> List[Dict]:
        """
        Generate evidence quotes from hit results.
        
        Args:
            hits: List of hit dictionaries
        
        Returns:
            List of evidence dictionaries with quotes
        """
        evidence = []
        for hit in hits:
            # Extract meaningful quote (first sentence or up to 150 chars)
            text = hit.get('text', '')
            sentences = text.split('.')
            if sentences:
                quote = sentences[0].strip()
                if len(quote) > 150:
                    quote = quote[:147] + "..."
            else:
                quote = text[:150] + "..." if len(text) > 150 else text
            
            evidence.append({
                "review_id": hit.get("review_id", ""),
                "quote": quote,
                "stars": hit.get("stars", 0),
                "date": hit.get("date", ""),
                "helpfulness": hit.get("helpfulness", 0)
            })
        
        return evidence
