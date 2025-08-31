from typing import List, Optional, Dict, Any
from pyabsa import AspectTermExtraction as ATEPC
from chromadb_integration import ChromaDBVectorStore

class AspectABSATool:
    """Tool for extracting aspects and sentiments from reviews."""

    def __init__(self, chroma_db_path: str = "./chroma_db"):
        self.aspect_extractor = ATEPC.AspectExtractor('english', auto_device=False, cal_perplexity=False)

        self.vector_store = ChromaDBVectorStore(
            collection_name="yelp_reviews",
            persist_directory=chroma_db_path
        )
        info = self.vector_store.get_collection_info()
        if "error" in info or info.get("count", 0) == 0:
            print("Warning: No data found in ChromaDB. Run migration first: python migrate_to_chromadb.py")
    def read_data(self,business_id: Optional[str] = None):
        results = []
        try:
            search_results = self.vector_store.similarity_search("", k = 500)
            for doc, score in search_results:  
                metadata = getattr(doc, "metadata", {}) or {}
                if metadata.get("business_id") == business_id:
                    results.append({
                        "review_id": metadata.get("review_id", ""),
                        "text": getattr(doc, "page_content", ""),
                        "business_id": business_id
                    })
        except Exception as e:
            print(f"Search error: {e}")
        return results

    def analyze_aspects(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not reviews:
            return {"aspects": {}, "representative_snippets": {}, "evidence": {}}

        content = [r.get("text", "") for r in reviews]
        review_ids = [r.get("review_id", "") for r in reviews]
        results = self.aspect_extractor.predict(
            content,
            print_result=False,
            save_result=False,
            ignore_error=True,
            pred_sentiment=True
        )
        
        # output structure
        output = {}
        representative_snippets = {}
        evidence = {}
        
        # Track all scores for averaging
        aspect_scores = {}  # {aspect: {"negative": [scores], "neutral": [scores], "positive": [scores]}}
        
        # Loop through the results
        for i, r in enumerate(results):
            if "aspect" in r and "probs" in r:
                aspects = r["aspect"]
                probs = r["probs"]
                
                # Loop through aspects and probability
                for j, (asp, prob) in enumerate(zip(aspects, probs)):
                    if asp not in output:
                        output[asp] = {
                            "negative": 0,
                            "neutral": 0,
                            "positive": 0,
                            "count": 0
                        }
                        representative_snippets[asp] = []
                        evidence[asp] = []
                        aspect_scores[asp] = {
                            "negative": [],
                            "neutral": [],
                            "positive": []
                        }
                    
                    # Collect scores for averaging
                    aspect_scores[asp]["negative"].append(prob[0] * 100)
                    aspect_scores[asp]["neutral"].append(prob[1] * 100)
                    aspect_scores[asp]["positive"].append(prob[2] * 100)
                    output[asp]["count"] += 1
                    
                    if len(representative_snippets[asp]) < 5:
                        representative_snippets[asp].append(content[i])
                   
                    # Add evidence
                    sentiment_score = (prob[2] - prob[0])  # positive - negative
                    evidence[asp].append({
                        "review": content[i],
                        "score": round(sentiment_score, 2),
                        "id": review_ids[i]
                    })
        
        # Calculate averages 
        for asp in output:
            if output[asp]["count"] > 0:
                output[asp]["negative"] = round(sum(aspect_scores[asp]["negative"]) / output[asp]["count"], 1)
                output[asp]["neutral"] = round(sum(aspect_scores[asp]["neutral"]) / output[asp]["count"], 1)
                output[asp]["positive"] = round(sum(aspect_scores[asp]["positive"]) / output[asp]["count"], 1)
                # Remove count from final output
                del output[asp]["count"]     
        # Final output
        final_output = {
            "aspects": output,
            "representative_snippets": representative_snippets,
            "evidence": evidence
        }
        return final_output

    def __call__(self,business_id: Optional[str] = None):
        review_data = self.read_data(business_id = business_id)
        final_result = self.analyze_aspects(review_data)
        return(final_result)

   