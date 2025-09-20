from typing import List, Optional, Dict, Any
import pandas as pd
from transformers import pipeline
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from database.db_manager import get_db_manager

class AspectABSAToolHF:
    def __init__(self):
        """Initialize with DuckDB database access"""
        self.db_manager = get_db_manager()
        
        # Check if database tables exist
        try:
            business_count = self.db_manager.execute_query("SELECT COUNT(*) as count FROM businesses")
            review_count = self.db_manager.execute_query("SELECT COUNT(*) as count FROM reviews")
            print(f"[INFO] Connected to DuckDB - {business_count.iloc[0, 0]:,} businesses, {review_count.iloc[0, 0]:,} reviews")
            self.db_available = True
        except Exception as e:
            print(f"[ERROR] Cannot access DuckDB tables: {e}")
            self.db_available = False

        # Implement model pipeline with proper error handling
        try:
            import torch
            # Set device explicitly to avoid meta tensor issues
            device = "cpu"  # Force CPU to avoid device mapping issues
            
            self.pipe = pipeline(
                task="ner",
                model="gauneg/deberta-v3-base-absa-ate-sentiment",
                aggregation_strategy="simple",
                device=device
            )
            self.model_available = True
            print(f"[INFO] ABSA model loaded successfully on {device}")
        except Exception as e:
            print(f"[WARNING] Failed to load ABSA model: {e}")
            print("[INFO] AspectABSAToolHF will use basic keyword analysis")
            self.pipe = None
            self.model_available = False

    # Take reviews from 1 business id
    def read_data(self, business_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return: List[{"review_id","text","business_id"}]
        """
        if not self.db_available:
            print("[ERROR] read_data: DuckDB not available")
            return []

        try:
            # Build SQL query based on business_id parameter
            if business_id is not None:
                bid = str(business_id).strip()
                query = """
                SELECT review_id, text, business_id 
                FROM reviews 
                WHERE business_id = ? 
                LIMIT 1000
                """
                df = self.db_manager.execute_query(query, [bid])
                print(f"[DEBUG] read_data: found {len(df)} reviews for business_id='{bid}'")
            else:
                query = """
                SELECT review_id, text, business_id 
                FROM reviews 
                LIMIT 1000
                """
                df = self.db_manager.execute_query(query)
                print(f"[DEBUG] read_data: no business_id provided, returning {len(df)} reviews")

            out: List[Dict[str, Any]] = []
            if df.empty:
                return out

            # Convert DataFrame to required format
            for _, row in df.iterrows():
                text = row.get("text", "")
                out.append({
                    "review_id": str(row.get("review_id", "")),
                    "text": "" if text is None else str(text),
                    "business_id": row.get("business_id", None),
                })
            
            return out
            
        except Exception as e:
            print(f"[ERROR] read_data: Database query failed: {e}")
            return []

    # Analyze aspects
    def analyze_aspects(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Output:
        {
          "aspects": {aspect: {"negative": float, "neutral": float, "positive": float}},
          "representative_snippets": {aspect: [text, ... up to 5]},
          "evidence": {aspect: [{"review": text, "score": +/-1.0..0.0, "id": review_id}, ...]}
        }
        """
        if not reviews:
            return {"aspects": {}, "representative_snippets": {}, "evidence": {}}
        
        # If model is not available, use basic analysis
        if not self.model_available or self.pipe is None:
            return self._basic_aspect_analysis(reviews)
        
        MAX_ASPECTS   = 10
        MAX_EVIDENCE  = 1
        

        aspects: Dict[str, Dict[str, float]] = {}
        representative_snippets: Dict[str, List[str]] = {}
        evidence: Dict[str, List[Dict[str, Any]]] = {}

        sums: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, Dict[str, int]] = {}

        for r in reviews:
            text = (r.get("text") or "").strip()
            rid = r.get("review_id", "")
            if not text:
                continue

            try:
                ents = self.pipe(text)  # [{'entity_group':'pos/neg/neu','score':..,'word':..}, ...]
            except Exception as e:
                print(f"[ERROR] Model inference failed: {e}")
                continue
            for ent in ents:
                asp = (ent.get("word") or "").lower().strip()
                if not asp or len(asp) < 2: #delete the aspect that only have 1 or 2 words
                    continue

                label = (ent.get("entity_group") or "").lower()  # 'pos'|'neg'|'neu'
                conf_pct = float(ent.get("score", 0.0)) * 100.0

                if asp not in sums:
                    sums[asp] = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
                    counts[asp] = {"negative": 0, "neutral": 0, "positive": 0}
                    representative_snippets[asp] = []
                    evidence[asp] = []

                if label == "pos":
                    sums[asp]["positive"] += conf_pct
                    counts[asp]["positive"] += 1
                    ev_score = round(conf_pct / 100.0, 2)
                elif label == "neg":
                    sums[asp]["negative"] += conf_pct
                    counts[asp]["negative"] += 1
                    ev_score = round(-conf_pct / 100.0, 2)
                else:
                    sums[asp]["neutral"] += conf_pct
                    counts[asp]["neutral"] += 1
                    ev_score = 0.0

                if len(representative_snippets[asp]) < 5:
                    representative_snippets[asp].append(text)

                if len(evidence[asp]) < MAX_EVIDENCE:
                    evidence[asp].append({"review": text, "score": ev_score, "id": rid})

        for asp in sums:
            neg_c = counts[asp]["negative"]
            neu_c = counts[asp]["neutral"]
            pos_c = counts[asp]["positive"]
            aspects[asp] = {
                "negative": round(sums[asp]["negative"] / neg_c, 1) if neg_c else 0.0,
                "neutral": round(sums[asp]["neutral"] / neu_c, 1) if neu_c else 0.0,
                "positive": round(sums[asp]["positive"] / pos_c, 1) if pos_c else 0.0,
            }
        #keep only 10 aspects and 1 evidence for 1 aspect which minimize the output
        kept_aspects = list(evidence.keys())[:MAX_ASPECTS]
        evidence = {a: evidence[a][:MAX_EVIDENCE] for a in kept_aspects}
        representative_snippets = {a: representative_snippets.get(a, [])[:5] for a in kept_aspects}
        aspects = {a: aspects[a] for a in kept_aspects}
        
        return {
            "aspects": aspects,
            "representative_snippets": representative_snippets,
            "evidence": evidence,
        }

    def _basic_aspect_analysis(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Basic aspect analysis without ML model - uses keyword matching
        """
        print("[INFO] Using basic aspect analysis (no ML model)")
        
        # Basic aspect keywords
        aspect_keywords = {
            "food": ["food", "taste", "flavor", "delicious", "meal", "dish", "cuisine", "pizza", "burger"],
            "service": ["service", "staff", "waiter", "waitress", "server", "friendly", "rude", "slow"],
            "price": ["price", "cost", "expensive", "cheap", "value", "money", "affordable"],
            "ambiance": ["atmosphere", "ambiance", "decoration", "music", "lighting", "cozy"],
            "location": ["location", "parking", "convenient", "accessible", "downtown"]
        }
        
        # Sentiment keywords
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "love", "best", "perfect", "awesome"]
        negative_words = ["bad", "terrible", "awful", "worst", "hate", "disappointing", "poor", "horrible"]
        
        aspects = {}
        representative_snippets = {}
        evidence = {}
        
        for aspect, keywords in aspect_keywords.items():
            aspect_reviews = []
            pos_count = 0
            neg_count = 0
            neu_count = 0
            
            for review in reviews[:100]:  # Limit to first 100 reviews
                text = (review.get("text") or "").lower()
                if any(keyword in text for keyword in keywords):
                    aspect_reviews.append(review)
                    
                    # Simple sentiment analysis
                    pos_score = sum(1 for word in positive_words if word in text)
                    neg_score = sum(1 for word in negative_words if word in text)
                    
                    if pos_score > neg_score:
                        pos_count += 1
                    elif neg_score > pos_score:
                        neg_count += 1
                    else:
                        neu_count += 1
            
            if aspect_reviews:
                total = len(aspect_reviews)
                aspects[aspect] = {
                    "positive": round((pos_count / total) * 100, 1),
                    "negative": round((neg_count / total) * 100, 1),
                    "neutral": round((neu_count / total) * 100, 1)
                }
                
                representative_snippets[aspect] = [
                    r.get("text", "")[:200] + "..." if len(r.get("text", "")) > 200 else r.get("text", "")
                    for r in aspect_reviews[:3]
                ]
                
                evidence[aspect] = [{
                    "review": r.get("text", "")[:200] + "..." if len(r.get("text", "")) > 200 else r.get("text", ""),
                    "score": 0.5,  # Neutral score for basic analysis
                    "id": r.get("review_id", "")
                } for r in aspect_reviews[:1]]
        
        return {
            "aspects": aspects,
            "representative_snippets": representative_snippets,
            "evidence": evidence,
            "analysis_type": "basic_keyword_matching",
            "note": "Basic analysis without ML model - limited accuracy"
        }