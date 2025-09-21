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

        # Implement model pipeline
        self.pipe = pipeline(
            task="ner",
            model="gauneg/deberta-v3-base-absa-ate-sentiment",
            aggregation_strategy="simple",
        )

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

            ents = self.pipe(text)  # [{'entity_group':'pos/neg/neu','score':..,'word':..}, ...]
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