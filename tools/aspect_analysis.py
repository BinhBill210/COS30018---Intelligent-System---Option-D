from typing import List, Optional, Dict, Any
from pyabsa import AspectTermExtraction as ATEPC
import uuid
import pandas as pd

class AspectABSAToolParquet:
    """
    Same functionality as AspectABSATool but the data source is a Parquet file.
    Parquet columns expected:
      - text (required)
      - review_id (optional)
      - business_id (optional)
    """

    def __init__(self):
        self.aspect_extractor = ATEPC.AspectExtractor(
            'english',
            auto_device=False,
            cal_perplexity=False
        )

    def read_data(
        self,
        parquet_path: str,
        business_id: Optional[str] = None,
        text_col: str = "text",
        review_id_col: str = "review_id",
        business_id_col: str = "business_id",
    ) -> List[Dict[str, Any]]:
        df = pd.read_parquet(parquet_path)
        if text_col not in df.columns:
            raise ValueError(f"Parquet must contain a '{text_col}' column.")
        # Filter business_id
        if business_id is not None:
            if business_id_col in df.columns:
                df = df[df[business_id_col] == business_id]

        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            t = str(row[text_col])
            rid = str(row[review_id_col]) if review_id_col in df.columns and pd.notna(row.get(review_id_col)) else str(uuid.uuid4())
            bid = row.get(business_id_col) if business_id_col in df.columns else business_id
            results.append({
                "review_id": rid,
                "text": t,
                "business_id": bid
            })
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

        output: Dict[str, Dict[str, Any]] = {}
        representative_snippets: Dict[str, List[str]] = {}
        evidence: Dict[str, List[Dict[str, Any]]] = {}
        aspect_scores: Dict[str, Dict[str, List[float]]] = {}

        for i, r in enumerate(results):
            if "aspect" in r and "probs" in r:
                aspects = r["aspect"]
                probs = r["probs"]
                for asp, prob in zip(aspects, probs):
                    if asp not in output:
                        output[asp] = {"negative": 0, "neutral": 0, "positive": 0, "count": 0}
                        representative_snippets[asp] = []
                        evidence[asp] = []
                        aspect_scores[asp] = {"negative": [], "neutral": [], "positive": []}

                    aspect_scores[asp]["negative"].append(prob[0] * 100)
                    aspect_scores[asp]["neutral"].append(prob[1] * 100)
                    aspect_scores[asp]["positive"].append(prob[2] * 100)
                    output[asp]["count"] += 1

                    if len(representative_snippets[asp]) < 5:
                        representative_snippets[asp].append(content[i])

                    sentiment_score = (prob[2] - prob[0])  # positive - negative
                    evidence[asp].append({
                        "review": content[i],
                        "score": round(sentiment_score, 2),
                        "id": review_ids[i]
                    })

        for asp in list(output.keys()):
            c = output[asp]["count"]
            if c > 0:
                output[asp]["negative"] = round(sum(aspect_scores[asp]["negative"]) / c, 1)
                output[asp]["neutral"]  = round(sum(aspect_scores[asp]["neutral"]) / c, 1)
                output[asp]["positive"] = round(sum(aspect_scores[asp]["positive"]) / c, 1)
                del output[asp]["count"]

        return {
            "aspects": output,
            "representative_snippets": representative_snippets,
            "evidence": evidence
        }

    def __call__(
        self,
        parquet_path: str,
        business_id: Optional[str] = None
    ) -> Dict[str, Any]:
        review_data = self.read_data(
            parquet_path=parquet_path,
            business_id=business_id
        )
        final_result = self.analyze_aspects(review_data)
        return final_result
