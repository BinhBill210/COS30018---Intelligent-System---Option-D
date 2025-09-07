from typing import List, Optional, Dict, Any
import pandas as pd
from transformers import pipeline


class AspectABSAToolHF:
    def __init__(
        self,
        business_data_path: str = "data/processed/business_cleaned.parquet",
        review_data_path: str = "data/processed/review_cleaned.parquet",
    ):
        """Initialize with business and review data - consistent with other tools"""
        self.business_df = (
            pd.read_parquet(business_data_path)
            if business_data_path.endswith(".parquet")
            else pd.read_csv(business_data_path)
        )
        self.review_df = (
            pd.read_parquet(review_data_path)
            if review_data_path.endswith(".parquet")
            else pd.read_csv(review_data_path)
        )

        # HF ATE + sentiment pipeline (POS/NEG/NEU)
        self.pipe = pipeline(
            task="ner",
            model="gauneg/deberta-v3-base-absa-ate-sentiment",
            aggregation_strategy="simple",
        )

    # 1) Lấy dữ liệu review theo business_id
    def read_data(self, business_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Trả về: List[{"review_id","text","business_id"}]
        Yêu cầu cột tối thiểu trong review_df: review_id, business_id, text
        """
        df = self.review_df
        if business_id is not None:
            df = df[df["business_id"] == business_id]

        out: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            out.append(
                {
                    "review_id": str(row.get("review_id", "")),
                    "text": str(row.get("text", "")) if row.get("text", "") is not None else "",
                    "business_id": row.get("business_id", None),
                }
            )
        return out

    # 2) Phân tích & format output giống tool cũ
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
                if not asp:
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

        return {
            "aspects": aspects,
            "representative_snippets": representative_snippets,
            "evidence": evidence,
        }


# --- Ví dụ dùng ---
# tool = AspectABSAToolHF()
# reviews = tool.read_data(business_id="pizza_123")
# result = tool.analyze_aspects(reviews)
# print(result)
