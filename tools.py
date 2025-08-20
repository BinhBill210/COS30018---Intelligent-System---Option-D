# tools.py
from typing import List, Dict, Any
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ReviewSearchTool:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index = faiss.read_index(str(self.index_dir / "reviews.index"))
        
        # Read JSON files with UTF-8 encoding
        with open(self.index_dir / "id_map.json", 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)
        with open(self.index_dir / "meta.json", 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
            
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def __call__(self, query: str, k: int = 5):
        # Embed the query
        query_embedding = self.embedder.encode([query], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy().astype("float32")
        
        # Search the index
        D, I = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, idx in enumerate(I[0]):
            if idx < 0:  # Invalid index
                continue
            chunk_id = self.id_map[idx]
            item_meta = self.meta.get(chunk_id, {})
            results.append({
                "review_id": item_meta.get("orig_id", ""),
                "text": item_meta.get("text", ""),
                "stars": item_meta.get("stars", ""),
                "business_id": item_meta.get("business_id", ""),
                "date": item_meta.get("date", ""),
                "score": float(D[0][i])
            })
        
        return results

class SentimentSummaryTool:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    def __call__(self, reviews: List[str]):
        # Analyze sentiment for each review
        sentiments = self.sentiment_analyzer(reviews)
        
        # Count sentiments
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0}
        for sentiment in sentiments:
            label = sentiment["label"]
            sentiment_counts[label] += 1
        
        # Calculate percentages
        total = len(sentiments)
        positive_pct = (sentiment_counts["POSITIVE"] / total) * 100 if total > 0 else 0
        negative_pct = (sentiment_counts["NEGATIVE"] / total) * 100 if total > 0 else 0
        
        return {
            "total_reviews": total,
            "positive_percentage": round(positive_pct, 2),
            "negative_percentage": round(negative_pct, 2),
            "sample_sentiments": sentiments[:3]  # Return first 3 as examples
        }

class DataSummaryTool:
    def __init__(self, data_path: str):
        self.df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    
    def __call__(self, business_id: str = None):
        if business_id:
            filtered_df = self.df[self.df["business_id"] == business_id]
        else:
            filtered_df = self.df
        
        # Calculate basic statistics
        avg_stars = filtered_df["stars"].mean()
        total_reviews = len(filtered_df)
        useful_avg = filtered_df["useful"].mean()
        funny_avg = filtered_df["funny"].mean()
        cool_avg = filtered_df["cool"].mean()
        
        return {
            "business_id": business_id or "all businesses",
            "total_reviews": total_reviews,
            "average_stars": round(avg_stars, 2),
            "average_useful": round(useful_avg, 2),
            "average_funny": round(funny_avg, 2),
            "average_cool": round(cool_avg, 2)
        }