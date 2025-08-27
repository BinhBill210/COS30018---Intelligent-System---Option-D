from transformers import pipeline
from typing import List

class SentimentSummaryTool:
    """Same as before - no changes needed"""
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def __call__(self, reviews: List[str]):
        sentiments = self.sentiment_analyzer(reviews)
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0}
        for sentiment in sentiments:
            label = sentiment["label"]
            sentiment_counts[label] += 1
        total = len(sentiments)
        positive_pct = (sentiment_counts["POSITIVE"] / total) * 100 if total > 0 else 0
        negative_pct = (sentiment_counts["NEGATIVE"] / total) * 100 if total > 0 else 0
        return {
            "total_reviews": total,
            "positive_percentage": round(positive_pct, 2),
            "negative_percentage": round(negative_pct, 2),
            "sample_sentiments": sentiments[:3]
        }
