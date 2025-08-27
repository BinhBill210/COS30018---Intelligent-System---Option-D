import pandas as pd
from typing import Optional


class DataSummaryTool:
    """Same as before - no changes needed"""
    def __init__(self, data_path: str):
        self.df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

    def __call__(self, business_id: Optional[str] = None):
        if business_id:
            filtered_df = self.df[self.df["business_id"] == business_id]
        else:
            filtered_df = self.df
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


