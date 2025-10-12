from typing import Optional, Dict, Any, List
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import DuckDBManager

class BusinessPulse:
    """
    T2 BusinessPulse - Business overview and sanity checks tool.
    
    Purpose: Provide quick overview and health metrics for a business,
    including review counts, rating trends, and basic sanity checks.
    """
    
    def __init__(self, data_path: str = "data/processed/review_cleaned.parquet"):
        """
        Initialize BusinessPulse tool.
        
        Args:
            data_path: Path to review data (for compatibility)
        """
        self.data_path = data_path
        self.db_manager = DuckDBManager()
        
    def __call__(self, business_id: str, time_range: Optional[str] = None) -> Dict[str, Any]:
        """
        Get business pulse overview and sanity checks.
        
        Args:
            business_id: Target business identifier
            time_range: Optional time range filter ('3M', '6M', '1Y', 'all')
        
        Returns:
            Dictionary with business metrics and health indicators
        """
        try:
            # Build time filter
            time_filter = ""
            if time_range and time_range != 'all':
                if time_range == '3M':
                    time_filter = "AND date >= CURRENT_DATE - INTERVAL 3 MONTH"
                elif time_range == '6M':
                    time_filter = "AND date >= CURRENT_DATE - INTERVAL 6 MONTH"
                elif time_range == '1Y':
                    time_filter = "AND date >= CURRENT_DATE - INTERVAL 1 YEAR"
            
            # Basic metrics query
            metrics_query = f"""
            SELECT 
                COUNT(*) as total_reviews,
                AVG(stars) as avg_rating,
                MIN(stars) as min_rating,
                MAX(stars) as max_rating,
                SUM(CASE WHEN stars >= 4 THEN 1 ELSE 0 END) as positive_reviews,
                SUM(CASE WHEN stars <= 2 THEN 1 ELSE 0 END) as negative_reviews,
                AVG(useful) as avg_useful,
                AVG(funny) as avg_funny,
                AVG(cool) as avg_cool,
                MIN(date) as earliest_review,
                MAX(date) as latest_review
            FROM reviews 
            WHERE business_id = '{business_id}' {time_filter}
            """
            
            metrics_df = self.db_manager.execute_query(metrics_query)
            
            if metrics_df.empty or metrics_df.iloc[0]['total_reviews'] == 0:
                return {
                    "business_id": business_id,
                    "status": "no_data",
                    "message": "No review data found for this business",
                    "time_range": time_range or "all"
                }
            
            metrics = metrics_df.iloc[0]
            
            # Rating distribution
            distribution_query = f"""
            SELECT 
                stars,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM reviews 
            WHERE business_id = '{business_id}' {time_filter}
            GROUP BY stars
            ORDER BY stars
            """
            
            distribution_df = self.db_manager.execute_query(distribution_query)
            rating_distribution = distribution_df.to_dict('records')
            
            # Recent trend (last 30 days vs previous 30 days)
            trend_query = f"""
            WITH recent_30 AS (
                SELECT AVG(stars) as avg_stars, COUNT(*) as count
                FROM reviews 
                WHERE business_id = '{business_id}' 
                AND date >= CURRENT_DATE - INTERVAL 30 DAY
            ),
            previous_30 AS (
                SELECT AVG(stars) as avg_stars, COUNT(*) as count
                FROM reviews 
                WHERE business_id = '{business_id}' 
                AND date >= CURRENT_DATE - INTERVAL 60 DAY
                AND date < CURRENT_DATE - INTERVAL 30 DAY
            )
            SELECT 
                r.avg_stars as recent_avg,
                r.count as recent_count,
                p.avg_stars as previous_avg,
                p.count as previous_count,
                COALESCE(r.avg_stars - p.avg_stars, 0) as rating_change,
                COALESCE(r.count - p.count, 0) as volume_change
            FROM recent_30 r
            CROSS JOIN previous_30 p
            """
            
            trend_df = self.db_manager.execute_query(trend_query)
            trend_data = trend_df.iloc[0] if not trend_df.empty else {}
            
            # Health indicators
            health_score = self._calculate_health_score(metrics, rating_distribution)
            sanity_checks = self._perform_sanity_checks(metrics, rating_distribution)
            
            # Compile results
            pulse_data = {
                "business_id": business_id,
                "time_range": time_range or "all",
                "status": "active",
                
                # Core metrics
                "metrics": {
                    "total_reviews": int(metrics['total_reviews']),
                    "avg_rating": round(float(metrics['avg_rating']), 2),
                    "rating_range": [float(metrics['min_rating']), float(metrics['max_rating'])],
                    "positive_reviews": int(metrics['positive_reviews']),
                    "negative_reviews": int(metrics['negative_reviews']),
                    "engagement": {
                        "avg_useful": round(float(metrics['avg_useful']), 2),
                        "avg_funny": round(float(metrics['avg_funny']), 2),
                        "avg_cool": round(float(metrics['avg_cool']), 2)
                    },
                    "date_range": [str(metrics['earliest_review']), str(metrics['latest_review'])]
                },
                
                # Rating distribution
                "rating_distribution": rating_distribution,
                
                # Trend analysis
                "recent_trend": {
                    "rating_change": round(float(trend_data.get('rating_change', 0)), 3),
                    "volume_change": int(trend_data.get('volume_change', 0)),
                    "recent_avg": round(float(trend_data.get('recent_avg', 0)), 2) if trend_data.get('recent_avg') else None,
                    "recent_count": int(trend_data.get('recent_count', 0))
                },
                
                # Health assessment
                "health": {
                    "score": health_score,
                    "indicators": sanity_checks
                }
            }
            
            return pulse_data
            
        except Exception as e:
            return {
                "business_id": business_id,
                "status": "error",
                "message": f"Error generating business pulse: {str(e)}",
                "time_range": time_range or "all"
            }
    
    def _calculate_health_score(self, metrics: pd.Series, distribution: List[Dict]) -> float:
        """Calculate overall business health score (0-100)"""
        score = 0
        
        # Rating score (40% weight)
        avg_rating = float(metrics['avg_rating'])
        rating_score = (avg_rating - 1) / 4 * 40  # Scale 1-5 to 0-40
        
        # Volume score (20% weight)
        total_reviews = int(metrics['total_reviews'])
        if total_reviews > 100:
            volume_score = 20
        elif total_reviews > 50:
            volume_score = 15
        elif total_reviews > 20:
            volume_score = 10
        elif total_reviews > 5:
            volume_score = 5
        else:
            volume_score = 0
        
        # Distribution score (20% weight) - penalize extremes
        if distribution:
            star_counts = {d['stars']: d['count'] for d in distribution}
            total = sum(star_counts.values())
            
            # Healthy distribution has more 4-5 stars but not exclusively
            high_stars = star_counts.get(4, 0) + star_counts.get(5, 0)
            high_ratio = high_stars / total if total > 0 else 0
            
            if 0.6 <= high_ratio <= 0.9:  # Sweet spot
                distribution_score = 20
            elif high_ratio > 0.9:  # Too perfect, suspicious
                distribution_score = 10
            else:
                distribution_score = high_ratio * 20
        else:
            distribution_score = 0
        
        # Engagement score (20% weight)
        avg_useful = float(metrics['avg_useful'])
        engagement_score = min(avg_useful * 5, 20)  # Cap at 20
        
        total_score = rating_score + volume_score + distribution_score + engagement_score
        return round(min(total_score, 100), 1)
    
    def _perform_sanity_checks(self, metrics: pd.Series, distribution: List[Dict]) -> List[Dict]:
        """Perform business health sanity checks"""
        checks = []
        
        # Check 1: Sufficient review volume
        total_reviews = int(metrics['total_reviews'])
        if total_reviews < 10:
            checks.append({
                "check": "review_volume",
                "status": "warning",
                "message": f"Low review count ({total_reviews}). Consider encouraging more reviews."
            })
        elif total_reviews > 100:
            checks.append({
                "check": "review_volume", 
                "status": "good",
                "message": f"Strong review volume ({total_reviews})"
            })
        
        # Check 2: Rating reasonableness
        avg_rating = float(metrics['avg_rating'])
        if avg_rating < 2.0:
            checks.append({
                "check": "rating_health",
                "status": "critical",
                "message": f"Very low average rating ({avg_rating:.1f}). Immediate attention needed."
            })
        elif avg_rating > 4.8 and total_reviews > 20:
            checks.append({
                "check": "rating_health",
                "status": "suspicious",
                "message": f"Unusually high rating ({avg_rating:.1f}) for {total_reviews} reviews. Verify authenticity."
            })
        elif avg_rating >= 4.0:
            checks.append({
                "check": "rating_health",
                "status": "good", 
                "message": f"Healthy average rating ({avg_rating:.1f})"
            })
        
        # Check 3: Rating distribution
        if distribution and total_reviews > 10:
            star_5_pct = next((d['percentage'] for d in distribution if d['stars'] == 5), 0)
            star_1_pct = next((d['percentage'] for d in distribution if d['stars'] == 1), 0)
            
            if star_5_pct > 80:
                checks.append({
                    "check": "rating_distribution",
                    "status": "suspicious",
                    "message": f"Extremely high 5-star percentage ({star_5_pct}%). May indicate fake reviews."
                })
            elif star_1_pct > 30:
                checks.append({
                    "check": "rating_distribution",
                    "status": "warning",
                    "message": f"High 1-star percentage ({star_1_pct}%). Check for systematic issues."
                })
            else:
                checks.append({
                    "check": "rating_distribution",
                    "status": "normal",
                    "message": "Rating distribution appears natural"
                })
        
        # Check 4: Engagement levels
        avg_useful = float(metrics['avg_useful'])
        if avg_useful > 2.0:
            checks.append({
                "check": "engagement",
                "status": "good",
                "message": f"High review engagement (avg useful: {avg_useful:.1f})"
            })
        elif avg_useful < 0.5 and total_reviews > 20:
            checks.append({
                "check": "engagement",
                "status": "low",
                "message": f"Low review engagement (avg useful: {avg_useful:.1f})"
            })
        
        return checks
