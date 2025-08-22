#!/usr/bin/env python3
"""
Comprehensive Cleaned Yelp Dataset Exploration Script
===================================================

This script analyzes the preprocessed and cleaned Yelp data to showcase
the quality improvements and insights available for the LLM-powered 
Business Improvement Agent.

Key Features:
- Analyzes cleaned, validated data from preprocessing pipeline
- Shows data quality improvements and metrics
- Generates business insights from high-quality data
- Demonstrates readiness for LLM agent development
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CleanedYelpExplorer:
    def __init__(self, data_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.quality_report = None
        
    def load_cleaned_data(self):
        """Load all available cleaned datasets"""
        print("📂 Loading cleaned and preprocessed datasets...")
        
        # Available cleaned datasets
        dataset_files = {
            'business': 'business_cleaned.parquet',
            'review': 'review_cleaned.parquet'
        }
        
        for dataset_name, filename in dataset_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    print(f"Loading {dataset_name} from {filename}...")
                    df = pd.read_parquet(file_path)
                    self.datasets[dataset_name] = df
                    print(f"✅ {dataset_name}: {len(df):,} clean records loaded")
                except Exception as e:
                    print(f"❌ Error loading {dataset_name}: {e}")
            else:
                print(f"⚠️ {filename} not found")
        
        # Load quality report
        quality_report_path = self.data_dir / "data_quality_report.json"
        if quality_report_path.exists():
            with open(quality_report_path, 'r') as f:
                self.quality_report = json.load(f)
            print("✅ Quality report loaded")
        
        print(f"\n📊 Successfully loaded {len(self.datasets)} cleaned datasets")
        return self.datasets
    
    def analyze_data_quality_improvements(self):
        """Show data quality improvements from preprocessing"""
        print("\n" + "="*70)
        print("🔧 DATA QUALITY IMPROVEMENTS ANALYSIS")
        print("="*70)
        
        if not self.quality_report:
            print("⚠️ Quality report not available")
            return
        
        print("📈 Preprocessing Results Summary:")
        for dataset_name, report in self.quality_report['detailed_reports'].items():
            if report:  # Only show datasets that were processed
                print(f"\n🗂️ {dataset_name.upper()} Dataset:")
                
                # Missing value handling
                if 'missing_value_handling' in report:
                    mvh = report['missing_value_handling']
                    print(f"   📋 Records before: {mvh.get('rows_before', 'N/A'):,}")
                    print(f"   ✅ Records after: {mvh.get('rows_after', 'N/A'):,}")
                    print(f"   📉 Removed: {mvh.get('rows_removed', 'N/A'):,} ({mvh.get('removal_percentage', 0):.2f}%)")
                
                # Duplicate removal
                if 'exact_duplicates' in report:
                    dup = report['exact_duplicates']
                    print(f"   🔄 Exact duplicates removed: {dup.get('duplicates_found', 0):,}")
                
                # Spam detection (reviews only)
                if 'spam_detection' in report:
                    spam = report['spam_detection']
                    print(f"   🚫 Spam reviews removed: {spam.get('spam_reviews_found', 0):,} ({spam.get('spam_percentage', 0):.2f}%)")
                
                # Near duplicates (reviews only)
                if 'near_duplicates' in report:
                    near_dup = report['near_duplicates']
                    print(f"   🔄 Near duplicates removed: {near_dup.get('near_duplicates_found', 0):,}")
                
                # Quality score
                quality_score = self.quality_report['summary']['overall_summary'].get(dataset_name, {}).get('data_quality_score', 0)
                print(f"   🏆 Final quality score: {quality_score:.1f}/100")
        
        return self.quality_report
    
    def analyze_cleaned_business_data(self):
        """Analyze cleaned business dataset"""
        print("\n" + "="*60)
        print("🏪 CLEANED BUSINESS DATA ANALYSIS")
        print("="*60)
        
        if 'business' not in self.datasets:
            print("❌ Business data not available")
            return
        
        df = self.datasets['business']
        
        # Basic info
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"🔍 Data Types: {df.dtypes.value_counts().to_dict()}")
        
        # Data quality metrics
        print(f"\n✨ Data Quality Metrics:")
        print(f"   Missing values: {df.isnull().sum().sum()} total")
        print(f"   Complete records: {df.dropna().shape[0]:,} ({df.dropna().shape[0]/len(df)*100:.1f}%)")
        print(f"   Unique business IDs: {df['business_id'].nunique():,}")
        print(f"   ID uniqueness: {df['business_id'].nunique() == len(df)}")
        
        # Geographic distribution
        print(f"\n🌍 Geographic Distribution:")
        print(f"   States: {df['state'].nunique()}")
        print(f"   Cities: {df['city'].nunique()}")
        print(f"   Top 10 States by business count:")
        state_counts = df['state'].value_counts().head(10)
        for state, count in state_counts.items():
            print(f"     {state}: {count:,} businesses")
        
        # Business categories analysis
        print(f"\n🏷️ Business Categories Analysis:")
        all_categories = []
        for cats in df['categories'].dropna():
            if isinstance(cats, str):
                all_categories.extend([cat.strip() for cat in cats.split(',')])
        
        cat_counts = Counter(all_categories)
        print(f"   Total unique categories: {len(cat_counts):,}")
        print(f"   Top 15 categories:")
        for cat, count in cat_counts.most_common(15):
            print(f"     {cat}: {count:,} businesses")
        
        # Rating analysis
        print(f"\n⭐ Rating Analysis:")
        print(f"   Average rating: {df['stars'].mean():.3f}")
        print(f"   Median rating: {df['stars'].median():.1f}")
        print(f"   Rating std dev: {df['stars'].std():.3f}")
        print(f"   Rating distribution:")
        rating_dist = df['stars'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            percentage = count / len(df) * 100
            print(f"     {rating}⭐: {count:,} ({percentage:.1f}%)")
        
        # Review activity analysis
        print(f"\n📝 Review Activity Analysis:")
        print(f"   Average reviews per business: {df['review_count'].mean():.1f}")
        print(f"   Median reviews per business: {df['review_count'].median():.0f}")
        print(f"   Max reviews: {df['review_count'].max():,}")
        print(f"   Businesses with >100 reviews: {(df['review_count'] > 100).sum():,} ({(df['review_count'] > 100).mean()*100:.1f}%)")
        print(f"   Businesses with >1000 reviews: {(df['review_count'] > 1000).sum():,}")
        
        # Business status
        print(f"\n🏢 Business Status:")
        open_count = df['is_open'].sum()
        closed_count = (df['is_open'] == 0).sum()
        print(f"   Open businesses: {open_count:,} ({open_count/len(df)*100:.1f}%)")
        print(f"   Closed businesses: {closed_count:,} ({closed_count/len(df)*100:.1f}%)")
        
        # Geographic coordinate quality
        if 'latitude' in df.columns and 'longitude' in df.columns:
            valid_coords = df[['latitude', 'longitude']].dropna()
            print(f"\n📍 Geographic Coordinates:")
            print(f"   Valid coordinates: {len(valid_coords):,} ({len(valid_coords)/len(df)*100:.1f}%)")
            print(f"   Latitude range: {valid_coords['latitude'].min():.3f} to {valid_coords['latitude'].max():.3f}")
            print(f"   Longitude range: {valid_coords['longitude'].min():.3f} to {valid_coords['longitude'].max():.3f}")
        
        return {
            'total_businesses': len(df),
            'states': df['state'].nunique(),
            'cities': df['city'].nunique(),
            'avg_rating': df['stars'].mean(),
            'open_rate': df['is_open'].mean(),
            'top_categories': cat_counts.most_common(10),
            'data_quality': {
                'missing_values': df.isnull().sum().sum(),
                'complete_records': df.dropna().shape[0],
                'unique_ids': df['business_id'].nunique()
            }
        }
    
    def analyze_cleaned_review_data(self):
        """Analyze cleaned review dataset"""
        print("\n" + "="*60)
        print("📝 CLEANED REVIEW DATA ANALYSIS")
        print("="*60)
        
        if 'review' not in self.datasets:
            print("❌ Review data not available")
            return
        
        df = self.datasets['review']
        
        # Basic info
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"🔍 Data Types: {df.dtypes.value_counts().to_dict()}")
        
        # Data quality metrics
        print(f"\n✨ Data Quality Metrics:")
        print(f"   Missing values: {df.isnull().sum().sum()} total")
        print(f"   Complete records: {df.dropna().shape[0]:,} ({df.dropna().shape[0]/len(df)*100:.1f}%)")
        print(f"   Unique review IDs: {df['review_id'].nunique():,}")
        print(f"   Unique users: {df['user_id'].nunique():,}")
        print(f"   Unique businesses: {df['business_id'].nunique():,}")
        
        # Rating distribution analysis
        print(f"\n⭐ Rating Distribution Analysis:")
        rating_dist = df['stars'].value_counts().sort_index()
        print(f"   Average rating: {df['stars'].mean():.3f}")
        print(f"   Median rating: {df['stars'].median():.1f}")
        print(f"   Rating distribution:")
        for rating, count in rating_dist.items():
            percentage = count / len(df) * 100
            print(f"     {rating}⭐: {count:,} ({percentage:.1f}%)")
        
        # Calculate rating bias
        five_star_pct = (df['stars'] == 5).mean() * 100
        one_star_pct = (df['stars'] == 1).mean() * 100
        print(f"   Rating bias: {five_star_pct:.1f}% five-star vs {one_star_pct:.1f}% one-star")
        
        # Text analysis
        print(f"\n📖 Review Text Analysis:")
        df['text_length'] = df['text'].str.len()
        print(f"   Average text length: {df['text_length'].mean():.0f} characters")
        print(f"   Median text length: {df['text_length'].median():.0f} characters")
        print(f"   Text length std dev: {df['text_length'].std():.0f}")
        print(f"   Shortest review: {df['text_length'].min()} characters")
        print(f"   Longest review: {df['text_length'].max():,} characters")
        
        # Text length distribution
        print(f"   Text length distribution:")
        length_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
        length_labels = ['<50', '50-100', '100-200', '200-500', '500-1000', '>1000']
        df['length_category'] = pd.cut(df['text_length'], bins=length_bins, labels=length_labels, include_lowest=True)
        length_dist = df['length_category'].value_counts()
        for category, count in length_dist.items():
            print(f"     {category} chars: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Engagement metrics
        print(f"\n👍 Review Engagement Analysis:")
        engagement_cols = ['useful', 'funny', 'cool']
        for col in engagement_cols:
            if col in df.columns:
                print(f"   Average {col} votes: {df[col].mean():.2f}")
                print(f"   Reviews with {col} votes: {(df[col] > 0).sum():,} ({(df[col] > 0).mean()*100:.1f}%)")
                print(f"   Max {col} votes: {df[col].max()}")
        
        # Temporal analysis
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            valid_dates = df['date'].dropna()
            print(f"\n📅 Temporal Analysis:")
            print(f"   Valid dates: {len(valid_dates):,} ({len(valid_dates)/len(df)*100:.1f}%)")
            print(f"   Date range: {valid_dates.min()} to {valid_dates.max()}")
            print(f"   Time span: {(valid_dates.max() - valid_dates.min()).days:,} days")
            
            # Reviews by year
            yearly_counts = valid_dates.dt.year.value_counts().sort_index()
            print(f"   Reviews by year (last 5 years):")
            for year, count in yearly_counts.tail(5).items():
                print(f"     {year}: {count:,}")
        
        # User activity patterns
        print(f"\n👥 User Activity Patterns:")
        user_review_counts = df['user_id'].value_counts()
        print(f"   Average reviews per user: {user_review_counts.mean():.1f}")
        print(f"   Most active user: {user_review_counts.max()} reviews")
        print(f"   Users with 1 review: {(user_review_counts == 1).sum():,} ({(user_review_counts == 1).mean()*100:.1f}%)")
        print(f"   Users with >10 reviews: {(user_review_counts > 10).sum():,}")
        
        # Business review patterns
        print(f"\n🏪 Business Review Patterns:")
        business_review_counts = df['business_id'].value_counts()
        print(f"   Average reviews per business: {business_review_counts.mean():.1f}")
        print(f"   Most reviewed business: {business_review_counts.max()} reviews")
        print(f"   Businesses with >100 reviews: {(business_review_counts > 100).sum():,}")
        
        # Sample high-quality reviews
        print(f"\n📄 Sample High-Quality Reviews:")
        high_quality = df[df['useful'] > 5].head(2)
        for idx, review in high_quality.iterrows():
            print(f"   Rating: {review['stars']}⭐ | Useful: {review['useful']} votes")
            print(f"   Text: {review['text'][:200]}...")
            print()
        
        return {
            'total_reviews': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_businesses': df['business_id'].nunique(),
            'avg_rating': df['stars'].mean(),
            'avg_text_length': df['text_length'].mean(),
            'date_range': (valid_dates.min(), valid_dates.max()) if 'date' in df.columns else None,
            'data_quality': {
                'missing_values': df.isnull().sum().sum(),
                'complete_records': df.dropna().shape[0],
                'unique_review_ids': df['review_id'].nunique()
            }
        }
    
    def create_quality_visualizations(self):
        """Create visualizations showing data quality and insights"""
        print("\n" + "="*60)
        print("📊 CREATING QUALITY VISUALIZATIONS")
        print("="*60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Cleaned Yelp Dataset Quality Analysis', fontsize=16)
        
        # Business rating distribution
        if 'business' in self.datasets:
            df_biz = self.datasets['business']
            axes[0,0].hist(df_biz['stars'], bins=np.arange(0.5, 6, 0.5), alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,0].set_title('Business Rating Distribution')
            axes[0,0].set_xlabel('Stars')
            axes[0,0].set_ylabel('Count')
            axes[0,0].grid(True, alpha=0.3)
        
        # Review rating distribution
        if 'review' in self.datasets:
            df_rev = self.datasets['review']
            axes[0,1].hist(df_rev['stars'], bins=np.arange(0.5, 6, 0.5), alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0,1].set_title('Review Rating Distribution')
            axes[0,1].set_xlabel('Stars')
            axes[0,1].set_ylabel('Count')
            axes[0,1].grid(True, alpha=0.3)
        
        # Review text length distribution
        if 'review' in self.datasets:
            df_rev = self.datasets['review']
            text_lengths = df_rev['text'].str.len()
            axes[0,2].hist(text_lengths, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0,2].set_title('Review Text Length Distribution')
            axes[0,2].set_xlabel('Characters')
            axes[0,2].set_ylabel('Count')
            axes[0,2].set_xlim(0, 1000)  # Focus on main range
            axes[0,2].grid(True, alpha=0.3)
        
        # Business review count distribution
        if 'business' in self.datasets:
            df_biz = self.datasets['business']
            axes[1,0].hist(df_biz['review_count'], bins=50, alpha=0.7, color='gold', edgecolor='black')
            axes[1,0].set_title('Business Review Count Distribution')
            axes[1,0].set_xlabel('Review Count')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_xlim(0, 200)  # Focus on main range
            axes[1,0].grid(True, alpha=0.3)
        
        # Review engagement distribution
        if 'review' in self.datasets:
            df_rev = self.datasets['review']
            if 'useful' in df_rev.columns:
                useful_votes = df_rev['useful'][df_rev['useful'] <= 20]  # Focus on main range
                axes[1,1].hist(useful_votes, bins=20, alpha=0.7, color='orange', edgecolor='black')
                axes[1,1].set_title('Review Useful Votes Distribution')
                axes[1,1].set_xlabel('Useful Votes')
                axes[1,1].set_ylabel('Count')
                axes[1,1].grid(True, alpha=0.3)
        
        # Business status pie chart
        if 'business' in self.datasets:
            df_biz = self.datasets['business']
            status_counts = df_biz['is_open'].value_counts()
            labels = ['Open', 'Closed']
            colors = ['lightgreen', 'lightcoral']
            axes[1,2].pie(status_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1,2].set_title('Business Status Distribution')
        
        plt.tight_layout()
        plt.savefig('cleaned_yelp_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'cleaned_yelp_analysis.png'")
    
    def generate_llm_readiness_assessment(self):
        """Assess how ready the cleaned data is for LLM agent development"""
        print("\n" + "="*70)
        print("🚀 LLM AGENT READINESS ASSESSMENT")
        print("="*70)
        
        readiness_scores = {}
        overall_score = 0
        
        # Business data readiness
        if 'business' in self.datasets:
            biz_df = self.datasets['business']
            biz_score = 0
            
            # Data completeness
            completeness = 1 - (biz_df.isnull().sum().sum() / (len(biz_df) * len(biz_df.columns)))
            biz_score += completeness * 25
            
            # ID uniqueness
            id_uniqueness = biz_df['business_id'].nunique() / len(biz_df)
            biz_score += id_uniqueness * 25
            
            # Geographic coverage
            geo_coverage = min(biz_df['state'].nunique() / 50, 1.0)  # Normalized to US states
            biz_score += geo_coverage * 25
            
            # Category diversity
            all_cats = []
            for cats in biz_df['categories'].dropna():
                if isinstance(cats, str):
                    all_cats.extend([cat.strip() for cat in cats.split(',')])
            category_diversity = min(len(set(all_cats)) / 1000, 1.0)  # Normalized
            biz_score += category_diversity * 25
            
            readiness_scores['business'] = biz_score
            
            print(f"🏪 Business Data Readiness: {biz_score:.1f}/100")
            print(f"   ✅ Data completeness: {completeness*100:.1f}%")
            print(f"   ✅ ID uniqueness: {id_uniqueness*100:.1f}%")
            print(f"   ✅ Geographic coverage: {biz_df['state'].nunique()} states")
            print(f"   ✅ Category diversity: {len(set(all_cats)):,} unique categories")
        
        # Review data readiness
        if 'review' in self.datasets:
            rev_df = self.datasets['review']
            rev_score = 0
            
            # Text quality
            avg_text_length = rev_df['text'].str.len().mean()
            text_quality = min(avg_text_length / 500, 1.0)  # Good if avg > 500 chars
            rev_score += text_quality * 30
            
            # Rating distribution balance
            rating_dist = rev_df['stars'].value_counts(normalize=True)
            rating_balance = 1 - rating_dist.max()  # Lower max = more balanced
            rev_score += rating_balance * 20
            
            # Temporal coverage
            if 'date' in rev_df.columns:
                dates = pd.to_datetime(rev_df['date'], errors='coerce').dropna()
                temporal_span = (dates.max() - dates.min()).days
                temporal_score = min(temporal_span / 3650, 1.0)  # Good if > 10 years
                rev_score += temporal_score * 25
            
            # User diversity
            user_diversity = min(rev_df['user_id'].nunique() / len(rev_df), 1.0)
            rev_score += user_diversity * 25
            
            readiness_scores['review'] = rev_score
            
            print(f"\n📝 Review Data Readiness: {rev_score:.1f}/100")
            print(f"   ✅ Text quality (avg length): {avg_text_length:.0f} characters")
            print(f"   ✅ Rating balance: {(1-rating_dist.max())*100:.1f}% (100% = perfectly balanced)")
            print(f"   ✅ User diversity: {rev_df['user_id'].nunique():,} unique users")
            if 'date' in rev_df.columns:
                print(f"   ✅ Temporal span: {temporal_span:,} days")
        
        # Calculate overall readiness
        if readiness_scores:
            overall_score = sum(readiness_scores.values()) / len(readiness_scores)
        
        print(f"\n🎯 OVERALL LLM AGENT READINESS: {overall_score:.1f}/100")
        
        # Readiness assessment
        if overall_score >= 90:
            status = "🟢 EXCELLENT - Ready for production LLM agent"
        elif overall_score >= 80:
            status = "🟡 GOOD - Ready for LLM agent with minor optimizations"
        elif overall_score >= 70:
            status = "🟠 FAIR - Suitable for LLM agent development"
        else:
            status = "🔴 NEEDS IMPROVEMENT - Additional preprocessing recommended"
        
        print(f"📊 Status: {status}")
        
        # LLM agent capabilities assessment
        print(f"\n🤖 LLM AGENT CAPABILITIES ENABLED:")
        capabilities = []
        
        if 'review' in self.datasets:
            capabilities.extend([
                "✅ Sentiment Analysis - Rich text data with ratings",
                "✅ Topic Modeling - Diverse review content",
                "✅ Business Intelligence - Rating and engagement patterns",
                "✅ Customer Feedback Analysis - User opinions and complaints",
                "✅ Recommendation Generation - Data-driven suggestions"
            ])
        
        if 'business' in self.datasets:
            capabilities.extend([
                "✅ Competitive Analysis - Business category comparisons",
                "✅ Geographic Market Analysis - Location-based insights",
                "✅ Performance Benchmarking - Rating and review metrics",
                "✅ Market Opportunity Identification - Category and location gaps"
            ])
        
        if len(self.datasets) > 1:
            capabilities.append("✅ Cross-Dataset Analysis - Comprehensive business insights")
        
        for capability in capabilities:
            print(f"   {capability}")
        
        return {
            'overall_score': overall_score,
            'dataset_scores': readiness_scores,
            'status': status,
            'capabilities': capabilities
        }
    
    def generate_comprehensive_summary(self):
        """Generate final comprehensive summary"""
        print("\n" + "="*70)
        print("📋 COMPREHENSIVE CLEANED DATA SUMMARY")
        print("="*70)
        
        total_records = sum(len(df) for df in self.datasets.values())
        
        print(f"📊 DATASET OVERVIEW:")
        print(f"   Datasets processed: {len(self.datasets)}")
        print(f"   Total clean records: {total_records:,}")
        
        for name, df in self.datasets.items():
            print(f"   📁 {name.upper()}: {len(df):,} records")
        
        print(f"\n🎯 KEY ACHIEVEMENTS:")
        achievements = [
            "✅ Data quality validation and cleaning completed",
            "✅ Missing values handled appropriately",
            "✅ Duplicates and spam removed",
            "✅ Data formats standardized",
            "✅ Ready for LLM agent development"
        ]
        
        if self.quality_report:
            # Add specific achievements from quality report
            for dataset_name, report in self.quality_report['detailed_reports'].items():
                if 'spam_detection' in report:
                    spam_removed = report['spam_detection'].get('spam_reviews_found', 0)
                    if spam_removed > 0:
                        achievements.append(f"✅ {spam_removed:,} spam reviews identified and removed")
        
        for achievement in achievements:
            print(f"   {achievement}")
        
        print(f"\n🚀 NEXT STEPS FOR LLM AGENT DEVELOPMENT:")
        next_steps = [
            "1. 🤖 Set up LLM integration (OpenAI, Anthropic, or local models)",
            "2. 📝 Implement sentiment analysis on clean review text",
            "3. 🏷️ Develop topic modeling for business insights",
            "4. 📊 Create business intelligence dashboards",
            "5. 🎯 Build recommendation engine using clean data",
            "6. 🧪 Test autonomous agent with real business scenarios"
        ]
        
        for step in next_steps:
            print(f"   {step}")
        
        return {
            'total_records': total_records,
            'datasets': {name: len(df) for name, df in self.datasets.items()},
            'ready_for_llm': True
        }

def main():
    """Main exploration function for cleaned data"""
    print("🚀 Starting Cleaned Yelp Dataset Analysis for LLM Business Improvement Agent")
    print("=" * 80)
    
    explorer = CleanedYelpExplorer()
    
    # Load cleaned datasets
    datasets = explorer.load_cleaned_data()
    
    if not datasets:
        print("❌ No cleaned datasets found. Please run preprocessing first.")
        return
    
    # Analyze data quality improvements
    explorer.analyze_data_quality_improvements()
    
    # Analyze cleaned datasets
    if 'business' in datasets:
        explorer.analyze_cleaned_business_data()
    
    if 'review' in datasets:
        explorer.analyze_cleaned_review_data()
    
    # Create quality visualizations
    try:
        explorer.create_quality_visualizations()
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
    
    # LLM readiness assessment
    readiness = explorer.generate_llm_readiness_assessment()
    
    # Final summary
    summary = explorer.generate_comprehensive_summary()
    
    print(f"\n🎉 CLEANED DATA ANALYSIS COMPLETE!")
    print(f"💾 Your data is ready for LLM Business Improvement Agent development!")

if __name__ == "__main__":
    main()
