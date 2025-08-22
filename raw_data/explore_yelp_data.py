#!/usr/bin/env python3
"""
Comprehensive Yelp Dataset Exploration Script
============================================

This script provides a detailed analysis of the Yelp Academic Dataset to highlight
key features and insights that would be valuable for an LLM-powered Business
Improvement Agent.

Dataset Overview:
- Business Data: Business information, location, categories, attributes
- Review Data: User reviews with ratings and text  
- User Data: User profiles and social network
- Tip Data: Short tips from users
- Check-in Data: Business check-in information
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import ijson
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class YelpDataExplorer:
    def __init__(self):
        self.datasets = {}
        self.sample_sizes = {
            'business': 10000,
            'review': 50000,
            'user': 10000,
            'tip': 10000,
            'checkin': 5000
        }
        
    def load_sample_data(self, dataset_name, sample_size=None):
        """Load a sample of data from large JSON files"""
        if sample_size is None:
            sample_size = self.sample_sizes.get(dataset_name, 10000)
        
        filename = f"yelp_academic_dataset_{dataset_name}.json"
        print(f"📂 Loading sample of {sample_size} records from {filename}...")
        
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=f"Loading {dataset_name}")):
                if i >= sample_size:
                    break
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(data)
        self.datasets[dataset_name] = df
        print(f"✅ Loaded {len(df)} records for {dataset_name} dataset")
        return df

    def analyze_business_data(self):
        """Analyze business dataset features"""
        print("\n" + "="*60)
        print("🏪 BUSINESS DATA ANALYSIS")
        print("="*60)
        
        df = self.datasets['business']
        
        # Basic info
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Geographic distribution
        print(f"\n🌍 Geographic Distribution:")
        print(f"States: {df['state'].nunique()}")
        print(f"Cities: {df['city'].nunique()}")
        print(f"Top 10 States:")
        print(df['state'].value_counts().head(10))
        
        # Business categories
        print(f"\n🏷️ Business Categories:")
        all_categories = []
        for cats in df['categories'].dropna():
            if isinstance(cats, str):
                all_categories.extend([cat.strip() for cat in cats.split(',')])
        
        cat_counts = Counter(all_categories)
        print(f"Total unique categories: {len(cat_counts)}")
        print(f"Top 15 categories:")
        for cat, count in cat_counts.most_common(15):
            print(f"  {cat}: {count}")
        
        # Rating distribution
        print(f"\n⭐ Rating Analysis:")
        print(f"Average rating: {df['stars'].mean():.2f}")
        print(f"Rating distribution:")
        print(df['stars'].value_counts().sort_index())
        
        # Review count analysis
        print(f"\n📝 Review Count Analysis:")
        print(f"Average reviews per business: {df['review_count'].mean():.1f}")
        print(f"Median reviews per business: {df['review_count'].median():.1f}")
        print(f"Max reviews: {df['review_count'].max()}")
        print(f"Businesses with >100 reviews: {(df['review_count'] > 100).sum()}")
        
        # Business status
        print(f"\n🏢 Business Status:")
        print(f"Open businesses: {df['is_open'].sum()} ({(df['is_open'].sum()/len(df)*100):.1f}%)")
        print(f"Closed businesses: {(df['is_open'] == 0).sum()} ({((df['is_open'] == 0).sum()/len(df)*100):.1f}%)")
        
        # Attributes analysis
        print(f"\n🔧 Business Attributes:")
        attr_sample = df['attributes'].dropna().iloc[0] if not df['attributes'].dropna().empty else {}
        if attr_sample:
            print(f"Example attributes: {list(attr_sample.keys())[:10]}")
        
        return {
            'total_businesses': len(df),
            'states': df['state'].nunique(),
            'cities': df['city'].nunique(),
            'avg_rating': df['stars'].mean(),
            'top_categories': cat_counts.most_common(10),
            'open_businesses': df['is_open'].sum()
        }

    def analyze_review_data(self):
        """Analyze review dataset features"""
        print("\n" + "="*60)
        print("📝 REVIEW DATA ANALYSIS") 
        print("="*60)
        
        df = self.datasets['review']
        
        # Basic info
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Rating distribution
        print(f"\n⭐ Rating Distribution:")
        rating_dist = df['stars'].value_counts().sort_index()
        for star, count in rating_dist.items():
            print(f"  {star} stars: {count} ({count/len(df)*100:.1f}%)")
        
        # Text analysis
        print(f"\n📖 Review Text Analysis:")
        df['text_length'] = df['text'].str.len()
        print(f"Average text length: {df['text_length'].mean():.0f} characters")
        print(f"Median text length: {df['text_length'].median():.0f} characters")
        print(f"Shortest review: {df['text_length'].min()} characters")
        print(f"Longest review: {df['text_length'].max()} characters")
        
        # Useful/funny/cool votes
        print(f"\n👍 Review Engagement:")
        print(f"Average useful votes: {df['useful'].mean():.2f}")
        print(f"Average funny votes: {df['funny'].mean():.2f}")  
        print(f"Average cool votes: {df['cool'].mean():.2f}")
        
        # Temporal analysis
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            print(f"\n📅 Temporal Analysis:")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Reviews by year:")
            yearly_counts = df['date'].dt.year.value_counts().sort_index()
            for year, count in yearly_counts.tail(5).items():
                print(f"  {year}: {count}")
        
        # Sample review
        print(f"\n📄 Sample Review:")
        sample_review = df.iloc[0]
        print(f"Rating: {sample_review['stars']} stars")
        print(f"Text (first 200 chars): {sample_review['text'][:200]}...")
        
        return {
            'total_reviews': len(df),
            'avg_rating': df['stars'].mean(),
            'rating_distribution': rating_dist.to_dict(),
            'avg_text_length': df['text_length'].mean(),
            'date_range': (df['date'].min(), df['date'].max()) if 'date' in df.columns else None
        }

    def analyze_user_data(self):
        """Analyze user dataset features"""
        print("\n" + "="*60)
        print("👥 USER DATA ANALYSIS")
        print("="*60)
        
        df = self.datasets['user']
        
        # Basic info
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # User activity
        print(f"\n📈 User Activity:")
        print(f"Average review count per user: {df['review_count'].mean():.1f}")
        print(f"Median review count per user: {df['review_count'].median():.1f}")
        print(f"Most active user: {df['review_count'].max()} reviews")
        
        # User ratings given
        print(f"\n⭐ User Rating Behavior:")
        print(f"Average user rating: {df['average_stars'].mean():.2f}")
        
        # Social network features
        print(f"\n🤝 Social Network:")
        if 'friends' in df.columns:
            df['friend_count'] = df['friends'].apply(lambda x: len(x.split(',')) if isinstance(x, str) and x.strip() else 0)
            print(f"Average friends per user: {df['friend_count'].mean():.1f}")
            print(f"Max friends: {df['friend_count'].max()}")
        
        # User engagement
        print(f"\n👍 User Engagement:")
        for col in ['useful', 'funny', 'cool']:
            if col in df.columns:
                print(f"Average {col} votes received: {df[col].mean():.1f}")
        
        # Elite users
        if 'elite' in df.columns:
            elite_users = df['elite'].apply(lambda x: len(x.split(',')) if isinstance(x, str) and x.strip() else 0)
            print(f"\n🌟 Elite Users:")
            print(f"Users with elite status: {(elite_users > 0).sum()}")
            print(f"Average elite years: {elite_users[elite_users > 0].mean():.1f}")
        
        return {
            'total_users': len(df),
            'avg_reviews_per_user': df['review_count'].mean(),
            'avg_user_rating': df['average_stars'].mean(),
            'social_features': 'friends' in df.columns
        }

    def analyze_tip_data(self):
        """Analyze tip dataset features"""
        print("\n" + "="*60)
        print("💡 TIP DATA ANALYSIS")
        print("="*60)
        
        df = self.datasets['tip']
        
        # Basic info
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Tip text analysis
        print(f"\n📝 Tip Text Analysis:")
        df['text_length'] = df['text'].str.len()
        print(f"Average tip length: {df['text_length'].mean():.0f} characters")
        print(f"Median tip length: {df['text_length'].median():.0f} characters")
        
        # Compliment count
        if 'compliment_count' in df.columns:
            print(f"\n👏 Compliments:")
            print(f"Average compliments per tip: {df['compliment_count'].mean():.2f}")
            print(f"Tips with compliments: {(df['compliment_count'] > 0).sum()}")
        
        # Sample tip
        print(f"\n💬 Sample Tip:")
        sample_tip = df.iloc[0]
        print(f"Text: {sample_tip['text'][:150]}...")
        
        return {
            'total_tips': len(df),
            'avg_tip_length': df['text_length'].mean()
        }

    def create_visualizations(self):
        """Create key visualizations"""
        print("\n" + "="*60)
        print("📊 CREATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Yelp Dataset Key Insights', fontsize=16)
        
        # Business rating distribution
        if 'business' in self.datasets:
            df_biz = self.datasets['business']
            axes[0,0].hist(df_biz['stars'], bins=10, alpha=0.7, color='skyblue')
            axes[0,0].set_title('Business Rating Distribution')
            axes[0,0].set_xlabel('Stars')
            axes[0,0].set_ylabel('Count')
        
        # Review rating distribution
        if 'review' in self.datasets:
            df_rev = self.datasets['review']
            axes[0,1].hist(df_rev['stars'], bins=5, alpha=0.7, color='lightcoral')
            axes[0,1].set_title('Review Rating Distribution')
            axes[0,1].set_xlabel('Stars')
            axes[0,1].set_ylabel('Count')
        
        # Review text length distribution
        if 'review' in self.datasets:
            df_rev = self.datasets['review']
            axes[1,0].hist(df_rev['text'].str.len(), bins=50, alpha=0.7, color='lightgreen')
            axes[1,0].set_title('Review Text Length Distribution')
            axes[1,0].set_xlabel('Characters')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_xlim(0, 2000)  # Focus on reasonable range
        
        # Business review count distribution
        if 'business' in self.datasets:
            df_biz = self.datasets['business']
            axes[1,1].hist(df_biz['review_count'], bins=50, alpha=0.7, color='gold')
            axes[1,1].set_title('Business Review Count Distribution')
            axes[1,1].set_xlabel('Review Count')
            axes[1,1].set_ylabel('Count')
            axes[1,1].set_xlim(0, 200)  # Focus on reasonable range
        
        plt.tight_layout()
        plt.savefig('yelp_dataset_insights.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_business_insights(self):
        """Generate key business insights for LLM agent"""
        print("\n" + "="*60)
        print("🎯 KEY BUSINESS INSIGHTS FOR LLM AGENT")
        print("="*60)
        
        insights = []
        
        # Business analysis insights
        if 'business' in self.datasets:
            biz_df = self.datasets['business']
            
            # Category insights
            all_cats = []
            for cats in biz_df['categories'].dropna():
                if isinstance(cats, str):
                    all_cats.extend([cat.strip() for cat in cats.split(',')])
            top_categories = Counter(all_cats).most_common(5)
            
            insights.append(f"🏷️ Most common business categories: {[cat for cat, _ in top_categories]}")
            insights.append(f"⭐ Average business rating: {biz_df['stars'].mean():.2f}")
            insights.append(f"🏪 Business closure rate: {((biz_df['is_open'] == 0).sum()/len(biz_df)*100):.1f}%")
        
        # Review analysis insights
        if 'review' in self.datasets:
            rev_df = self.datasets['review']
            
            # Rating bias
            rating_dist = rev_df['stars'].value_counts(normalize=True)
            insights.append(f"📊 Review rating bias: {rating_dist.idxmax()}-star reviews are most common ({rating_dist.max()*100:.1f}%)")
            
            # Text analysis opportunities
            rev_df['text_length'] = rev_df['text'].str.len()
            insights.append(f"📝 Review text range: {rev_df['text_length'].min()}-{rev_df['text_length'].max()} characters")
            insights.append(f"💬 Average review length: {rev_df['text_length'].mean():.0f} characters")
        
        # User behavior insights  
        if 'user' in self.datasets:
            user_df = self.datasets['user']
            insights.append(f"👥 Average user activity: {user_df['review_count'].mean():.1f} reviews per user")
            insights.append(f"🎯 User rating tendency: {user_df['average_stars'].mean():.2f} stars average")
        
        print("🔍 KEY INSIGHTS FOR BUSINESS IMPROVEMENT AGENT:")
        print("-" * 50)
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
            
        print(f"\n💡 DATA RICHNESS ASSESSMENT:")
        print(f"   - Geographic diversity: {biz_df['state'].nunique() if 'business' in self.datasets else 'N/A'} states")
        print(f"   - Business diversity: {len(Counter(all_cats)) if 'business' in self.datasets else 'N/A'} unique categories") 
        print(f"   - Temporal span: Multi-year review history available")
        print(f"   - Text richness: Variable-length reviews with engagement metrics")
        print(f"   - Social features: User networks and elite status available")
        
        return insights

    def dataset_summary(self):
        """Provide comprehensive dataset summary"""
        print("\n" + "="*70)
        print("📋 COMPREHENSIVE DATASET SUMMARY")
        print("="*70)
        
        summary = {
            'datasets_loaded': list(self.datasets.keys()),
            'total_records': sum(len(df) for df in self.datasets.values()),
            'business_insights': {},
            'review_insights': {},
            'user_insights': {},
            'tip_insights': {}
        }
        
        # Add dataset-specific insights
        if 'business' in self.datasets:
            df = self.datasets['business']
            summary['business_insights'] = {
                'count': len(df),
                'avg_rating': df['stars'].mean(),
                'states': df['state'].nunique(),
                'cities': df['city'].nunique(),
                'open_rate': df['is_open'].mean()
            }
        
        if 'review' in self.datasets:
            df = self.datasets['review']
            summary['review_insights'] = {
                'count': len(df),
                'avg_rating': df['stars'].mean(),
                'avg_text_length': df['text'].str.len().mean(),
                'rating_distribution': df['stars'].value_counts().to_dict()
            }
        
        if 'user' in self.datasets:
            df = self.datasets['user']  
            summary['user_insights'] = {
                'count': len(df),
                'avg_reviews': df['review_count'].mean(),
                'avg_rating': df['average_stars'].mean()
            }
        
        if 'tip' in self.datasets:
            df = self.datasets['tip']
            summary['tip_insights'] = {
                'count': len(df),
                'avg_length': df['text'].str.len().mean()
            }
        
        print(f"📊 LOADED DATASETS: {', '.join(summary['datasets_loaded'])}")
        print(f"📈 TOTAL SAMPLE RECORDS: {summary['total_records']:,}")
        
        print(f"\n🎯 BUSINESS IMPROVEMENT AGENT OPPORTUNITIES:")
        print(f"   ✅ Sentiment Analysis: Rich review text with ratings")
        print(f"   ✅ Topic Modeling: Diverse business categories and review content")
        print(f"   ✅ Geographic Analysis: Multi-state/city coverage")
        print(f"   ✅ Temporal Analysis: Date-stamped reviews for trend analysis") 
        print(f"   ✅ User Behavior Analysis: User profiles and activity patterns")
        print(f"   ✅ Business Performance: Rating and review count metrics")
        print(f"   ✅ Competitive Analysis: Category-based business comparisons")
        print(f"   ✅ Recommendation Systems: User-business interaction data")
        
        return summary

def main():
    """Main exploration function"""
    print("🚀 Starting Yelp Dataset Exploration for LLM Business Improvement Agent")
    print("=" * 80)
    
    explorer = YelpDataExplorer()
    
    # Load sample datasets
    datasets_to_load = ['business', 'review', 'user', 'tip']
    
    for dataset in datasets_to_load:
        try:
            explorer.load_sample_data(dataset)
        except FileNotFoundError:
            print(f"⚠️ Warning: {dataset} dataset not found, skipping...")
            continue
        except Exception as e:
            print(f"❌ Error loading {dataset}: {e}")
            continue
    
    # Run analyses
    if 'business' in explorer.datasets:
        explorer.analyze_business_data()
    
    if 'review' in explorer.datasets:
        explorer.analyze_review_data()
    
    if 'user' in explorer.datasets:
        explorer.analyze_user_data()
    
    if 'tip' in explorer.datasets:
        explorer.analyze_tip_data()
    
    # Generate insights
    explorer.generate_business_insights()
    
    # Create visualizations
    try:
        explorer.create_visualizations()
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
    
    # Final summary
    summary = explorer.dataset_summary()
    
    print(f"\n🎉 EXPLORATION COMPLETE!")
    print(f"💾 Sample data loaded and analyzed for LLM Business Improvement Agent development")

if __name__ == "__main__":
    main()
