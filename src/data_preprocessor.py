#!/usr/bin/env python3
"""
Comprehensive Data Preprocessing Pipeline for Yelp Dataset
=========================================================

This module handles data quality issues including:
- Missing value handling
- Duplicate removal  
- Data consistency standardization
- Spam detection
- Data validation

Designed specifically for LLM-powered Business Improvement Agent requirements.
"""

import json
import pandas as pd
import numpy as np
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import logging
from pathlib import Path
import ijson
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YelpDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for Yelp dataset.
    Handles missing values, duplicates, consistency issues, and spam detection.
    """
    
    def __init__(self, raw_data_path: str = "raw_data", processed_data_path: str = "data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Quality metrics tracking
        self.quality_report = {
            'business': {},
            'review': {},
            'user': {},
            'tip': {},
            'checkin': {}
        }
        
        # Spam detection patterns
        self.spam_patterns = [
            r'^[A-Z\s!]+$',  # ALL CAPS
            r'(call|text|phone).{0,20}\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # Phone numbers
            r'(visit|check|goto).{0,20}(www\.|http)',  # URLs
            r'(.)\1{4,}',  # Repeated characters (aaaaa)
            r'^(.{1,20})\1{3,}',  # Repeated phrases
        ]
        
    def load_dataset_stream(self, dataset_name: str, max_records: Optional[int] = None) -> pd.DataFrame:
        """Load dataset with streaming to handle large files efficiently."""
        filename = f"yelp_academic_dataset_{dataset_name}.json"
        filepath = self.raw_data_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        logger.info(f"Loading {dataset_name} dataset from {filepath}")
        
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=f"Loading {dataset_name}")):
                if max_records and i >= max_records:
                    break
                try:
                    record = json.loads(line.strip())
                    data.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {i+1}: {e}")
                    continue
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records for {dataset_name}")
        return df
    
    def assess_missing_values(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Comprehensive missing value assessment."""
        logger.info(f"Assessing missing values for {dataset_name} dataset")
        
        missing_stats = {}
        total_rows = len(df)
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100
            
            # Additional checks for empty strings and whitespace
            if df[column].dtype == 'object':
                empty_strings = (df[column] == '').sum()
                whitespace_only = df[column].str.strip().eq('').sum() if df[column].dtype == 'object' else 0
                
                missing_stats[column] = {
                    'null_count': missing_count,
                    'null_percentage': missing_percentage,
                    'empty_strings': empty_strings,
                    'whitespace_only': whitespace_only,
                    'total_missing': missing_count + empty_strings + whitespace_only
                }
            else:
                missing_stats[column] = {
                    'null_count': missing_count,
                    'null_percentage': missing_percentage,
                    'total_missing': missing_count
                }
        
        self.quality_report[dataset_name]['missing_values'] = missing_stats
        
        # Log critical missing values
        critical_columns = self._get_critical_columns(dataset_name)
        for col in critical_columns:
            if col in missing_stats:
                total_missing = missing_stats[col]['total_missing']
                if total_missing > 0:
                    logger.warning(f"Critical column '{col}' has {total_missing} missing values ({missing_stats[col].get('null_percentage', 0):.2f}%)")
        
        return missing_stats
    
    def _get_critical_columns(self, dataset_name: str) -> List[str]:
        """Define critical columns for each dataset type."""
        critical_columns = {
            'business': ['business_id', 'name', 'stars'],
            'review': ['review_id', 'user_id', 'business_id', 'stars', 'text'],
            'user': ['user_id', 'review_count'],
            'tip': ['user_id', 'business_id', 'text'],
            'checkin': ['business_id']
        }
        return critical_columns.get(dataset_name, [])
    
    def handle_missing_values(self, df: pd.DataFrame, dataset_name: str, strategy: str = 'smart') -> pd.DataFrame:
        """Handle missing values based on dataset type and business requirements."""
        logger.info(f"Handling missing values for {dataset_name} with strategy: {strategy}")
        
        df_clean = df.copy()
        rows_before = len(df_clean)
        
        if dataset_name == 'review':
            # Critical: Remove reviews with missing text or ratings
            df_clean = df_clean.dropna(subset=['text', 'stars', 'review_id', 'user_id', 'business_id'])
            
            # Remove empty text reviews
            df_clean = df_clean[df_clean['text'].str.strip() != '']
            
            # Handle engagement metrics (useful, funny, cool) - fill with 0
            engagement_cols = ['useful', 'funny', 'cool']
            for col in engagement_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(0)
        
        elif dataset_name == 'business':
            # Critical: Remove businesses without ID, name, or rating
            df_clean = df_clean.dropna(subset=['business_id', 'name', 'stars'])
            
            # Handle optional fields
            if 'review_count' in df_clean.columns:
                df_clean['review_count'] = df_clean['review_count'].fillna(0)
            
            if 'is_open' in df_clean.columns:
                # If missing, assume business is closed (conservative approach)
                df_clean['is_open'] = df_clean['is_open'].fillna(0)
        
        elif dataset_name == 'user':
            # Critical: Remove users without ID
            df_clean = df_clean.dropna(subset=['user_id'])
            
            # Handle missing review counts and ratings
            if 'review_count' in df_clean.columns:
                df_clean['review_count'] = df_clean['review_count'].fillna(0)
            
            if 'average_stars' in df_clean.columns:
                # Remove users without rating history for quality analysis
                df_clean = df_clean.dropna(subset=['average_stars'])
        
        elif dataset_name == 'tip':
            # Critical: Remove tips without text or essential IDs
            df_clean = df_clean.dropna(subset=['user_id', 'business_id', 'text'])
            df_clean = df_clean[df_clean['text'].str.strip() != '']
        
        rows_after = len(df_clean)
        removed_rows = rows_before - rows_after
        
        logger.info(f"Removed {removed_rows} rows ({removed_rows/rows_before*100:.2f}%) due to missing critical values")
        
        self.quality_report[dataset_name]['missing_value_handling'] = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'rows_removed': removed_rows,
            'removal_percentage': removed_rows/rows_before*100
        }
        
        return df_clean
    
    def detect_exact_duplicates(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Remove exact duplicate records."""
        logger.info(f"Detecting exact duplicates for {dataset_name}")
        
        rows_before = len(df)
        
        # Define subset of columns for duplicate detection based on dataset
        duplicate_subsets = {
            'review': ['user_id', 'business_id', 'text', 'stars', 'date'],
            'business': ['business_id'],  # Business ID should be unique
            'user': ['user_id'],  # User ID should be unique
            'tip': ['user_id', 'business_id', 'text', 'date'],
            'checkin': ['business_id', 'date']
        }
        
        subset_cols = duplicate_subsets.get(dataset_name, None)
        available_cols = [col for col in subset_cols if col in df.columns] if subset_cols else None
        
        if available_cols:
            df_clean = df.drop_duplicates(subset=available_cols, keep='first')
        else:
            df_clean = df.drop_duplicates(keep='first')
        
        rows_after = len(df_clean)
        duplicates_removed = rows_before - rows_after
        
        logger.info(f"Removed {duplicates_removed} exact duplicates ({duplicates_removed/rows_before*100:.2f}%)")
        
        self.quality_report[dataset_name]['exact_duplicates'] = {
            'duplicates_found': duplicates_removed,
            'duplicate_percentage': duplicates_removed/rows_before*100
        }
        
        return df_clean
    
    def detect_near_duplicates_reviews(self, df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
        """Detect and remove near-duplicate reviews using text similarity."""
        logger.info("Detecting near-duplicate reviews")
        
        if 'text' not in df.columns:
            return df
        
        rows_before = len(df)
        
        # Create text hashes for similarity comparison
        df['text_normalized'] = df['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        df['text_hash'] = df['text_normalized'].apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:8])
        
        # Group by user_id and business_id
        near_duplicates = []
        
        for (user_id, business_id), group in tqdm(df.groupby(['user_id', 'business_id']), 
                                                  desc="Checking near duplicates"):
            if len(group) > 1:
                # Multiple reviews from same user for same business
                texts = group['text_normalized'].tolist()
                indices = group.index.tolist()
                
                # Simple similarity check using character overlap
                for i in range(len(texts)):
                    for j in range(i+1, len(texts)):
                        similarity = self._calculate_text_similarity(texts[i], texts[j])
                        if similarity > threshold:
                            # Keep the longer review (more informative)
                            if len(texts[i]) >= len(texts[j]):
                                near_duplicates.append(indices[j])
                            else:
                                near_duplicates.append(indices[i])
        
        # Remove near duplicates
        df_clean = df.drop(index=near_duplicates).drop(columns=['text_normalized', 'text_hash'])
        
        rows_after = len(df_clean)
        near_dups_removed = rows_before - rows_after
        
        logger.info(f"Removed {near_dups_removed} near-duplicate reviews ({near_dups_removed/rows_before*100:.2f}%)")
        
        self.quality_report['review']['near_duplicates'] = {
            'near_duplicates_found': near_dups_removed,
            'near_duplicate_percentage': near_dups_removed/rows_before*100
        }
        
        return df_clean
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on character overlap."""
        if not text1 or not text2:
            return 0.0
        
        # Simple Jaccard similarity on character 3-grams
        def get_ngrams(text, n=3):
            return set([text[i:i+n] for i in range(len(text)-n+1)])
        
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def detect_spam_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and remove spam/bot-generated reviews."""
        logger.info("Detecting spam reviews")
        
        if 'text' not in df.columns:
            return df
        
        rows_before = len(df)
        spam_indices = set()
        
        for idx, text in tqdm(df['text'].items(), desc="Checking for spam", total=len(df)):
            if self._is_spam_review(text):
                spam_indices.add(idx)
        
        # Additional heuristics
        # 1. Very short reviews with extreme ratings
        short_extreme = df[(df['text'].str.len() < 10) & 
                          ((df['stars'] == 1) | (df['stars'] == 5))].index
        spam_indices.update(short_extreme)
        
        # 2. Reviews with excessive repeated characters
        excessive_repeats = df[df['text'].str.contains(r'(.)\1{5,}', regex=True, na=False)].index
        spam_indices.update(excessive_repeats)
        
        # Remove spam reviews
        df_clean = df.drop(index=list(spam_indices))
        
        rows_after = len(df_clean)
        spam_removed = rows_before - rows_after
        
        logger.info(f"Removed {spam_removed} spam reviews ({spam_removed/rows_before*100:.2f}%)")
        
        self.quality_report['review']['spam_detection'] = {
            'spam_reviews_found': spam_removed,
            'spam_percentage': spam_removed/rows_before*100
        }
        
        return df_clean
    
    def _is_spam_review(self, text: str) -> bool:
        """Check if a review matches spam patterns."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return True
        
        text_clean = text.strip()
        
        # Check against spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return True
        
        # Additional checks
        # 1. Too short and generic
        if len(text_clean) < 5:
            return True
        
        # 2. All same character
        if len(set(text_clean.replace(' ', ''))) <= 2:
            return True
        
        # 3. Excessive punctuation
        punct_ratio = sum(1 for c in text_clean if not c.isalnum() and c != ' ') / len(text_clean)
        if punct_ratio > 0.5:
            return True
        
        return False
    
    def standardize_data_consistency(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Standardize data formats for consistency."""
        logger.info(f"Standardizing data consistency for {dataset_name}")
        
        df_clean = df.copy()
        
        # Standardize star ratings (ensure 1-5 scale)
        if 'stars' in df_clean.columns:
            # Convert to numeric and filter valid range
            df_clean['stars'] = pd.to_numeric(df_clean['stars'], errors='coerce')
            valid_ratings = (df_clean['stars'] >= 1) & (df_clean['stars'] <= 5)
            df_clean = df_clean[valid_ratings]
            
            # Round to nearest 0.5 (Yelp's rating system)
            df_clean['stars'] = (df_clean['stars'] * 2).round() / 2
        
        # Standardize dates
        date_columns = ['date', 'yelping_since']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Clean text encoding
        text_columns = ['text', 'name', 'address', 'categories']
        for col in text_columns:
            if col in df_clean.columns and df_clean[col].dtype == 'object':
                # Handle encoding issues
                df_clean[col] = df_clean[col].astype(str)
                # Remove null bytes and other problematic characters
                df_clean[col] = df_clean[col].str.replace(r'[\x00-\x1f\x7f-\x9f]', '', regex=True)
                # Normalize whitespace
                df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Standardize boolean columns
        boolean_columns = ['is_open']
        for col in boolean_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(int)
        
        # Standardize numeric columns
        numeric_columns = ['latitude', 'longitude', 'review_count', 'useful', 'funny', 'cool']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        self.quality_report[dataset_name]['data_standardization'] = {
            'standardized_columns': [col for col in df_clean.columns if col in text_columns + date_columns + boolean_columns + numeric_columns],
            'final_row_count': len(df_clean)
        }
        
        return df_clean
    
    def validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Comprehensive data quality validation."""
        logger.info(f"Validating data quality for {dataset_name}")
        
        validation_results = {}
        
        # 1. Check data types
        validation_results['data_types'] = df.dtypes.to_dict()
        
        # 2. Check value ranges
        if 'stars' in df.columns:
            validation_results['rating_range'] = {
                'min': df['stars'].min(),
                'max': df['stars'].max(),
                'valid_range': (df['stars'] >= 1).all() and (df['stars'] <= 5).all()
            }
        
        # 3. Check for remaining missing values
        validation_results['missing_values'] = df.isnull().sum().to_dict()
        
        # 4. Check text quality
        if 'text' in df.columns:
            text_stats = {
                'min_length': df['text'].str.len().min(),
                'max_length': df['text'].str.len().max(),
                'avg_length': df['text'].str.len().mean(),
                'empty_texts': (df['text'].str.strip() == '').sum()
            }
            validation_results['text_quality'] = text_stats
        
        # 5. Check ID uniqueness where expected
        id_columns = ['business_id', 'user_id', 'review_id']
        for col in id_columns:
            if col in df.columns:
                unique_count = df[col].nunique()
                total_count = len(df)
                validation_results[f'{col}_uniqueness'] = {
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'is_unique': unique_count == total_count
                }
        
        self.quality_report[dataset_name]['validation'] = validation_results
        return validation_results
    
    def process_dataset(self, dataset_name: str, max_records: Optional[int] = None, 
                       save_processed: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline for a dataset."""
        logger.info(f"Starting complete preprocessing for {dataset_name}")
        
        # Load data
        df = self.load_dataset_stream(dataset_name, max_records)
        
        # Step 1: Assess missing values
        self.assess_missing_values(df, dataset_name)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df, dataset_name)
        
        # Step 3: Remove exact duplicates
        df = self.detect_exact_duplicates(df, dataset_name)
        
        # Step 4: Handle near duplicates (reviews only)
        if dataset_name == 'review':
            df = self.detect_near_duplicates_reviews(df)
            df = self.detect_spam_reviews(df)
        
        # Step 5: Standardize data consistency
        df = self.standardize_data_consistency(df, dataset_name)
        
        # Step 6: Final validation
        self.validate_data_quality(df, dataset_name)
        
        # Save processed data
        if save_processed:
            output_path = self.processed_data_path / f"{dataset_name}_cleaned.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved cleaned {dataset_name} data to {output_path}")
            
            # Also save as CSV for easy inspection
            csv_path = self.processed_data_path / f"{dataset_name}_cleaned.csv"
            df.to_csv(csv_path, index=False)
        
        return df
    
    def generate_quality_report(self, save_report: bool = True) -> Dict:
        """Generate comprehensive data quality report."""
        logger.info("Generating comprehensive quality report")
        
        # Add summary statistics
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'datasets_processed': list(self.quality_report.keys()),
            'overall_summary': {}
        }
        
        for dataset_name, report in self.quality_report.items():
            if report:  # Only process datasets that were actually processed
                summary['overall_summary'][dataset_name] = {
                    'final_record_count': report.get('validation', {}).get('total_count', 0),
                    'data_quality_score': self._calculate_quality_score(report)
                }
        
        full_report = {
            'summary': summary,
            'detailed_reports': self.quality_report
        }
        
        if save_report:
            report_path = self.processed_data_path / "data_quality_report.json"
            with open(report_path, 'w') as f:
                json.dump(full_report, f, indent=2, default=str)
            logger.info(f"Saved quality report to {report_path}")
        
        return full_report
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """Calculate a simple data quality score (0-100)."""
        score = 100.0
        
        # Deduct for missing values in critical columns
        if 'missing_value_handling' in report:
            removal_pct = report['missing_value_handling'].get('removal_percentage', 0)
            score -= min(removal_pct * 0.5, 20)  # Max 20 point deduction
        
        # Deduct for duplicates
        if 'exact_duplicates' in report:
            dup_pct = report['exact_duplicates'].get('duplicate_percentage', 0)
            score -= min(dup_pct * 0.3, 10)  # Max 10 point deduction
        
        # Deduct for spam (reviews only)
        if 'spam_detection' in report:
            spam_pct = report['spam_detection'].get('spam_percentage', 0)
            score -= min(spam_pct * 0.5, 15)  # Max 15 point deduction
        
        return max(score, 0.0)

def main():
    """Main preprocessing pipeline."""
    logger.info("Starting Yelp Data Preprocessing Pipeline")
    
    # Initialize preprocessor
    preprocessor = YelpDataPreprocessor()
    
    # Define datasets to process (with sample sizes for development)
    datasets_config = {
        'business': 50000,  # Sample size for development
        'review': 100000,   # Larger sample for reviews
        'user': 30000,      # Sample for users
        'tip': 20000,       # Sample for tips
        # 'checkin': 10000    # Uncomment if needed
    }
    
    processed_datasets = {}
    
    # Process each dataset
    for dataset_name, max_records in datasets_config.items():
        try:
            logger.info(f"Processing {dataset_name} dataset...")
            df_processed = preprocessor.process_dataset(dataset_name, max_records)
            processed_datasets[dataset_name] = df_processed
            logger.info(f"✅ Successfully processed {dataset_name}: {len(df_processed)} clean records")
        except Exception as e:
            logger.error(f"❌ Failed to process {dataset_name}: {e}")
            continue
    
    # Generate final quality report
    quality_report = preprocessor.generate_quality_report()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DATA PREPROCESSING COMPLETE")
    logger.info("="*60)
    
    for dataset_name, df in processed_datasets.items():
        logger.info(f"✅ {dataset_name.upper()}: {len(df):,} clean records ready for LLM analysis")
    
    logger.info(f"\n📊 Quality reports and cleaned data saved to: {preprocessor.processed_data_path}")
    logger.info("🚀 Data is now ready for LLM Business Improvement Agent!")

if __name__ == "__main__":
    main()
