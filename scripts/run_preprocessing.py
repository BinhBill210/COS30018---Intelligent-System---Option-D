#!/usr/bin/env python3
"""
Run Data Preprocessing Pipeline for Yelp Dataset
===============================================

This script executes the complete data preprocessing pipeline including:
- Missing value handling
- Duplicate removal
- Spam detection
- Data consistency standardization
- Quality validation

Usage:
    python scripts/run_preprocessing.py [options]
"""

import sys
import argparse
from pathlib import Path
import logging
import time

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_preprocessor import YelpDataPreprocessor

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(project_root / "logs" / "preprocessing.log")
        ]
    )

def main():
    """Main preprocessing execution."""
    parser = argparse.ArgumentParser(
        description="Run Yelp data preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all datasets with default settings
    python scripts/run_preprocessing.py

    # Process only specific datasets
    python scripts/run_preprocessing.py --datasets business review

    # Use different sample sizes
    python scripts/run_preprocessing.py --sample-size 1000000

    # Skip spam detection for faster processing
    python scripts/run_preprocessing.py --no-spam-detection

    # Full processing (no limits)
    python scripts/run_preprocessing.py --full-dataset
        """
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['business', 'review', 'user', 'tip', 'checkin'],
        default=['business', 'review', 'user', 'tip'],
        help='Datasets to process (default: business review user tip)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Sample size for processing (overrides config)'
    )
    
    parser.add_argument(
        '--full-dataset',
        action='store_true',
        help='Process full datasets (no sampling)'
    )
    
    parser.add_argument(
        '--no-spam-detection',
        action='store_true',
        help='Skip spam detection for faster processing'
    )
    
    parser.add_argument(
        '--no-near-duplicates',
        action='store_true',
        help='Skip near-duplicate detection for faster processing'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create logs directory if it doesn't exist
    (project_root / "logs").mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("YELP DATA PREPROCESSING PIPELINE")
    logger.info("="*60)
    
    # Initialize preprocessor
    preprocessor = YelpDataPreprocessor(
        raw_data_path=project_root / "raw_data",
        processed_data_path=project_root / args.output_dir
    )
    
    # Configure dataset processing
    datasets_config = {}
    
    # Default sample sizes
    default_sizes = {
        'business': 50000,
        'review': 100000,
        'user': 30000,
        'tip': 20000,
        'checkin': 10000
    }
    
    for dataset in args.datasets:
        if args.full_dataset:
            datasets_config[dataset] = None  # Process all records
        elif args.sample_size:
            datasets_config[dataset] = args.sample_size
        else:
            datasets_config[dataset] = default_sizes[dataset]
    
    # Display processing plan
    logger.info("Processing Plan:")
    logger.info(f"  Datasets: {', '.join(args.datasets)}")
    for dataset, size in datasets_config.items():
        if size is None:
            logger.info(f"  {dataset}: Full dataset")
        else:
            logger.info(f"  {dataset}: {size:,} records")
    
    logger.info(f"  Spam detection: {'Disabled' if args.no_spam_detection else 'Enabled'}")
    logger.info(f"  Near-duplicate detection: {'Disabled' if args.no_near_duplicates else 'Enabled'}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    if args.dry_run:
        logger.info("DRY RUN - No actual processing will occur")
        return
    
    # Process datasets
    processed_datasets = {}
    total_start_time = time.time()
    
    for dataset_name, max_records in datasets_config.items():
        try:
            logger.info(f"\n🔄 Processing {dataset_name} dataset...")
            start_time = time.time()
            
            # Load data
            df = preprocessor.load_dataset_stream(dataset_name, max_records)
            
            # Step 1: Assess missing values
            preprocessor.assess_missing_values(df, dataset_name)
            
            # Step 2: Handle missing values
            df = preprocessor.handle_missing_values(df, dataset_name)
            
            # Step 3: Remove exact duplicates
            df = preprocessor.detect_exact_duplicates(df, dataset_name)
            
            # Step 4: Handle near duplicates and spam (reviews only)
            if dataset_name == 'review':
                if not args.no_near_duplicates:
                    df = preprocessor.detect_near_duplicates_reviews(df)
                else:
                    logger.info("  Skipping near-duplicate detection")
                
                if not args.no_spam_detection:
                    df = preprocessor.detect_spam_reviews(df)
                else:
                    logger.info("  Skipping spam detection")
            
            # Step 5: Standardize data consistency
            df = preprocessor.standardize_data_consistency(df, dataset_name)
            
            # Step 6: Final validation
            preprocessor.validate_data_quality(df, dataset_name)
            
            # Save processed data
            output_path = preprocessor.processed_data_path / f"{dataset_name}_cleaned.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved cleaned {dataset_name} data to {output_path}")
            
            # Also save as CSV for easy inspection
            csv_path = preprocessor.processed_data_path / f"{dataset_name}_cleaned.csv"
            df.to_csv(csv_path, index=False)
            
            processed_datasets[dataset_name] = df
            
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"✅ {dataset_name}: {len(df):,} clean records ({processing_time:.1f}s)")
            
        except FileNotFoundError as e:
            logger.warning(f"⚠️ {dataset_name} dataset file not found: {e}")
            continue
        except Exception as e:
            logger.error(f"❌ Failed to process {dataset_name}: {e}")
            if args.log_level == 'DEBUG':
                import traceback
                logger.debug(traceback.format_exc())
            continue
    
    # Generate comprehensive quality report
    try:
        logger.info("\n📊 Generating quality report...")
        quality_report = preprocessor.generate_quality_report()
        logger.info("✅ Quality report generated")
    except Exception as e:
        logger.error(f"❌ Failed to generate quality report: {e}")
    
    # Final summary
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*60)
    
    if processed_datasets:
        logger.info("📈 Successfully processed datasets:")
        total_records = 0
        for dataset_name, df in processed_datasets.items():
            record_count = len(df)
            total_records += record_count
            logger.info(f"  ✅ {dataset_name.upper()}: {record_count:,} records")
        
        logger.info(f"\n📊 Total clean records: {total_records:,}")
        logger.info(f"⏱️ Total processing time: {total_time:.1f} seconds")
        logger.info(f"📁 Output location: {preprocessor.processed_data_path}")
        logger.info("🚀 Data ready for LLM Business Improvement Agent!")
        
        # Show next steps
        logger.info("\n🎯 Next Steps:")
        logger.info("  1. Review quality report: data/processed/data_quality_report.json")
        logger.info("  2. Examine cleaned data: data/processed/*_cleaned.parquet")
        logger.info("  3. Start LLM agent development with clean data")
        
    else:
        logger.warning("⚠️ No datasets were successfully processed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
