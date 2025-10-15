# data_split.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths - Using pathlib for better cross-platform compatibility
PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'processed_data.parquet'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
TRAIN_PATH = ARTIFACTS_DIR / 'train_data.parquet'
TEST_PATH = ARTIFACTS_DIR / 'test_data.parquet'


def split_and_save_data(data_path=DATA_PATH, test_size=0.3, random_state=42, stratify=True):
    """
    Split processed data into train and test sets and save them.
    Uses stratification by 'target' column to maintain class distribution.
    
    Args:
        data_path (Path/str): Path to processed data file
        test_size (float): Proportion of data for test set (default: 0.3)
        random_state (int): Random seed for reproducibility (default: 42)
        stratify (bool): Whether to use stratified splitting (default: True)
    
    Returns:
        tuple: (train_df, test_df) - DataFrames for train and test sets
    """
    try:
        # Check if data file exists
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Processed data file not found at {data_path}. Please run data processing first.")
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Check if target column exists
        if 'target' not in df.columns:
            logger.warning("Target column 'target' not found. Using random split.")
            stratify = False
        
        # Prepare stratification
        if stratify:
            stratify_col = df['target']
            logger.info("Using stratified splitting by 'target' column")
            logger.info(f"Target distribution:\n{df['target'].value_counts().sort_index()}")
        else:
            stratify_col = None
            logger.info("Using random splitting (no stratification)")
        
        # Split into train and test
        logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_col
        )
        
        # Ensure artifacts directory exists
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        train_df.to_parquet(TRAIN_PATH, index=False)
        test_df.to_parquet(TEST_PATH, index=False)
        
        logger.info(f"✅ Train data saved to {TRAIN_PATH} (shape: {train_df.shape})")
        logger.info(f"✅ Test data saved to {TEST_PATH} (shape: {test_df.shape})")
        
        # Print split statistics
        print_split_statistics(train_df, test_df, stratify)
        
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise


def print_split_statistics(train_df, test_df, stratify=True):
    """Print detailed statistics about the data split."""
    print("\n" + "="*50)
    print("📊 DATA SPLIT STATISTICS")
    print("="*50)
    print(f"Train set size: {len(train_df):,} rows ({len(train_df)/(len(train_df)+len(test_df))*100:.1f}%)")
    print(f"Test set size:  {len(test_df):,} rows ({len(test_df)/(len(train_df)+len(test_df))*100:.1f}%)")
    
    if stratify and 'target' in train_df.columns:
        print(f"\n🎯 Target Distribution (Stratified):")
        
        # Original distribution
        original_ratio = (train_df['target'].value_counts().sort_index() + 
                         test_df['target'].value_counts().sort_index())
        original_pct = (original_ratio / original_ratio.sum() * 100).round(1)
        
        # Train distribution
        train_counts = train_df['target'].value_counts().sort_index()
        train_pct = (train_counts / len(train_df) * 100).round(1)
        
        # Test distribution  
        test_counts = test_df['target'].value_counts().sort_index()
        test_pct = (test_counts / len(test_df) * 100).round(1)
        
        print(f"{'Class':<10} {'Original':<12} {'Train':<12} {'Test':<12}")
        print("-" * 50)
        for class_val in original_ratio.index:
            print(f"{class_val:<10} {original_pct[class_val]:<11}% {train_pct[class_val]:<11}% {test_pct[class_val]:<11}%")
        
        # Check if stratification worked properly
        stratification_quality = all(abs(train_pct - test_pct) < 2)  # Within 2% difference
        if stratification_quality:
            print("✅ Excellent stratification - class distributions are well balanced")
        else:
            print("⚠️  Stratification warning - some class distributions differ between splits")
    
    print(f"\n🔢 Features: {train_df.shape[1]} columns")
    print(f"🎯 Target variable: {'target' if 'target' in train_df.columns else 'NOT FOUND'}")
    print(f"📁 Train data saved to: {TRAIN_PATH}")
    print(f"📁 Test data saved to:  {TEST_PATH}")


def load_split_data():
    """
    Load previously split train and test data.
    
    Returns:
        tuple: (train_df, test_df) or (None, None) if files don't exist
    """
    try:
        if not TRAIN_PATH.exists() or not TEST_PATH.exists():
            logger.warning("Split data files not found. Run split_and_save_data() first.")
            return None, None
        
        train_df = pd.read_parquet(TRAIN_PATH)
        test_df = pd.read_parquet(TEST_PATH)
        
        logger.info(f"Loaded train data: {train_df.shape}")
        logger.info(f"Loaded test data: {test_df.shape}")
        
        # Print quick stats
        if 'target' in train_df.columns:
            print(f"Train target distribution: {dict(train_df['target'].value_counts().sort_index())}")
            print(f"Test target distribution:  {dict(test_df['target'].value_counts().sort_index())}")
        
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"Error loading split data: {e}")
        return None, None


def validate_split_quality(train_df, test_df, max_diff_pct=2.0):
    """
    Validate that the train-test split maintains target distribution.
    
    Args:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Test data
        max_diff_pct (float): Maximum allowed percentage difference between splits
    
    Returns:
        bool: True if split quality is acceptable
    """
    if 'target' not in train_df.columns or 'target' not in test_df.columns:
        logger.warning("Cannot validate split quality - 'target' column not found")
        return True
    
    train_pct = (train_df['target'].value_counts(normalize=True) * 100).sort_index()
    test_pct = (test_df['target'].value_counts(normalize=True) * 100).sort_index()
    
    differences = abs(train_pct - test_pct)
    max_diff = differences.max()
    
    logger.info(f"Maximum class distribution difference: {max_diff:.2f}%")
    
    if max_diff <= max_diff_pct:
        logger.info("✅ Split quality validation passed")
        return True
    else:
        logger.warning(f"⚠️  Split quality validation failed - max difference {max_diff:.2f}% > {max_diff_pct}%")
        return False


if __name__ == "__main__":
    print("🔍 Data Split Utility with Stratification")
    print("="*50)
    
    try:
        # Check if processed data exists
        if not DATA_PATH.exists():
            print(f"❌ Processed data not found at: {DATA_PATH}")
            print("\n💡 Please run data processing first:")
            print("   python data_processor.py")
            exit(1)
        
        # Load a sample to check target column
        sample_df = pd.read_parquet(DATA_PATH)
        print(f"📊 Processed data shape: {sample_df.shape}")
        
        if 'target' in sample_df.columns:
            target_dist = sample_df['target'].value_counts().sort_index()
            print(f"🎯 Target distribution: {dict(target_dist)}")
            print(f"📈 Class balance: {(target_dist[1] / len(sample_df) * 100):.1f}% positive")
        
        # Perform stratified split
        print("\n--- Performing Stratified Split ---")
        train_df, test_df = split_and_save_data(
            test_size=0.3, 
            random_state=42,
            stratify=True  # Enabled since we have target column
        )
        
        # Validate split quality
        print("\n--- Validating Split Quality ---")
        is_quality_good = validate_split_quality(train_df, test_df)
        
        if train_df is not None and test_df is not None and is_quality_good:
            print("\n🎉 Data splitting completed successfully with stratification!")
            print("💡 Your data is ready for model training.")
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")