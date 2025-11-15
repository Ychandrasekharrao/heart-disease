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
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts' / 'split data sets'
X_TRAIN_PATH = ARTIFACTS_DIR / 'X_train.parquet'
X_TEST_PATH = ARTIFACTS_DIR / 'X_test.parquet'
Y_TRAIN_PATH = ARTIFACTS_DIR / 'y_train.parquet'
Y_TEST_PATH = ARTIFACTS_DIR / 'y_test.parquet'

def split_and_save_data(data_path=DATA_PATH, test_size=0.3, random_state=42, stratify=True):
    """
    Split processed data into train and test sets, save features and target as separate parquet files.
    """
    try:
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Processed data file not found at {data_path}. Please run data processing first.")

        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")

        if 'target' not in df.columns:
            logger.warning("Target column 'target' not found. Using random split.")
            stratify = False

        if stratify:
            stratify_col = df['target']
            logger.info("Using stratified splitting by 'target' column")
            logger.info(f"Target distribution:\n{df['target'].value_counts().sort_index()}")
        else:
            stratify_col = None
            logger.info("Using random splitting (no stratification)")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        # Feature/target split
        X_train = train_df.drop(columns=['target'])
        y_train = train_df[['target']]
        X_test = test_df.drop(columns=['target'])
        y_test = test_df[['target']]

        # Ensure output directory exists
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save splits
        X_train.to_parquet(X_TRAIN_PATH, index=False)
        X_test.to_parquet(X_TEST_PATH, index=False)
        y_train.to_parquet(Y_TRAIN_PATH, index=False)
        y_test.to_parquet(Y_TEST_PATH, index=False)

        logger.info(f"✅ Feature/target splits saved to {ARTIFACTS_DIR} (shapes: {X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape})")

        print_split_statistics(train_df, test_df, stratify)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise

# Adjust rest of functions as in your existing script, but update logic to load/split files as needed.

if __name__ == "__main__":
    print("🔍 Data Split Utility with Stratification")
    print("="*50)

    try:
        if not DATA_PATH.exists():
            print(f"❌ Processed data not found at: {DATA_PATH}")
            print("💡 Please run data processing first: python data_processor.py")
            exit(1)

        sample_df = pd.read_parquet(DATA_PATH)
        print(f"📊 Processed data shape: {sample_df.shape}")
        if 'target' in sample_df.columns:
            target_dist = sample_df['target'].value_counts().sort_index()
            print(f"🎯 Target distribution: {dict(target_dist)}")
            print(f"📈 Class balance: {(target_dist[1] / len(sample_df) * 100):.1f}% positive")

        print("\n--- Performing Stratified Split ---")
        X_train, X_test, y_train, y_test = split_and_save_data(
            test_size=0.3,
            random_state=42,
            stratify=True if 'target' in sample_df.columns else False
        )

        # You can add quality validation step here as needed

        print("\n🎉 Data splitting completed successfully with stratification!")
        print("💡 Your data is ready for model training.")

    except FileNotFoundError as e:
        print(f"❌ {e}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
