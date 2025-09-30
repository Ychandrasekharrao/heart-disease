import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = os.path.join('data', 'processed', 'processed_data.parquet')  # <-- FIXED PATH
TRAIN_PATH = os.path.join('artifacts', 'train_data.parquet')
TEST_PATH = os.path.join('artifacts', 'test_data.parquet')

def split_and_save_data(data_path=DATA_PATH, test_size=0.3, random_state=42):
    # Check if data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found at {data_path}. Please check the path.")

    # Load processed data
    df = pd.read_parquet(data_path)
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Ensure artifacts directory exists
    os.makedirs('artifacts', exist_ok=True)
    
    # Save splits
    train_df.to_parquet(TRAIN_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)
    print(f"Train and test data saved to {TRAIN_PATH} and {TEST_PATH}")

if __name__ == "__main__":
    split_and_save_data()