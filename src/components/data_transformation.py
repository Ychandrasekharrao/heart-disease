"""
data_preprocessor.py

This script loads the main DataFrame, splits it into train/test sets, and saves them to the data/ folder.
Reference: data_analysis.ipynb
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os


# Set paths
DATA_DIR = (
    Path(__file__).parent.parent / "data"
    if (Path(__file__).parent.parent / "data").exists()
    else Path(__file__).parents[2] / "data"
)
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.csv"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
TRAIN_PATH = ARTIFACTS_DIR / "train.csv"
TEST_PATH = ARTIFACTS_DIR / "test.csv"


# Create artifacts directory if it doesn't exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# Load processed data
df = pd.read_csv(PROCESSED_DATA_PATH)

# Optionally: drop duplicates, handle missing, etc. (customize as needed)
df = df.drop_duplicates()
# df = df.dropna()  # Uncomment if you want to drop missing values


# Split into train (80%), test (20%)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["target"] if "target" in df.columns else None,
)

# Save to CSV
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print(f"Train data saved to: {TRAIN_PATH}")
print(f"Test data saved to: {TEST_PATH}")
