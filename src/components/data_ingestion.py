"""Data ingestion module for heart disease prediction project.
Handles reading, validating, and splitting the raw data.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

# Set project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Store artifacts OUTSIDE the src directory, e.g., at <project_root>/artifacts/
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion file paths."""
    train_data_path: str = str(ARTIFACTS_DIR / 'train.csv')
    test_data_path: str = str(ARTIFACTS_DIR / 'test.csv')
    raw_data_path: str = str(ARTIFACTS_DIR / 'data.csv')

class DataIngestion:
    """Handles the ingestion of raw data and splitting into train/test."""

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """Read, validate, and split the raw data into train/test sets."""
        logging.info("Entered the data ingestion method or component")
        try:
            # Use the correct relative path from the project root
            source_csv_path = PROJECT_ROOT / 'data' / 'raw' / 'heart disease.csv'
            if not source_csv_path.exists():
                raise FileNotFoundError(
                    f"Source data file not found: {source_csv_path.resolve()}"
                )
            df = pd.read_csv(source_csv_path)
            logging.info(f"Read the dataset as a dataframe from: {source_csv_path}")

            # Defensive: Check for empty dataframe
            if df.empty:
                raise ValueError(f"Loaded dataframe from {source_csv_path} is empty.")

            # Save a copy of the raw data in the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data is saved in artifacts")

            # Defensive: Check for 'target' column existence
            if 'target' not in df.columns:
                raise ValueError("'target' column missing in the source data.")

            # Drop duplicates and reset index for robustness
            before = df.shape[0]
            df = df.drop_duplicates().reset_index(drop=True)
            after = df.shape[0]
            if after < before:
                logging.info(f"Removed {before - after} duplicate rows from raw data.")

            # Check for missing values and log a warning
            if df.isnull().any().any():
                logging.warning(
                    "Missing values detected in the raw data. "
                    "Please handle missing data in preprocessing."
                )

            # Split the data into training and test sets
            logging.info("Performing train-test split")
            try:
                train_set, test_set = train_test_split(
                    df,
                    test_size=0.2,
                    random_state=42,
                    stratify=df['target']
                )
            except ValueError as split_err:
                logging.warning(
                    f"Stratified split failed: {split_err}. Falling back to random split."
                )
                train_set, test_set = train_test_split(
                    df,
                    test_size=0.2,
                    random_state=42
                )

            # Save the train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test sets saved in artifacts")
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except (FileNotFoundError, ValueError) as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys) from e

    def get_artifact_paths(self):
        """Return the paths to the train, test, and raw data artifacts."""
        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path,
            self.ingestion_config.raw_data_path
        )

if __name__ == "__main__":
    logging.info("--- Starting Data Ingestion standalone test ---")
    obj = DataIngestion()
    try:
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        print(
            f"Data ingestion successful. "
            f"Train data saved at: {train_data_path}"
        )
        print(f"Test data saved at: {test_data_path}")
    except CustomException as e:
        print(f"Data ingestion failed: {e}")
    logging.info("--- Data Ingestion standalone test finished ---\n")