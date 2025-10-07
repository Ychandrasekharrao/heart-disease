"""Data transformation module for heart disease prediction.
Handles feature preprocessing, encoding, and creating a preprocessor pipeline.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

from src.exception import CustomException
from src.logger import logging

# Set project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Use the same artifacts directory as data_ingestion
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation paths."""
    preprocessor_path: str = str(ARTIFACTS_DIR / "preprocessor.pkl")
    transformed_train_path: str = str(ARTIFACTS_DIR / "transformed_train.npz")
    transformed_test_path: str = str(ARTIFACTS_DIR / "transformed_test.npz")

class DataTransformation:
    """Handles feature preprocessing, encoding, and scaling."""

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Define the feature engineering pipeline"""
        try:
            # Define feature types (customize based on your specific dataset)
            numerical_features = ['Age_Years', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Pulse_Pressure']
            categorical_features = ['Sex', 'Cholesterol_Level', 'Glucose_Level', 
                                   'Smoking_Status', 'Alcohol_Intake', 'Physical_Activity']

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler())  # Better for data with outliers
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
                ]
            )

            logging.info("Numerical and categorical preprocessing pipelines created")

            # Combine pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ],
                remainder='drop'  # Drop any columns not specified
            )

            return preprocessor
        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        """Apply transformations to the train and test data."""
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Obtain preprocessor
            preprocessor = self.get_data_transformer_object()
            logging.info("Obtained preprocessing object")

            target_column = 'target'
            
            # Split features and target from train and test data
            if target_column not in train_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in training data")
                
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info("Split input and target features from train and test data")
                        
            # Apply preprocessing
            logging.info("Applying preprocessing on training and testing datasets")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            # Convert to numpy arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save preprocessor object
            with open(self.transformation_config.preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            logging.info(f"Saved preprocessing object at {self.transformation_config.preprocessor_path}")
            
            # Save transformed data
            np.savez(self.transformation_config.transformed_train_path, data=train_arr)
            np.savez(self.transformation_config.transformed_test_path, data=test_arr)
            logging.info("Saved transformed training and testing data")
            
            return (
                self.transformation_config.transformed_train_path,
                self.transformation_config.transformed_test_path,
                self.transformation_config.preprocessor_path
            )
            
        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    # This allows standalone testing of data transformation
    from src.components.data_ingestion import DataIngestion
    
    logging.info("--- Starting Data Transformation standalone test ---")
    
    # First run data ingestion to get train and test data paths
    try:
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        
        # Now perform transformation
        transformation = DataTransformation()
        transformed_train_path, transformed_test_path, preprocessor_path = (
            transformation.initiate_data_transformation(train_path, test_path)
        )
        
        print(f"Data transformation successful.")
        print(f"Transformed train data saved at: {transformed_train_path}")
        print(f"Transformed test data saved at: {transformed_test_path}")
        print(f"Preprocessor saved at: {preprocessor_path}")
        
    except CustomException as e:
        print(f"Error occurred: {e}")
    
    logging.info("--- Data Transformation standalone test finished ---\n")