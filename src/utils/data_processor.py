import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
from .data_loader import load_raw_data, save_processed_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data():
    try:
        # Load the dataset
        df = load_raw_data("heart disease.csv")
        if df.empty:
            raise ValueError("Loaded dataset is empty.")
        logging.info(f"Loaded raw data with shape: {df.shape}")

        # Drop 'id' if exists
        if "id" in df.columns:
            df = df.drop("id", axis=1)
            logging.info("Dropped 'id' column.")

        # Rename columns
        column_mapping = {
            "age": "Age",
            "gender": "Sex",
            "height": "Height",
            "weight": "Weight",
            "ap_hi": "Systolic_BP",
            "ap_lo": "Diastolic_BP",
            "cholesterol": "Cholesterol_Level",
            "gluc": "Glucose_Level",
            "smoke": "Smoking_Status",
            "alco": "Alcohol_Intake",
            "active": "Physical_Activity",
            "cardio": "target",
        }
        df = df.rename(columns=column_mapping)
        logging.info("Renamed columns.")

        # Validate required columns exist
        required_cols = list(column_mapping.values())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns after renaming: {missing_cols}")

        # Convert to appropriate dtypes
        cat_cols = [
            "Sex",
            "Cholesterol_Level",
            "Glucose_Level",
            "Smoking_Status",
            "Alcohol_Intake",
            "Physical_Activity",
            "target",
        ]
        num_cols = ["Age", "Height", "Weight", "Systolic_BP", "Diastolic_BP"]

        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        logging.info("Converted data types.")

        # Remove duplicates
        initial_shape = df.shape
        df = df.drop_duplicates()
        if df.shape[0] < initial_shape[0]:
            logging.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

        # Convert age to years
        if "Age" in df.columns:
            df["Age_Years"] = round(df["Age"] / 365, 0)
            df.drop("Age", axis=1, inplace=True)
            logging.info("Converted age to years.")

        # Convert height to meters
        if "Height" in df.columns:
            df["Height_mt"] = df["Height"] / 100
            df.drop("Height", axis=1, inplace=True)
            logging.info("Converted height to meters.")

        # Apply cleaning rules for BP
        initial_shape = df.shape
        df = df[
            (df["Systolic_BP"] >= 90)
            & (df["Systolic_BP"] <= 250)
            & (df["Diastolic_BP"] >= 60)
            & (df["Diastolic_BP"] <= 150)
            & (df["Diastolic_BP"] <= df["Systolic_BP"])
        ]
        if df.shape[0] < initial_shape[0]:
            logging.info(f"Filtered {initial_shape[0] - df.shape[0]} rows based on BP rules.")

        # Calculate BMI
        if "Weight" in df.columns and "Height_mt" in df.columns:
            df["BMI"] = df["Weight"] / (df["Height_mt"] ** 2)
            logging.info("Calculated BMI.")
        else:
            raise ValueError("Weight or Height_mt column missing for BMI calculation.")

        # Filter based on realistic ranges
        BMI_MIN, BMI_MAX = 15, 60
        HEIGHT_MIN, HEIGHT_MAX = 1.3, 2.1
        initial_shape = df.shape
        df = df[(df["Height_mt"] >= HEIGHT_MIN) & (df["Height_mt"] <= HEIGHT_MAX)].copy()
        df["BMI"] = df["Weight"] / (df["Height_mt"] ** 2)
        df = df[(df["BMI"] >= BMI_MIN) & (df["BMI"] <= BMI_MAX)]
        if df.shape[0] < initial_shape[0]:
            logging.info(f"Filtered {initial_shape[0] - df.shape[0]} rows based on realistic ranges.")

        # Check for nulls
        if df.isnull().sum().sum() > 0:
            logging.warning("Null values present after preprocessing. Filling with medians for numeric columns.")
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].median(), inplace=True)

        # Save processed data
        save_processed_data(df, "processed_heart_disease.csv")
        logging.info(f"Preprocessing completed. Final shape: {df.shape}")

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    preprocess_data()
            pass
        if df["Physical_Activity"].isnull().any():
            mode_val = df["Physical_Activity"].mode()[0]
            df["Physical_Activity"] = df["Physical_Activity"].fillna(mode_val)

    # Save processed data
    save_processed_data(df, "processed_heart_disease.csv")


if __name__ == "__main__":
    preprocess_data()
            pass
        if df["Physical_Activity"].isnull().any():
            mode_val = df["Physical_Activity"].mode()[0]
            df["Physical_Activity"] = df["Physical_Activity"].fillna(mode_val)

    # Save processed data
    output_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "processed"
        / "processed_heart_disease.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    preprocess_data()
