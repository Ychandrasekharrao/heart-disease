# ======================================================================
# UTILITY SCRIPT: DATA LOADER
# DESCRIPTION: Central functions for loading and saving data for the project.
# ======================================================================

import pandas as pd
from pathlib import Path
import os

def get_project_root() -> Path:
    """
    Finds the project root directory by searching upwards for a 'data' folder.
    This makes the script runnable from any sub-directory.
    """
    try:
        current_path = Path(__file__).resolve()
    except NameError:
        current_path = Path(os.getcwd()).resolve()
    
    project_root = current_path
    while not (project_root / "data").exists() and project_root != project_root.parent:
        project_root = project_root.parent
    
    return project_root

def load_raw_data(filename: str = "heart disease.csv") -> pd.DataFrame:
    """
    Loads the raw data from the 'data/raw' directory and performs initial cleaning.
    """
    project_root = get_project_root()
    file_path = project_root / "data" / "raw" / filename
    try:
        print(f"Loading raw data from: {file_path}")
        df = pd.read_csv(file_path)
        if 'id' in df.columns:
            df = df.drop('id', axis=1, errors='ignore')
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: Raw data file not found at {file_path}")
        return pd.DataFrame()

def save_as_parquet(df: pd.DataFrame, filename: str, sub_dir: str = "raw"):
    """
    Saves a DataFrame to a specified subdirectory within the data folder as a Parquet file.
    Overwrites the file if it already exists.
    """
    project_root = get_project_root()
    output_dir = project_root / "data" / sub_dir
    
    # Only create folder if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"[INFO] Created folder: {output_dir.resolve()}")
    else:
        print(f"[INFO] Folder already exists: {output_dir.resolve()}")

    output_path = output_dir / filename

    # Remove existing file if it exists to overwrite
    if output_path.exists():
        output_path.unlink()
        print(f"[INFO] Existing file removed: {output_path.resolve()}")

    try:
        df.to_parquet(output_path, index=False)
        print(f"✅ Successfully saved DataFrame to Parquet: {output_path.resolve()}")
    except Exception as e:
        print(f"❌ Error saving Parquet file: {e}")

def save_processed_data(df: pd.DataFrame, filename: str = "processed_heart_disease.csv"):
    """
    Saves a DataFrame to the 'data/processed' directory as a CSV.
    Overwrites the file if it already exists.
    """
    project_root = get_project_root()
    output_dir = project_root / "data" / "processed"

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"[INFO] Created folder: {output_dir.resolve()}")
    else:
        print(f"[INFO] Folder already exists: {output_dir.resolve()}")

    output_path = output_dir / filename

    if output_path.exists():
        output_path.unlink()
        print(f"[INFO] Existing file removed: {output_path.resolve()}")

    try:
        df.to_csv(output_path, index=False)
        print(f"✅ Data saved successfully to: {output_path.resolve()}")
    except Exception as e:
        print(f"❌ Error saving data: {e}")

def load_processed_data(filename: str = "processed_heart_disease.csv") -> pd.DataFrame:
    """
    Loads the final processed data from the 'data/processed' directory.
    """
    project_root = get_project_root()
    file_path = project_root / "data" / "processed" / filename
    try:
        print(f"Loading processed data from: {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Processed data file not found at {file_path}")
        return pd.DataFrame()

# --- Example usage: convert raw CSV to Parquet, overwrite if already exists ---
if __name__ == "__main__":
    raw_df = load_raw_data("heart disease.csv")
    if not raw_df.empty:
        save_as_parquet(raw_df, "heart_disease.parquet", sub_dir="raw")