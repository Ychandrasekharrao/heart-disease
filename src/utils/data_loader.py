# ======================================================================
# UTILITY SCRIPT: DATA LOADER
# DESCRIPTION: Central functions for loading and saving data for the project.
# VERSION: 2.0
# ======================================================================

import pandas as pd
from pathlib import Path
import os
from datetime import datetime
from typing import Optional, Union, Tuple

class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass

def get_project_root() -> Path:
    """
    Finds the project root directory by searching upwards for a 'data' folder.
    Makes the script runnable from any sub-directory.
    
    Returns:
        Path: Project root directory path
    
    Raises:
        FileNotFoundError: If data directory cannot be found
    """
    try:
        current_path = Path(__file__).resolve()
    except NameError:
        current_path = Path(os.getcwd()).resolve()
    
    project_root = current_path
    while not (project_root / "data").exists() and project_root != project_root.parent:
        project_root = project_root.parent
    
    if not (project_root / "data").exists():
        raise FileNotFoundError("Could not find 'data' directory in parent path")
    
    return project_root

def load_raw_data(filename: str = "heart disease.csv") -> pd.DataFrame:
    """
    Loads the raw data from the 'data/raw' directory and performs initial cleaning.
    
    Args:
        filename (str): Name of the raw data file to load
        
    Returns:
        pd.DataFrame: Loaded and initially cleaned DataFrame
        
    Raises:
        DataLoadError: If file cannot be loaded or is empty
    """
    project_root = get_project_root()
    file_path = project_root / "data" / "raw" / filename
    
    try:
        print(f"Loading raw data from: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")
            
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        if df.empty:
            raise DataLoadError("Loaded DataFrame is empty")
            
        # Initial cleaning
        if 'id' in df.columns:
            df = df.drop('id', axis=1, errors='ignore')
            
        return df
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load raw data: {str(e)}")
        return pd.DataFrame()

def save_as_parquet(
    df: pd.DataFrame, 
    filename: str, 
    sub_dir: str = "raw", 
    overwrite: bool = True
) -> Tuple[bool, Path]:
    """
    Saves a DataFrame to a specified subdirectory within the data folder as a Parquet file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of the output file
        sub_dir (str): Subdirectory within data folder to save to
        overwrite (bool): If True, overwrites existing file. If False, creates versioned file
        
    Returns:
        Tuple[bool, Path]: Success status and path where file was saved
        
    Raises:
        ValueError: If DataFrame is empty or filename is invalid
    """
    if df.empty:
        raise ValueError("Cannot save empty DataFrame")
        
    if not filename.endswith('.parquet'):
        filename = filename + '.parquet'
    
    project_root = get_project_root()
    output_dir = project_root / "data" / sub_dir
    
    # Create directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    # Handle existing file
    if output_path.exists():
        if overwrite:
            output_path.unlink()
            print(f"[INFO] Overwriting existing file: {output_path.resolve()}")
        else:
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_parts = filename.rsplit('.', 1)
            new_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
            output_path = output_dir / new_filename
            print(f"[INFO] Creating new versioned file: {output_path.resolve()}")

    try:
        df.to_parquet(output_path, index=False)
        print(f"✅ Successfully saved DataFrame to: {output_path.resolve()}")
        return True, output_path
    except Exception as e:
        print(f"❌ Error saving Parquet file: {e}")
        return False, output_path

def save_processed_data(
    df: pd.DataFrame, 
    filename: str = "processed_heart_disease.csv", 
    overwrite: bool = True
) -> Tuple[bool, Path]:
    """
    Saves a DataFrame to the 'data/processed' directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of the output file
        overwrite (bool): If True, overwrites existing file. If False, creates versioned file
        
    Returns:
        Tuple[bool, Path]: Success status and path where file was saved
    """
    if df.empty:
        raise ValueError("Cannot save empty DataFrame")
    
    project_root = get_project_root()
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    # Handle existing file
    if output_path.exists():
        if overwrite:
            output_path.unlink()
            print(f"[INFO] Overwriting existing file: {output_path.resolve()}")
        else:
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_parts = filename.rsplit('.', 1)
            new_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
            output_path = output_dir / new_filename
            print(f"[INFO] Creating new versioned file: {output_path.resolve()}")

    try:
        if filename.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif filename.endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
            
        print(f"✅ Successfully saved DataFrame to: {output_path.resolve()}")
        return True, output_path
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False, output_path

def load_processed_data(filename: str = "processed_heart_disease.csv") -> pd.DataFrame:
    """
    Loads the final processed data from the 'data/processed' directory.
    
    Args:
        filename (str): Name of the processed data file to load
        
    Returns:
        pd.DataFrame: Loaded processed DataFrame
        
    Raises:
        DataLoadError: If file cannot be loaded or is empty
    """
    project_root = get_project_root()
    file_path = project_root / "data" / "processed" / filename
    
    try:
        print(f"Loading processed data from: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {file_path}")
            
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        if df.empty:
            raise DataLoadError("Loaded DataFrame is empty")
            
        return df
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load processed data: {str(e)}")
        return pd.DataFrame()

# --- Example usage demonstrating all features ---
if __name__ == "__main__":
    # Load raw CSV data
    raw_df = load_raw_data("heart disease.csv")
    
    if not raw_df.empty:
        # Save with overwrite=True (default behavior)
        success, path = save_as_parquet(
            raw_df, 
            "heart disease.parquet", 
            sub_dir="raw", 
            overwrite=True
        )
        
        # Save with overwrite=False (creates versioned file)
        success, path = save_as_parquet(
            raw_df, 
            "heart disease.parquet", 
            sub_dir="raw", 
            overwrite=False
        )
        
        # Process the data (example)
        processed_df = raw_df.copy()
        
        # Save processed data with versioning
        success, path = save_processed_data(
            processed_df,
            "processed_heart_disease.parquet",
            overwrite=False
        )
        
        # Load and verify processed data
        loaded_df = load_processed_data("processed_heart_disease.parquet")
        if not loaded_df.empty:
            print("✅ Data loading pipeline verified successfully")