# data_loader.py

"""
Data Loader Utility for Heart Disease Dataset

Simple utility functions for loading and saving data files with versioning support.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os


def load_raw_data(file_path, verbose=False):
    """
    Load raw data from various file formats with basic validation
    
    Args:
        file_path (str): Path to the data file
        verbose (bool): Whether to print detailed information
        
    Returns:
        pd.DataFrame: Loaded data or empty DataFrame if failed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()
    
    try:
        if file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
            if verbose:
                print(f"✅ Loaded {len(df):,} rows from parquet file")
                
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            if verbose:
                print(f"✅ Loaded {len(df):,} rows from CSV file")
                
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
            if verbose:
                print(f"✅ Loaded {len(df):,} rows from JSON file")
                
        else:
            print(f"❌ Unsupported file format: {file_path.suffix}")
            return pd.DataFrame()
        
        if verbose:
            print(f"📊 Data shape: {df.shape}")
            print(f"🔢 Columns: {', '.join(df.columns)}")
            print(f"📅 Data types:\n{df.dtypes}")
            
        return df
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return pd.DataFrame()


def save_as_parquet(df, filename, sub_dir="raw", overwrite=False, verbose=False):
    """
    Save DataFrame as parquet file with versioning support
    
    Args:
        df (pd.DataFrame): Data to save
        filename (str): Output filename
        sub_dir (str): Subdirectory under 'data' folder
        overwrite (bool): Whether to overwrite existing file
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (success bool, saved_file_path)
    """
    try:
        # Create data directory structure
        data_dir = Path("data") / sub_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = data_dir / filename
        
        if file_path.exists() and not overwrite:
            # Create versioned filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_stem = file_path.stem
            versioned_filename = f"{name_stem}_{timestamp}{file_path.suffix}"
            file_path = data_dir / versioned_filename
            
            if verbose:
                print(f"📁 File exists, saving as: {versioned_filename}")
        
        # Save the file
        df.to_parquet(file_path, index=False)
        
        if verbose:
            print(f"✅ Successfully saved data to: {file_path}")
            print(f"📊 Data shape: {df.shape}")
            print(f"💾 File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
            
        return True, file_path
        
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False, None


def load_processed_data(filename, sub_dir="processed", verbose=False):
    """
    Load processed data from parquet files
    
    Args:
        filename (str): Name of the file to load
        sub_dir (str): Subdirectory under 'data' folder
        verbose (bool): Whether to print detailed information
        
    Returns:
        pd.DataFrame: Loaded data or empty DataFrame if failed
    """
    try:
        file_path = Path("data") / sub_dir / filename
        
        if not file_path.exists():
            print(f"❌ Processed data file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(file_path)
        
        if verbose:
            print(f"✅ Loaded processed data from: {file_path}")
            print(f"📊 Data shape: {df.shape}")
            print(f"🔢 Columns: {', '.join(df.columns)}")
            
        return df
        
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        return pd.DataFrame()


def save_processed_data(df, filename, sub_dir="processed", overwrite=False, verbose=False):
    """
    Save processed data with versioning support
    
    Args:
        df (pd.DataFrame): Processed data to save
        filename (str): Output filename
        sub_dir (str): Subdirectory under 'data' folder
        overwrite (bool): Whether to overwrite existing file
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (success bool, saved_file_path)
    """
    return save_as_parquet(df, filename, sub_dir, overwrite, verbose)


def get_data_info(df, detailed=False):
    """
    Get basic information about the dataset
    
    Args:
        df (pd.DataFrame): Data to analyze
        detailed (bool): Whether to show detailed information
        
    Returns:
        dict: Data information
    """
    if df.empty:
        return {"error": "DataFrame is empty"}
    
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    if detailed:
        info.update({
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "duplicate_rows": df.duplicated().sum(),
            "description": df.describe().to_dict()
        })
    
    return info

# Demonstration of the utility functions
if __name__ == "__main__":
    print("🔍 Data Loader Utility Demonstration")
    print("="*50)
    
    # Step 1: Loading Raw Data
    print("\n--- Step 1: Loading Raw Data ---")
    raw_df = load_raw_data("heart disease.parquet", verbose=True)
    
    if raw_df.empty:
        print("❌ Could not proceed because raw data failed to load.")
        exit(1)
    else:
        print(f"✅ Successfully loaded raw data with shape: {raw_df.shape}\n")

    # Step 2: Saving Raw Data as Parquet (Versioning Demonstration)
    print("--- Step 2: Saving Raw Data as Parquet (Versioning Demonstration) ---")
    # Save with overwrite=True
    success1, path1 = save_as_parquet(raw_df, "heart_disease_raw.parquet", sub_dir="raw", overwrite=True, verbose=True)
    # Save with overwrite=False to create a versioned file
    success2, path2 = save_as_parquet(raw_df, "heart_disease_raw.parquet", sub_dir="raw", overwrite=False, verbose=True)

    # Step 3: Basic Data Processing
    print("\n--- Step 3: Processing Data ---")
    processed_df = raw_df.copy()
    
    # Simple processing examples
    initial_shape = processed_df.shape
    processed_df = processed_df.drop_duplicates().reset_index(drop=True)
    processed_df = processed_df.dropna()  # Simple NA handling for demo
    
    print(f"📊 Shape after removing duplicates: {processed_df.shape}")
    print(f"🗑️  Removed {initial_shape[0] - processed_df.shape[0]} duplicate rows")
    print(f"🗑️  Removed {initial_shape[0] - processed_df.shape[0]} rows with NA values")

    # Step 4: Saving Processed Data
    print("\n--- Step 4: Saving Processed Data ---")
    success, saved_path = save_processed_data(processed_df, "processed_heart_disease.parquet", overwrite=False, verbose=True)
    
    if success:
        print(f"✅ Processed data saved to: {saved_path}")

        # Step 5: Loading the Saved Processed Data
        print("\n--- Step 5: Loading the Saved Processed Data ---")
        loaded_df = load_processed_data(saved_path.name, verbose=True)
        
        if not loaded_df.empty:
            print(f"✅ Successfully verified pipeline. Loaded data shape: {loaded_df.shape}")
            
            # Step 6: Data Information
            print("\n--- Step 6: Data Information ---")
            info = get_data_info(loaded_df, detailed=True)
            print(f"📊 Shape: {info['shape']}")
            print(f"💾 Memory usage: {info['memory_usage_mb']:.2f} MB")
            print(f"🔢 Columns ({len(info['columns'])}): {', '.join(info['columns'])}")
            print(f"📈 Numeric columns: {', '.join(info['numeric_columns'])}")
            print(f"🏷️  Categorical columns: {', '.join(info['categorical_columns'])}")
            print(f"🔍 Duplicate rows: {info['duplicate_rows']}")
            
        else:
            print("❌ Loaded processed data is empty.")
    else:
        print("❌ Failed to save processed data.")

    print("\n" + "="*50)
    print("🎉 DATA LOADER DEMONSTRATION COMPLETED")
    print("="*50)
    
    # Additional usage examples
    print("\n💡 Additional Usage Examples:")
    print("""
    # Basic loading
    df = load_raw_data("your_data.csv", verbose=True)
    
    # Saving with versioning
    success, path = save_as_parquet(df, "data.parquet", overwrite=False)
    
    # Loading processed data
    processed = load_processed_data("processed_data.parquet")
    
    # Get data information
    info = get_data_info(df, detailed=True)
    """)