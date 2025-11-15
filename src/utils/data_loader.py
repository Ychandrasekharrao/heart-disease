# data_loader.py
"""
CSV to Parquet Converter - Handles semicolon-delimited files
"""

import pandas as pd
from pathlib import Path


def load_data(file_path):
    """Load CSV (with semicolon or comma) or Parquet file"""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.csv':
        # Try semicolon delimiter first (Kaggle cardiovascular dataset)
        try:
            df = pd.read_csv(file_path, sep=';')
            if len(df.columns) == 1:  # Failed - try comma
                df = pd.read_csv(file_path, sep=',')
        except:
            df = pd.read_csv(file_path)  # Default pandas behavior
            
    elif file_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Drop ID column if exists (case-insensitive)
    id_cols = [col for col in df.columns if col.lower() == 'id']
    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"✅ Dropped '{id_cols[0]}' column")
    
    print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def save_parquet(df, output_path):
    """Save DataFrame as Parquet"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved to: {output_path}")


# Demo
if __name__ == "__main__":
    # Load CSV (handles both ; and , delimiters)
    df = load_data("data/raw/raw.csv")
    
    # Display info
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    # Save as Parquet
    save_parquet(df, "data/raw/raw.parquet")
