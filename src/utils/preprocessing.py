"""
Data Preprocessing - Robust Scaling + Train/Test Split
Script only - no saving of processed data
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from pathlib import Path


def get_project_root() -> Path:
    """Find project root directory."""
    current = Path.cwd().resolve()
    max_depth = 5
    depth = 0
    
    while depth < max_depth and current != current.parent:
        if (current / 'artifacts').exists():
            return current
        current = current.parent
        depth += 1
    
    return Path.cwd()


def create_preprocessor():
    """
    Creates preprocessor with RobustScaler for numerical features.
    Categorical features (already encoded as int) pass through.
    """
    
    numerical_features = [
        'Systolic_BP',
        'Diastolic_BP',
        'Age_Years',
        'BMI',
        'Systolic_Age_risk',
        'Diastolic_Age_risk',
        'Metabolic_Syndrome_Score',
        'Low_Risk_Paradox_Score'
    ]
    
    categorical_features = [
        'Sex',
        'Cholesterol_Level',
        'Glucose_Level',
        'Smoking_Status',
        'Alcohol_Intake',
        'Physical_Activity'
    ]
    
    # RobustScaler for numerical (resistant to outliers)
    # Passthrough for categorical (already encoded)
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numerical_features),
        ('cat', 'passthrough', categorical_features)
    ], remainder='drop')
    
    return preprocessor


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DATA PREPROCESSING (Robust Scaling)")
    print("="*80)
    
    # Setup paths
    PROJECT_ROOT = get_project_root()
    DATA_PATH = PROJECT_ROOT / 'data' / 'processed'
    ARTIFACTS_PATH = PROJECT_ROOT / 'artifacts' / 'split data sets'
    
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Path: {DATA_PATH}")
    
    try:
        # Load data
        print("\n[1/3] Loading processed data...")
        processed_data = pd.read_parquet(DATA_PATH / 'processed_data.parquet')
        print(f"✓ Loaded: {processed_data.shape}")
        
        # Split features and target
        X = processed_data.drop(columns=['target'])
        y = processed_data['target']
        
        print(f"✓ Features: {X.shape}")
        print(f"✓ Target prevalence: {y.mean():.1%}")
        
        # Train/test split (80/20)
        print("\n[2/3] Creating train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Train: {X_train.shape[0]:,} samples ({y_train.mean():.1%} disease)")
        print(f"✓ Test:  {X_test.shape[0]:,} samples ({y_test.mean():.1%} disease)")
        
        # Save ONLY train/test splits (no preprocessor, no scaled data)
        print("\n[3/3] Saving train/test splits...")
        ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
        
        X_train.to_parquet(ARTIFACTS_PATH / 'X_train.parquet')
        X_test.to_parquet(ARTIFACTS_PATH / 'X_test.parquet')
        y_train.to_frame('target').to_parquet(ARTIFACTS_PATH / 'y_train.parquet')
        y_test.to_frame('target').to_parquet(ARTIFACTS_PATH / 'y_test.parquet')
        
        print(f"✓ X_train.parquet ({X_train.shape})")
        print(f"✓ X_test.parquet ({X_test.shape})")
        print(f"✓ y_train.parquet")
        print(f"✓ y_test.parquet")
        
        print("\n" + "="*80)
        print("✅ SPLITS SAVED")
        print("="*80)
        print(f"\nSaved to: {ARTIFACTS_PATH}")
        print(f"Features: {list(X_train.columns)}")
        print("\n📝 Note: No preprocessor or scaled data saved")
        print("   Tree models will use raw features directly")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
