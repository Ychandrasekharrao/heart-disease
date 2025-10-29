import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path

def create_preprocessor():
    """Creates preprocessor for heart disease prediction dataset."""
    
    numerical_features = ['Systolic_BP', 'Diastolic_BP', 'Age_Years', 'Pulse_Pressure', 'BMI']
    ordinal_features = ['Glucose_Level', 'Cholesterol_Level']
    nominal_features = [
        'Sex', 'Smoking_Status', 'Alcohol_Intake', 'Physical_Activity', 
        'MetabolicRisk', 'LifestyleRisk', 'HiddenStrain', 'AgeRisk', 
        'Young_WithMetabolic', 'NormalBP_WithAbnormalPP',
        'NormalWeight_WithMetabolic', 'NormalWeight_WithAbnormalPP'
    ]
    
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numerical_features),
        ('ord', OrdinalEncoder(
            categories=[['Normal', 'Prediabetes', 'Diabetes'], 
                       ['Normal', 'Borderline High', 'High']],
            handle_unknown='use_encoded_value', unknown_value=-1
        ), ordinal_features),
        ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nominal_features)
    ], remainder='drop')
    
    return preprocessor

def fit_preprocessor(X_train, artifacts_path: Path):
    """Fits preprocessor and saves it."""
    preprocessor = create_preprocessor()
    X_processed = preprocessor.fit_transform(X_train)
    
    # Create DataFrame with proper column names
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X_train.index)
    
    # Ensure numeric types
    for col in X_processed_df.columns:
        X_processed_df[col] = pd.to_numeric(X_processed_df[col], errors='coerce')
    
    # Save preprocessor
    artifacts_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifacts_path / 'preprocessor.joblib')
    
    return preprocessor, X_processed_df

if __name__ == '__main__':
    # Setup paths
    current_dir = Path.cwd()
    if current_dir.name == 'notebooks' and current_dir.parent.name == 'src':
        project_root = current_dir.parent.parent
    elif current_dir.name == 'src':
        project_root = current_dir.parent
    else:
        project_root = current_dir
    
    artifacts_path = project_root / 'artifacts'
    
    try:
        # Load and process training data
        X_train = pd.read_parquet(artifacts_path / 'X_train.parquet')
        print(f"Loaded training data: {X_train.shape}")
        
        preprocessor, X_train_processed = fit_preprocessor(X_train, artifacts_path)
        print(f"Processed training data: {X_train_processed.shape}")
        
        # Process test data
        X_test = pd.read_parquet(artifacts_path / 'X_test.parquet')
        X_test_processed = preprocessor.transform(X_test)
        X_test_processed_df = pd.DataFrame(
            X_test_processed, 
            columns=preprocessor.get_feature_names_out(), 
            index=X_test.index
        )
        
        for col in X_test_processed_df.columns:
            X_test_processed_df[col] = pd.to_numeric(X_test_processed_df[col], errors='coerce')
        
        # Save processed data
        X_train_processed.to_parquet(artifacts_path / 'X_train_processed.parquet')
        X_test_processed_df.to_parquet(artifacts_path / 'X_test_processed.parquet')
        
        print(f"✅ Preprocessing completed. Files saved to {artifacts_path}")
        print(f"Final feature count: {X_train_processed.shape[1]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")