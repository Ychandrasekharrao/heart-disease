import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

class HeartDiseaseTrainer:
    def __init__(self, 
                 data_path=None, 
                 output_dir='../../models',
                 test_size=0.2, 
                 random_state=42):
        """
        Initialize the heart disease model trainer.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset (parquet format)
        output_dir : str
            Directory to save the trained pipeline
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize containers
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipeline = None
        
    def load_data(self):
        """Load and preprocess the heart disease dataset"""
        print(f"Loading data from {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
        # Load data based on file extension
        if self.data_path.endswith('.parquet'):
            self.df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use .parquet or .csv")
            
        # Drop id column if it exists
        if 'id' in self.df.columns:
            self.df = self.df.drop(columns=['id'])
            
        # Drop duplicates
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self
    
    def prepare_features(self):
        """Prepare and engineer features"""
        print("Preparing features...")
        
        # Assume target column is named 'target' or 'cardio'
        if 'target' in self.df.columns:
            target_col = 'target'
        elif 'cardio' in self.df.columns:
            target_col = 'cardio'
        else:
            raise ValueError("Target column not found. Expected 'target' or 'cardio'")
            
        # Feature engineering based on domain knowledge
        if 'Age' in self.df.columns and self.df['Age'].max() > 1000:
            # Convert age from days to years if needed
            self.df['Age_Years'] = (self.df['Age'] / 365.25).round().astype(int)
            self.df = self.df.drop(columns=['Age'])
            
        # Calculate BMI if height and weight are present
        if 'Height' in self.df.columns and 'Weight' in self.df.columns:
            self.df['BMI'] = self.df['Weight'] / (self.df['Height'] / 100) ** 2
            
        # Calculate Pulse Pressure if BP columns exist
        if 'Systolic_BP' in self.df.columns and 'Diastolic_BP' in self.df.columns:
            self.df['Pulse_Pressure'] = self.df['Systolic_BP'] - self.df['Diastolic_BP']
            
        # Handle any missing values
        for col in self.df.columns:
            if self.df[col].isna().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        # Identify feature types
        self.numerical_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.numerical_features = [f for f in self.numerical_features if f != target_col]
        
        self.categorical_features = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Split data
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Features prepared. Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        return self
    
    def build_pipeline(self, model_params=None):
        """Build the ML pipeline with preprocessing and model"""
        print("Building pipeline...")
        
        # Default XGBoost parameters if none provided
        if model_params is None:
            model_params = {
                'n_estimators': 400,
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.7,
                'colsample_bytree': 0.6,
                'random_state': self.random_state,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
        
        # Create preprocessor
        preprocessor = ColumnTransformer(transformers=[
            ('num', RobustScaler(), self.numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), self.categorical_features)
        ], remainder='drop')
        
        # Create pipeline with SMOTE for class imbalance
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=self.random_state)),
            ('classifier', XGBClassifier(**model_params))
        ])
        
        print("Pipeline built successfully")
        return self
        
    def train(self):
        """Train the model pipeline"""
        print("Training model...")
        start_time = datetime.now()
        
        self.pipeline.fit(self.X_train, self.y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"Model trained in {training_time:.2f} seconds")
        return self
    
    def evaluate(self):
        """Evaluate model performance"""
        print("\n--- Model Evaluation ---")
        
        # Make predictions
        y_pred = self.pipeline.predict(self.X_test)
        y_prob = self.pipeline.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(self.y_test, y_prob)
        
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        return self
    
    def save_pipeline(self, filename=None):
        """Save the trained pipeline to disk"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heart_disease_pipeline_{timestamp}.pkl"
        
        output_path = self.output_dir / filename
        
        print(f"Saving pipeline to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
            
        print(f"Pipeline saved successfully to {output_path}")
        return output_path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Heart Disease Model Trainer")
    
    parser.add_argument(
        "--data", "-d", 
        default="../../data/processed/heart_disease.parquet",
        help="Path to the dataset (.parquet or .csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="../../models",
        help="Directory to save the trained pipeline"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    try:
        trainer = HeartDiseaseTrainer(
            data_path=args.data,
            output_dir=args.output,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Execute training pipeline
        (trainer
            .load_data()
            .prepare_features()
            .build_pipeline()
            .train()
            .evaluate()
            .save_pipeline())
        
        print("✅ Training pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())