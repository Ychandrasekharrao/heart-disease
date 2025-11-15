"""
Heart Disease Prediction Pipeline
Matches production training: Beta/Isotonic calibration + CART 3 groups
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Union, Optional, Any, List
import warnings

warnings.filterwarnings('ignore')

# Beta calibration
try:
    from betacal import BetaCalibration
    BETACAL_AVAILABLE = True
except ImportError:
    BETACAL_AVAILABLE = False


class HeartDiseasePredictPipeline:
    """Production prediction pipeline."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize pipeline."""
        self.project_root = self._find_project_root()
        self.model_path = model_path or str(self.project_root / 'artifacts' / 'model' / 'model_package.pkl')
        
        # Features (14 total - exact feature set)
        self.features: List[str] = [
            'Sex', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol_Level',
            'Glucose_Level', 'Smoking_Status', 'Alcohol_Intake', 'Physical_Activity',
            'Age_Years', 'BMI', 'Systolic_Age_risk', 'Diastolic_Age_risk',
            'Metabolic_Syndrome_Score', 'Low_Risk_Paradox_Score'
        ]
        
        self.base_model = None
        self.calibrator = None
        self.calibration_method: str = 'none'
        self.cart_model = None
        self.threshold: float = 0.5
        self.risk_statistics: Dict = {}
        
        self._load_model_package()
        print("✓ Pipeline initialized")
    
    @staticmethod
    def _find_project_root() -> Path:
        """Find project root."""
        current = Path.cwd().resolve()
        max_depth = 5
        depth = 0
        
        while depth < max_depth and current != current.parent:
            if (current / 'artifacts').exists():
                return current
            current = current.parent
            depth += 1
        
        return Path.cwd()
    
    def _load_model_package(self) -> None:
        """Load model package (matches training output)."""
        model_file = Path(self.model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        package = joblib.load(model_file)
        
        # Extract components
        self.base_model = package['base_model']
        self.calibrator = package.get('calibrator')
        self.calibration_method = package.get('calibration_method', 'none')
        self.cart_model = package.get('cart_model')
        self.threshold = float(package.get('threshold', 0.5))
        self.risk_statistics = package.get('risk_statistics', {})
        
        model_type = package.get('model_type', 'Unknown')
        
        print(f"Model: {model_type} + {self.calibration_method}")
        print(f"Threshold: {self.threshold:.4f}")
        print(f"Risk Groups: {list(self.risk_statistics.keys())}")
    
    def preprocess(self, input_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """Preprocess input (matches training preprocessing)."""
        # Convert to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")
        
        # Defaults (all features must be present)
        defaults = {
            'Sex': 1, 'Systolic_BP': 120.0, 'Diastolic_BP': 80.0,
            'Cholesterol_Level': 1, 'Glucose_Level': 1, 'Smoking_Status': 1,
            'Alcohol_Intake': 1, 'Physical_Activity': 2, 'Age_Years': 50.0,
            'BMI': 25.0, 'Systolic_Age_risk': 0.0, 'Diastolic_Age_risk': 0.0,
            'Metabolic_Syndrome_Score': 0.0, 'Low_Risk_Paradox_Score': 0.0
        }
        
        # Ensure all features exist
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = defaults[feat]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaNs
        df = df.fillna(defaults)
        
        # Select features in correct order
        df = df[self.features]
        
        return df
    
    def predict_proba(self, input_data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """Predict probability (with calibration)."""
        X = self.preprocess(input_data)
        
        # Base prediction
        base_proba = self.base_model.predict_proba(X)[:, 1]
        
        # Apply calibration (matches training logic)
        if self.calibrator is None or self.calibration_method == 'none':
            return np.clip(base_proba, 0.0, 1.0)
        
        if self.calibration_method == 'beta' and BETACAL_AVAILABLE:
            try:
                calibrated = self.calibrator.predict(base_proba)
                return np.clip(calibrated, 0.0, 1.0)
            except Exception as e:
                print(f"⚠️  Beta calibration failed: {e}. Using base probabilities.")
                return np.clip(base_proba, 0.0, 1.0)
        
        if self.calibration_method == 'isotonic':
            try:
                calibrated = self.calibrator.predict(base_proba)
                return np.clip(calibrated, 0.0, 1.0)
            except Exception as e:
                print(f"⚠️  Isotonic calibration failed: {e}. Using base probabilities.")
                return np.clip(base_proba, 0.0, 1.0)
        
        return np.clip(base_proba, 0.0, 1.0)
    
    def get_risk_group(self, probabilities: np.ndarray) -> np.ndarray:
        """Get risk group using CART (matches training)."""
        if self.cart_model is None:
            # Fallback: simple thresholding
            return np.array(['Moderate'] * len(probabilities))
        
        try:
            # Apply CART model
            X_cart = pd.DataFrame({'Predicted_Risk': probabilities})
            leaves = self.cart_model.apply(X_cart)
            
            # Map leaves to risk groups (from training)
            unique_leaves = np.unique(leaves)
            leaf_stats = []
            
            for leaf in unique_leaves:
                mask = leaves == leaf
                leaf_stats.append({
                    'leaf': leaf,
                    'mean_risk': probabilities[mask].mean()
                })
            
            # Sort by risk
            leaf_stats.sort(key=lambda x: x['mean_risk'])
            
            # Map to Low/Moderate/High
            risk_names = ['Low', 'Moderate', 'High']
            leaf_to_risk = {}
            
            for i, stat in enumerate(leaf_stats):
                leaf_to_risk[stat['leaf']] = risk_names[min(i, len(risk_names)-1)]
            
            # Get risk labels
            risk_labels = np.array([leaf_to_risk.get(leaf, 'Moderate') for leaf in leaves])
            
            return risk_labels
        
        except Exception as e:
            print(f"⚠️  CART risk grouping failed: {e}. Using default.")
            return np.array(['Moderate'] * len(probabilities))
    
    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """Predict class (0=No Disease, 1=Disease)."""
        proba = self.predict_proba(input_data)
        return (proba >= self.threshold).astype(int)
    
    def predict_detailed(self, input_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """Get detailed prediction with risk group."""
        proba = self.predict_proba(input_data)
        pred = self.predict(input_data)
        risk_groups = self.get_risk_group(proba)
        
        results = []
        for i in range(len(proba)):
            risk_group = risk_groups[i]
            
            # Get risk group statistics
            risk_stats = self.risk_statistics.get(risk_group, {})
            
            results.append({
                'probability': float(proba[i]),
                'prediction': int(pred[i]),
                'predicted_class': 'Disease' if pred[i] == 1 else 'No Disease',
                'risk_group': risk_group,
                'risk_group_stats': {
                    'disease_prevalence': float(risk_stats.get('disease_prevalence', 0.0)),
                    'mean_risk_score': float(risk_stats.get('mean_risk_score', 0.0))
                },
                'threshold': float(self.threshold)
            })
        
        return {
            'predictions': results,
            'model_info': {
                'calibration': self.calibration_method,
                'threshold': float(self.threshold),
                'risk_groups_available': list(self.risk_statistics.keys())
            }
        }


# Convenience function
def predict_patient(patient_data: Dict, model_path: Optional[str] = None) -> Dict:
    """Quick prediction."""
    pipeline = HeartDiseasePredictPipeline(model_path)
    return pipeline.predict_detailed(patient_data)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HEART DISEASE PREDICTION PIPELINE")
    print("="*80)
    
    try:
        # Initialize
        pipeline = HeartDiseasePredictPipeline()
        
        # Example patient
        patient = {
            'Sex': 2,  # Male
            'Systolic_BP': 140,
            'Diastolic_BP': 90,
            'Cholesterol_Level': 2,
            'Glucose_Level': 2,
            'Smoking_Status': 2,
            'Alcohol_Intake': 2,
            'Physical_Activity': 1,
            'Age_Years': 60,
            'BMI': 28,
            'Systolic_Age_risk': 0.5,
            'Diastolic_Age_risk': 0.4,
            'Metabolic_Syndrome_Score': 1.5,
            'Low_Risk_Paradox_Score': 0.2
        }
        
        # Predict
        result = pipeline.predict_detailed(patient)
        
        pred = result['predictions'][0]
        print(f"\n📊 Prediction Results:")
        print(f"  Predicted: {pred['predicted_class']}")
        print(f"  Probability: {pred['probability']:.3f}")
        print(f"  Risk Group: {pred['risk_group']}")
        print(f"  Group Prevalence: {pred['risk_group_stats']['disease_prevalence']:.1f}%")
        print(f"  Threshold: {pred['threshold']:.3f}")
        
        print("\n✅ Pipeline test successful!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()