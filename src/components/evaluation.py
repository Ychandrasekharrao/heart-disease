# src/components/evaluation.py
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
import json
import warnings
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss, average_precision_score,
    roc_curve, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy import stats
import shap

# Beta Calibration
try:
    from betacal import BetaCalibration
    BETACAL_AVAILABLE = True
except Exception:
    BETACAL_AVAILABLE = False
    print("⚠️  Beta calibration not available")

warnings.filterwarnings('ignore')

@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    model_dir: Path = Path("artifacts/model")
    data_dir: Path = Path("artifacts/split data sets")
    reports_dir: Path = Path("reports/evaluation")
    figures_dir: Path = Path("reports/evaluation/figures")
    metrics_dir: Path = Path("reports/evaluation/metrics")
    random_state: int = 42
    n_bootstraps: int = 500
    confidence_level: float = 0.95
    shap_sample_size: int = 6000


class ModelEvaluator:
    """Model evaluation with clinical focus."""
    
    def __init__(self):
        self.config = EvaluationConfig()
        
        for directory in [self.config.reports_dir, self.config.figures_dir, self.config.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.model_package: Optional[Dict] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[np.ndarray] = None
        self.y_proba: Optional[np.ndarray] = None
        
        print("ModelEvaluator initialized")
    
    def _safe_confusion(self, y_true, y_pred) -> Tuple[int, int, int, int]:
        """Return tn, fp, fn, tp safely."""
        try:
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            
            if len(y_true) == 0:
                return 0, 0, 0, 0
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                return int(tn), int(fp), int(fn), int(tp)
            
            # Fallback
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            return tn, fp, fn, tp
        except:
            return 0, 0, 0, 0
    
    def load_model_and_data(self) -> bool:
        """Load model and test data."""
        model_path = self.config.model_dir / 'model_package.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model_package = joblib.load(model_path)
        print(f"Loaded: {self.model_package.get('model_type', 'Unknown')}")
        
        # Load and split test data
        X_test_full = pd.read_parquet(self.config.data_dir / 'X_test.parquet')
        y_test_full = pd.read_parquet(self.config.data_dir / 'y_test.parquet').iloc[:, 0].values
        
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_test_full, y_test_full, test_size=0.5,
            stratify=y_test_full, random_state=self.config.random_state
        )
        
        self.X_test = X_test.reset_index(drop=True)
        self.y_test = np.asarray(y_test).ravel()
        
        print(f"Test set: {self.X_test.shape}, Prevalence: {np.mean(self.y_test):.1%}")
        return True
    
    def get_predictions(self) -> np.ndarray:
        """Get calibrated predictions."""
        base_model = self.model_package['base_model']
        calibrator = self.model_package.get('calibrator')
        calibration_method = self.model_package.get('calibration_method')
        
        # Base predictions
        y_proba_base = base_model.predict_proba(self.X_test)[:, 1]
        
        # Apply calibration (only Beta or Isotonic)
        if calibrator is not None:
            if calibration_method == 'beta' and BETACAL_AVAILABLE:
                y_proba = calibrator.predict(y_proba_base)
            elif calibration_method == 'isotonic':
                y_proba = calibrator.predict(y_proba_base)
            else:
                y_proba = y_proba_base
        else:
            y_proba = y_proba_base
        
        self.y_proba = np.clip(y_proba, 0.0, 1.0)
        print(f"Predictions using {calibration_method or 'base'} calibration")
        
        return self.y_proba
    
    def calculate_bca_ci(self, bootstrap_estimates: np.ndarray, observed_statistic: float,
                        confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate BCa confidence intervals."""
        try:
            bootstrap_estimates = np.asarray(bootstrap_estimates)
            if bootstrap_estimates.size == 0:
                return float(observed_statistic), float(observed_statistic)
            
            B = len(bootstrap_estimates)
            alpha = 1 - confidence_level
            bootstrap_sorted = np.sort(bootstrap_estimates)
            
            # Bias correction
            prop_less = np.mean(bootstrap_estimates < observed_statistic)
            prop_less = np.clip(prop_less, 1.0/(B+1), 1-1.0/(B+1))
            z0 = stats.norm.ppf(prop_less)
            z0 = np.clip(z0, -5, 5)
            
            # Acceleration
            jack_mean = np.mean(bootstrap_estimates)
            diff = jack_mean - bootstrap_estimates
            denom = np.sum(diff ** 2)
            denominator = 6 * (denom ** 1.5) if denom > 0 else np.nan
            numerator = np.sum(diff ** 3)
            a = numerator / denominator if np.isfinite(denominator) and abs(denominator) > 1e-12 else 0.0
            a = np.clip(a, -1.0, 1.0)
            
            # Z-scores
            z_alpha_lower = stats.norm.ppf(alpha / 2)
            z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
            
            denom_lower = 1 - a * (z0 + z_alpha_lower)
            denom_upper = 1 - a * (z0 + z_alpha_upper)
            
            if abs(denom_lower) < 1e-12 or abs(denom_upper) < 1e-12:
                p_lower = alpha / 2
                p_upper = 1 - alpha / 2
            else:
                p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / denom_lower)
                p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / denom_upper)
            
            p_lower = float(np.clip(p_lower, 1e-6, 1-1e-6) * 100)
            p_upper = float(np.clip(p_upper, 1e-6, 1-1e-6) * 100)
            
            if p_lower >= p_upper:
                p_lower = (alpha / 2) * 100.0
                p_upper = (1 - alpha / 2) * 100.0
            
            lower = float(np.percentile(bootstrap_sorted, p_lower))
            upper = float(np.percentile(bootstrap_sorted, p_upper))
            
            return lower, upper
        except:
            alpha = 1 - confidence_level
            return (float(np.percentile(bootstrap_estimates, alpha/2 * 100)),
                   float(np.percentile(bootstrap_estimates, (1-alpha/2) * 100)))
    
    def bootstrap_metrics(self) -> Dict[str, Any]:
        """Calculate bootstrap CIs."""
        auc_scores = []
        brier_scores = []
        f1_scores = []
        
        threshold = float(self.model_package.get('threshold', 0.5))
        rng = np.random.RandomState(self.config.random_state)
        
        n = len(self.y_test)
        
        for _ in range(self.config.n_bootstraps):
            indices = rng.randint(0, n, size=n)
            y_boot = self.y_test[indices]
            proba_boot = self.y_proba[indices]
            
            if len(np.unique(y_boot)) > 1:
                try:
                    auc_scores.append(roc_auc_score(y_boot, proba_boot))
                    brier_scores.append(brier_score_loss(y_boot, proba_boot))
                    pred_boot = (proba_boot >= threshold).astype(int)
                    f1_scores.append(f1_score(y_boot, pred_boot))
                except:
                    continue
        
        def safe_stats(arr):
            arr = np.asarray(arr)
            if arr.size == 0:
                return float('nan'), (float('nan'), float('nan'))
            mean = float(np.mean(arr))
            ci = self.calculate_bca_ci(arr, mean, self.config.confidence_level)
            return mean, ci
        
        auc_mean, auc_ci = safe_stats(auc_scores)
        brier_mean, brier_ci = safe_stats(brier_scores)
        f1_mean, f1_ci = safe_stats(f1_scores)
        
        return {
            'auc': {'mean': auc_mean, 'ci': auc_ci},
            'brier': {'mean': brier_mean, 'ci': brier_ci},
            'f1': {'mean': f1_mean, 'ci': f1_ci}
        }
    
    def evaluate_performance(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        threshold = float(self.model_package.get('threshold', 0.5))
        y_pred = (self.y_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = self._safe_confusion(self.y_test, y_pred)
        
        def safe_div(nom, denom):
            return float(nom / denom) if denom and denom != 0 else float('nan')
        
        metrics = {
            'auc': float(roc_auc_score(self.y_test, self.y_proba)),
            'brier': float(brier_score_loss(self.y_test, self.y_proba)),
            'f1': float(f1_score(self.y_test, y_pred)),
            'accuracy': float(accuracy_score(self.y_test, y_pred)),
            'precision': float(precision_score(self.y_test, y_pred)),
            'recall': float(recall_score(self.y_test, y_pred)),
            'sensitivity': safe_div(tp, (tp + fn)),
            'specificity': safe_div(tn, (tn + fp)),
            'ppv': safe_div(tp, (tp + fp)),
            'npv': safe_div(tn, (tn + fn)),
            'fnr': safe_div(fn, (fn + tp)),
            'fpr': safe_div(fp, (fp + tn)),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'threshold': float(threshold)
        }
        
        bootstrap_results = self.bootstrap_metrics()
        
        return {
            'point_estimates': metrics,
            'bootstrap_ci': bootstrap_results
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation."""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        self.load_model_and_data()
        self.get_predictions()
        performance = self.evaluate_performance()
        
        model_type = self.model_package.get('model_type', 'Unknown')
        calibration = self.model_package.get('calibration_method', 'none')
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Model: {model_type} + {calibration}")
        print(f"AUC: {performance['bootstrap_ci']['auc']['mean']:.4f}")
        print(f"Brier: {performance['bootstrap_ci']['brier']['mean']:.4f}")
        print(f"Sensitivity: {performance['point_estimates']['sensitivity']:.4f}")
        print(f"Specificity: {performance['point_estimates']['specificity']:.4f}")
        print("="*80 + "\n")
        
        return performance


if __name__ == "__main__":
    try:
        evaluator = ModelEvaluator()
        results = evaluator.run_evaluation()
        print("✅ Evaluation complete!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()