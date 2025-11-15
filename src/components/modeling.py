"""
Heart Disease Modeling Pipeline
Only Beta + Isotonic Calibration
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import warnings
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

import optuna
from optuna.samplers import TPESampler

try:
    from betacal import BetaCalibration
    BETACAL_AVAILABLE = True
except ImportError:
    BETACAL_AVAILABLE = False
    print("⚠️  Beta calibration not available")

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)

PROJECT_ROOT = Path.cwd()

@dataclass
class ModelingConfig:
    """Configuration for modeling pipeline."""
    model_dir: str = str(PROJECT_ROOT / "artifacts" / "model")
    data_dir: str = str(PROJECT_ROOT / "artifacts" / "split data sets")
    random_state: int = 42
    n_trials: int = 50
    n_estimators: int = 1500
    cat_iterations: int = 400
    min_sensitivity: float = 0.80


class HeartDiseaseModeling:
    """Modeling pipeline with only Beta + Isotonic calibration."""
    
    def __init__(self, config: Optional[ModelingConfig] = None):
        self.config = config or ModelingConfig()
        
        for directory in [self.config.model_dir, self.config.data_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        np.random.seed(self.config.random_state)
        
        self.models: Dict = {}
        self.best_model = None
        self.best_calibrator = None
        self.best_calibration_method = 'none'
        
        print(f"Initialized (random_state={self.config.random_state})")
    
    def load_data(self) -> Tuple:
        """Load processed data and split."""
        data_path = PROJECT_ROOT / 'data' / 'processed' / 'processed_data.parquet'
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        
        df = pd.read_parquet(data_path)
        X = df.drop('target', axis=1)
        y = df['target']
        
        # 70% train, 30% test
        X_train, X_test_full, y_train, y_test_full = train_test_split(
            X, y, test_size=0.3, random_state=self.config.random_state, stratify=y
        )
        
        # Split test: 15% cal, 15% test
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_test_full, y_test_full, test_size=0.5, 
            stratify=y_test_full, random_state=self.config.random_state
        )
        
        # Save splits
        split_dir = Path(self.config.data_dir)
        X_train.to_parquet(split_dir / 'X_train.parquet', index=False)
        X_cal.to_parquet(split_dir / 'X_cal.parquet', index=False)
        X_test.to_parquet(split_dir / 'X_test.parquet', index=False)
        y_train.to_frame().to_parquet(split_dir / 'y_train.parquet', index=False)
        y_cal.to_frame().to_parquet(split_dir / 'y_cal.parquet', index=False)
        y_test.to_frame().to_parquet(split_dir / 'y_test.parquet', index=False)
        
        print(f"Split: Train={X_train.shape}, Cal={X_cal.shape}, Test={X_test.shape}")
        
        return X_train, X_cal, X_test, y_train, y_cal, y_test
    
    def optimize_hyperparameters(self, model_type: str, X_train, y_train, X_val, y_val) -> Dict:
        """Optimize using Optuna."""
        pbar = tqdm(total=self.config.n_trials, desc=f"  {model_type.upper()}", leave=False)
        
        def objective(trial):
            try:
                if model_type == 'xgb':
                    model = XGBClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=trial.suggest_int('max_depth', 3, 9),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                        min_child_weight=trial.suggest_int('min_child_weight', 1, 15),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        reg_alpha=trial.suggest_float('reg_alpha', 0, 5.0),
                        reg_lambda=trial.suggest_float('reg_lambda', 0, 5.0),
                        random_state=self.config.random_state,
                        verbosity=0
                    )
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                elif model_type == 'lgbm':
                    model = LGBMClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=trial.suggest_int('max_depth', 4, 7),
                        num_leaves=trial.suggest_int('num_leaves', 20, 60),
                        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                        min_child_samples=trial.suggest_int('min_child_samples', 10, 40),
                        subsample=trial.suggest_float('subsample', 0.7, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.7, 1.0),
                        random_state=self.config.random_state,
                        verbosity=-1
                    )
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                             callbacks=[lgb.early_stopping(50, verbose=False)])
                
                else:  # catboost
                    model = CatBoostClassifier(
                        iterations=self.config.cat_iterations,
                        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
                        depth=trial.suggest_int('depth', 4, 7),
                        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 8),
                        random_strength=trial.suggest_float('random_strength', 0, 1.5),
                        random_seed=self.config.random_state,
                        verbose=False
                    )
                    model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val), 
                             early_stopping_rounds=40, verbose=False)
                
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
                
                pbar.update(1)
                return score
            except:
                pbar.update(1)
                return 0.0
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config.random_state))
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)
        pbar.close()
        
        print(f"    Best AUC: {study.best_value:.4f}")
        return study.best_params
    
    def train_models(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train all models."""
        print("Optimizing hyperparameters...")
        
        params = {}
        for model_type in ['xgb', 'lgbm', 'cat']:
            params[model_type] = self.optimize_hyperparameters(model_type, X_train, y_train, X_val, y_val)
        
        print("Training models...")
        models = {}
        
        # XGBoost
        xgb_model = XGBClassifier(**{**params['xgb'], 'n_estimators': self.config.n_estimators,
                                     'random_state': self.config.random_state, 'verbosity': 0})
        xgb_model.fit(X_train, y_train, verbose=False)
        models['XGBoost'] = {'model': xgb_model, 'params': params['xgb']}
        
        # LightGBM
        lgbm_model = LGBMClassifier(**{**params['lgbm'], 'n_estimators': self.config.n_estimators,
                                      'random_state': self.config.random_state, 'verbosity': -1})
        lgbm_model.fit(X_train, y_train)
        models['LightGBM'] = {'model': lgbm_model, 'params': params['lgbm']}
        
        # CatBoost
        cat_model = CatBoostClassifier(**{**params['cat'], 'iterations': self.config.cat_iterations,
                                         'random_seed': self.config.random_state, 'verbose': False})
        cat_model.fit(X_train, y_train, verbose=False)
        models['CatBoost'] = {'model': cat_model, 'params': params['cat']}
        
        self.models = models
        print(f"✓ Trained {len(models)} models")
        
        return models
    
    def evaluate_calibrations(self, model, X_cal, y_cal, X_test, y_test) -> Dict:
        """Evaluate ONLY Beta + Isotonic."""
        results = {}
        
        # Base (no calibration)
        base_proba = model.predict_proba(X_test)[:, 1]
        results['none'] = {
            'calibrator': None,
            'brier': brier_score_loss(y_test, base_proba),
            'auc': roc_auc_score(y_test, base_proba)
        }
        
        # Isotonic
        try:
            y_cal_proba = model.predict_proba(X_cal)[:, 1]
            iso_cal = IsotonicRegression(out_of_bounds='clip')
            iso_cal.fit(y_cal_proba, y_cal)
            iso_proba = iso_cal.predict(base_proba)
            
            results['isotonic'] = {
                'calibrator': iso_cal,
                'brier': brier_score_loss(y_test, iso_proba),
                'auc': roc_auc_score(y_test, iso_proba)
            }
        except Exception as e:
            print(f"  Isotonic failed: {e}")
        
        # Beta
        if BETACAL_AVAILABLE:
            try:
                y_cal_proba = model.predict_proba(X_cal)[:, 1]
                beta_cal = BetaCalibration(parameters="abm")
                beta_cal.fit(y_cal_proba, y_cal)
                beta_proba = beta_cal.predict(base_proba)
                
                results['beta'] = {
                    'calibrator': beta_cal,
                    'brier': brier_score_loss(y_test, beta_proba),
                    'auc': roc_auc_score(y_test, beta_proba)
                }
            except Exception as e:
                print(f"  Beta failed: {e}")
        
        return results
    
    def select_best(self, X_cal, y_cal, X_test, y_test) -> Tuple:
        """Select best model + calibration."""
        print("Evaluating calibrations...")
        
        all_results = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            cal_results = self.evaluate_calibrations(model, X_cal, y_cal, X_test, y_test)
            
            for cal_method, results in cal_results.items():
                combo_name = f"{model_name}_{cal_method}"
                all_results[combo_name] = {
                    'model_name': model_name,
                    'base_model': model,
                    'calibration_method': cal_method,
                    **results
                }
                print(f"  {combo_name}: Brier={results['brier']:.4f}, AUC={results['auc']:.4f}")
        
        # Select best by Brier
        best_combo = min(all_results.keys(), key=lambda x: all_results[x]['brier'])
        best_result = all_results[best_combo]
        
        self.best_model = best_result['base_model']
        self.best_calibrator = best_result['calibrator']
        self.best_calibration_method = best_result['calibration_method']
        
        print(f"✓ Selected: {best_combo} (Brier: {best_result['brier']:.4f})")
        
        return best_combo, best_result
    
    def save_model(self, model_name: str, performance: Dict, threshold: float) -> str:
        """Save model package."""
        package = {
            'base_model': self.best_model,
            'calibrator': self.best_calibrator,
            'calibration_method': self.best_calibration_method,
            'model_type': model_name,
            'threshold': threshold,
            'performance': performance,
            'created_at': datetime.now().isoformat()
        }
        
        model_path = Path(self.config.model_dir) / 'model_package.pkl'
        joblib.dump(package, model_path, compress=3)
        
        print(f"✓ Saved: {model_path}")
        return str(model_path)
    
    def run_pipeline(self) -> Dict:
        """Run complete pipeline."""
        print("\n" + "="*80)
        print("HEART DISEASE MODELING PIPELINE")
        print("="*80)
        
        # Load data
        X_train, X_cal, X_test, y_train, y_cal, y_test = self.load_data()
        
        # Train
        X_train_val, X_val, y_train_val, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=self.config.random_state
        )
        self.train_models(X_train_val, y_train_val, X_val, y_val)
        
        # Select best
        best_combo, best_result = self.select_best(X_cal, y_cal, X_test, y_test)
        
        # Evaluate
        if self.best_calibrator:
            if self.best_calibration_method == 'beta':
                base_proba = self.best_model.predict_proba(X_test)[:, 1]
                y_proba = self.best_calibrator.predict(base_proba)
            else:
                base_proba = self.best_model.predict_proba(X_test)[:, 1]
                y_proba = self.best_calibrator.predict(base_proba)
        else:
            y_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Find threshold
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        valid_mask = tpr >= self.config.min_sensitivity
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            youden_j = tpr[valid_indices] - fpr[valid_indices]
            best_idx = valid_indices[np.argmax(youden_j)]
        else:
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
        
        threshold = float(thresholds[best_idx])
        y_pred = (y_proba >= threshold).astype(int)
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        performance = {
            'auc': roc_auc_score(y_test, y_proba),
            'brier': brier_score_loss(y_test, y_proba),
            'sensitivity': tp / (tp + fn),
            'specificity': tn / (tn + fp),
            'fnr': fn / (tp + fn)
        }
        
        # Save
        model_path = self.save_model(best_combo, performance, threshold)
        
        print("\n" + "="*80)
        print("COMPLETE")
        print("="*80)
        print(f"Model: {best_combo}")
        print(f"AUC: {performance['auc']:.4f}")
        print(f"Sensitivity: {performance['sensitivity']:.4f}")
        print("="*80 + "\n")
        
        return {
            'model_path': model_path,
            'model_name': best_combo,
            'performance': performance,
            'threshold': threshold
        }


if __name__ == "__main__":
    pipeline = HeartDiseaseModeling()
    results = pipeline.run_pipeline()
    print("✅ Pipeline complete!")