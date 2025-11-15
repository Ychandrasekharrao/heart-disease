"""
Heart Disease Training Pipeline - TRAINING ONLY
No prediction code - use separate predict_pipeline.py for testing
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
from tqdm import tqdm
import warnings
from typing import Dict, Any, Tuple, Optional, Callable

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss, roc_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

import optuna
from optuna.samplers import TPESampler
from scipy import stats
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)

try:
    from betacal import BetaCalibration
    BETACAL_AVAILABLE = True
except ImportError:
    BETACAL_AVAILABLE = False
    print("⚠️  Beta calibration not available")


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG: Dict[str, Any] = {
    'random_state': 42,
    'n_trials': 50,
    'n_estimators': 1500,
    'cat_iterations': 400,
    'n_bootstraps': 1000,
    'min_sensitivity': 0.80,
    'cart_max_depth': 2,
    'cart_min_samples_leaf': 500,
    'cart_min_impurity_decrease': 0.01,
    'calibration_methods': ['isotonic'] + (['beta'] if BETACAL_AVAILABLE else []),
}

RANDOM_STATE = CONFIG['random_state']
np.random.seed(RANDOM_STATE)


# ============================================================================
# HELPER FUNCTIONS (from your reference)
# ============================================================================

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, 
                          min_sensitivity: float = 0.80) -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    valid_mask = tpr >= min_sensitivity
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print(f"  ⚠️  No threshold achieves sensitivity ≥ {min_sensitivity:.0%}")
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
    else:
        youden_j = tpr[valid_indices] - fpr[valid_indices]
        best_idx = valid_indices[np.argmax(youden_j)]
    
    threshold = float(thresholds[best_idx])
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'threshold': threshold,
        'sensitivity': float(tp / (tp + fn)),
        'specificity': float(tn / (tn + fp)),
        'fnr': float(fn / (tp + fn))
    }


def calculate_bca_ci(y_true: np.ndarray, y_proba: np.ndarray, threshold: float, 
                    metric_fn: Callable, n_bootstraps: int = 1000, 
                    n_jobs: int = -1) -> Dict[str, float]:
    n = len(y_true)
    
    try:
        theta_hat = metric_fn(y_true, y_proba, threshold)
    except:
        return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
    
    def bootstrap_sample(seed: int):
        try:
            idx = resample(range(n), n_samples=n, random_state=seed, stratify=y_true)
            return metric_fn(y_true[idx], y_proba[idx], threshold)
        except:
            return None
    
    theta_boot = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_sample)(i) for i in range(n_bootstraps)
    )
    theta_boot = [x for x in theta_boot if x is not None]
    
    if len(theta_boot) < 100:
        return {'mean': theta_hat, 'ci_lower': np.nan, 'ci_upper': np.nan}
    
    theta_boot = np.array(theta_boot)
    p_less = np.clip(np.mean(theta_boot < theta_hat), 0.001, 0.999)
    z0 = stats.norm.ppf(p_less)
    
    theta_mean = np.mean(theta_boot)
    diff = theta_boot - theta_mean
    numerator = np.sum(diff ** 3)
    denominator = 6 * (np.sum(diff ** 2) ** 1.5)
    a = np.clip(numerator / denominator if denominator != 0 else 0.0, -0.5, 0.5)
    
    z_alpha = stats.norm.ppf(0.025)
    z_1_alpha = stats.norm.ppf(0.975)
    
    denom_lower = max(1 - a * (z0 + z_alpha), 0.001)
    denom_upper = max(1 - a * (z0 + z_1_alpha), 0.001)
    
    p_lower = np.clip(stats.norm.cdf(z0 + (z0 + z_alpha) / denom_lower), 0.001, 0.999)
    p_upper = np.clip(stats.norm.cdf(z0 + (z0 + z_1_alpha) / denom_upper), 0.001, 0.999)
    
    ci_lower = np.percentile(theta_boot, p_lower * 100)
    ci_upper = np.percentile(theta_boot, p_upper * 100)
    
    return {'mean': float(theta_hat), 'ci_lower': float(ci_lower), 'ci_upper': float(ci_upper)}


def bootstrap_bca_ci(y_true: np.ndarray, y_proba: np.ndarray, threshold: float, 
                    n_bootstraps: int = 1000, n_jobs: int = -1) -> Dict:
    
    def auc_fn(y_t, y_p, t):
        try:
            return roc_auc_score(y_t, y_p)
        except:
            return np.nan
    
    def sens_fn(y_t, y_p, t):
        y_pred = (y_p >= t).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
            return tp / (tp + fn) if (tp + fn) > 0 else np.nan
        except:
            return np.nan
    
    def spec_fn(y_t, y_p, t):
        y_pred = (y_p >= t).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else np.nan
        except:
            return np.nan
    
    def fnr_fn(y_t, y_p, t):
        return 1 - sens_fn(y_t, y_p, t)
    
    print(f"\n  Computing BCa bootstrap CIs ({n_bootstraps} samples)...")
    
    metrics = {'AUC': auc_fn, 'SENSITIVITY': sens_fn, 'SPECIFICITY': spec_fn, 'FNR': fnr_fn}
    
    results = {}
    for metric_name, metric_fn in tqdm(metrics.items(), desc="  Progress", leave=False):
        results[metric_name] = calculate_bca_ci(y_true, y_proba, threshold, metric_fn, n_bootstraps, n_jobs)
    
    print(f"\n  BCa 95% Confidence Intervals:")
    for metric_name, result in results.items():
        ci_str = "[NaN, NaN]" if np.isnan(result['ci_lower']) else f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"
        print(f"    {metric_name:<12}: {result['mean']:.3f} {ci_str}")
    
    return results


def create_cart_3_groups(y_true: np.ndarray, y_proba: np.ndarray, 
                        config: Dict) -> Tuple[DecisionTreeClassifier, Dict, Dict]:
    """
    ✅ RETURNS: cart_model, leaf_to_risk mapping, risk_statistics
    """
    print("\n  Creating CART-based risk stratification (3 groups)...")
    
    X_cart = pd.DataFrame({'Predicted_Risk': y_proba})
    
    cart = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=config['cart_min_samples_leaf'],
        min_samples_split=config['cart_min_samples_leaf'] * 2,
        min_impurity_decrease=config.get('cart_min_impurity_decrease', 0.01),
        class_weight='balanced',
        random_state=config['random_state']
    )
    
    cart.fit(X_cart, y_true)
    leaves = cart.apply(X_cart)
    unique_leaves = np.unique(leaves)
    
    print(f"  ✅ CART created {len(unique_leaves)} risk groups")
    
    # Build leaf statistics
    leaf_stats = []
    for leaf in unique_leaves:
        mask = leaves == leaf
        leaf_stats.append({
            'leaf': leaf,
            'n': mask.sum(),
            'prevalence': y_true[mask].mean(),
            'mean_risk': y_proba[mask].mean()
        })
    
    # Sort by disease prevalence (Low → High)
    leaf_stats.sort(key=lambda x: x['prevalence'])
    
    # Create leaf_to_risk mapping
    risk_names = ['Low', 'Moderate', 'High']
    leaf_to_risk = {}
    
    for i, stat in enumerate(leaf_stats):
        leaf_to_risk[stat['leaf']] = risk_names[min(i, len(risk_names)-1)]
    
    # Print risk groups
    print("\n  📊 Risk Groups:")
    print(f"  {'Group':<15} {'N':<10} {'%':<8} {'Prevalence':<12} {'Mean Risk':<12}")
    print("  " + "-"*65)
    
    risk_statistics = {}
    for risk in risk_names:
        # Find leaves for this risk group
        matching_leaves = [leaf for leaf, rname in leaf_to_risk.items() if rname == risk]
        if not matching_leaves:
            continue
        
        # Get all samples in this risk group
        mask = np.isin(leaves, matching_leaves)
        
        if mask.sum() == 0:
            continue
        
        n = mask.sum()
        pct = n / len(y_true) * 100
        prevalence = y_true[mask].mean()
        mean_risk = y_proba[mask].mean()
        
        print(f"  {risk:<15} {n:<10} {pct:<8.1f} {prevalence*100:<12.1f} {mean_risk:<12.3f}")
        
        risk_statistics[risk] = {
            'n': int(n),
            'percentage': float(pct),
            'disease_prevalence': float(prevalence * 100),
            'mean_risk_score': float(mean_risk)
        }
    
    print(f"\n  💡 3 groups based on natural risk distribution")
    
    return cart, leaf_to_risk, risk_statistics


def train_calibrator(model, X_train: pd.DataFrame, y_train: np.ndarray, 
                     X_test: pd.DataFrame, method: str = 'isotonic') -> Tuple:
    
    y_test_base = model.predict_proba(X_test)[:, 1]
    
    if method == 'isotonic':
        try:
            y_train_base = model.predict_proba(X_train)[:, 1]
            iso_cal = IsotonicRegression(out_of_bounds='clip')
            iso_cal.fit(y_train_base, y_train)
            return iso_cal.predict(y_test_base), iso_cal
        except Exception as e:
            print(f"  ⚠️  Isotonic failed: {e}")
            return y_test_base, None
    
    elif method == 'beta' and BETACAL_AVAILABLE:
        try:
            y_train_base = model.predict_proba(X_train)[:, 1]
            beta_cal = BetaCalibration(parameters="abm")
            beta_cal.fit(y_train_base, y_train)
            return beta_cal.predict(y_test_base), beta_cal
        except Exception as e:
            print(f"  ⚠️  Beta failed: {e}")
            return y_test_base, None
    
    return y_test_base, None


def optimize_model(model_type: str, X_train: pd.DataFrame, y_train: np.ndarray, 
                  config: Dict) -> Dict:
    
    print(f"  Optimizing {model_type.upper()}...")
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=config['random_state']
    )
    
    def objective(trial):
        if model_type == 'xgb':
            params = {
                'n_estimators': config['n_estimators'],
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'random_state': config['random_state'],
                'verbosity': 0
            }
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        elif model_type == 'lgbm':
            params = {
                'n_estimators': config['n_estimators'],
                'max_depth': trial.suggest_int('max_depth', 4, 7),
                'num_leaves': trial.suggest_int('num_leaves', 20, 60),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 3.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 3.0),
                'device': 'cpu',
                'random_state': config['random_state'],
                'verbosity': -1
            }
            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50, verbose=False)])
        
        else:  # catboost
            params = {
                'iterations': config['cat_iterations'],
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
                'depth': trial.suggest_int('depth', 4, 7),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 8),
                'random_strength': trial.suggest_float('random_strength', 0, 1.5),
                'task_type': 'CPU',
                'random_seed': config['random_state'],
                'verbose': False
            }
            model = CatBoostClassifier(**params)
            model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val), 
                     early_stopping_rounds=40, verbose=False)
        
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=config['random_state']))
    study.optimize(objective, n_trials=config['n_trials'], show_progress_bar=True)
    
    print(f"    Best AUC: {study.best_value:.4f}")
    return study.best_params


def get_project_root() -> Path:
    current = Path.cwd().resolve()
    max_depth = 5
    depth = 0
    
    while depth < max_depth and current != current.parent:
        if (current / 'artifacts').exists():
            return current
        current = current.parent
        depth += 1
    
    raise RuntimeError("Could not find project root")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main() -> Dict:
    print("\n" + "="*80)
    print("HEART DISEASE PREDICTION - PRODUCTION MODEL TRAINING")
    print("Auto Selection + 3 Risk Groups + BCa Bootstrap")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    PROJECT_ROOT = get_project_root()
    SPLIT_DIR = PROJECT_ROOT / 'artifacts' / 'split data sets'
    MODEL_DIR = PROJECT_ROOT / 'artifacts' / 'model'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[1/8] Loading data...")
    X_train = pd.read_parquet(SPLIT_DIR / 'X_train.parquet')
    X_test_full = pd.read_parquet(SPLIT_DIR / 'X_test.parquet')
    y_train = pd.read_parquet(SPLIT_DIR / 'y_train.parquet').iloc[:, 0].values
    y_test_full = pd.read_parquet(SPLIT_DIR / 'y_test.parquet').iloc[:, 0].values
    
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_test_full, y_test_full, test_size=0.5, stratify=y_test_full, random_state=RANDOM_STATE
    )
    
    print(f"  ✅ Train: {len(X_train):,} | Cal: {len(X_cal):,} | Test: {len(X_test):,}")
    
    print(f"\n[2/8] Hyperparameter optimization ({CONFIG['n_trials']} trials)...")
    best_params = {}
    for model_type in ['xgb', 'lgbm', 'cat']:
        best_params[model_type] = optimize_model(model_type, X_train, y_train, CONFIG)
    
    print("\n[3/8] Training models...")
    models = {}
    
    xgb_params = {**best_params['xgb'], 'n_estimators': CONFIG['n_estimators'],
                  'tree_method': 'gpu_hist', 'gpu_id': 0, 'random_state': RANDOM_STATE, 'verbosity': 0}
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], early_stopping_rounds=50, verbose=False)
    models['XGBoost'] = {'model': xgb_model, 'params': best_params['xgb']}
    print("  ✅ XGBoost (GPU)")
    
    lgbm_params = {**best_params['lgbm'], 'n_estimators': CONFIG['n_estimators'],
                   'device': 'cpu', 'random_state': RANDOM_STATE, 'verbosity': -1}
    lgbm_model = LGBMClassifier(**lgbm_params)
    lgbm_model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], 
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    models['LightGBM'] = {'model': lgbm_model, 'params': best_params['lgbm']}
    print("  ✅ LightGBM (CPU)")
    
    cat_params = {**best_params['cat'], 'iterations': CONFIG['cat_iterations'],
                  'task_type': 'CPU', 'random_seed': RANDOM_STATE, 'verbose': False}
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_cal, y_cal), early_stopping_rounds=40, verbose=False)
    models['CatBoost'] = {'model': cat_model, 'params': best_params['cat']}
    print("  ✅ CatBoost (CPU)")
    
    print("\n[4/8] Calibration selection...")
    X_cal_train, X_cal_val, y_cal_train, y_cal_val = train_test_split(
        X_cal, y_cal, test_size=0.5, stratify=y_cal, random_state=RANDOM_STATE
    )
    
    best_brier = float('inf')
    best_model_name = None
    best_cal_method = None
    
    for model_name, model_info in models.items():
        for cal_method in CONFIG['calibration_methods']:
            y_cal_val_proba, _ = train_calibrator(
                model_info['model'], X_cal_train, y_cal_train, X_cal_val, cal_method
            )
            brier = brier_score_loss(y_cal_val, y_cal_val_proba)
            if brier < best_brier:
                best_brier = brier
                best_model_name = model_name
                best_cal_method = cal_method
    
    print(f"  🏆 Selected: {best_model_name} + {best_cal_method}")
    print(f"  ✅ Calibration Brier: {best_brier:.4f}")
    
    best_model = models[best_model_name]['model']
    y_test_proba, final_calibrator = train_calibrator(best_model, X_cal, y_cal, X_test, best_cal_method)
    
    print("\n[5/8] Threshold optimization...")
    y_cal_proba, _ = train_calibrator(best_model, X_cal, y_cal, X_cal, best_cal_method)
    threshold_result = find_optimal_threshold(y_cal, y_cal_proba, CONFIG['min_sensitivity'])
    threshold = threshold_result['threshold']
    
    print(f"  ✅ Threshold: {threshold:.4f}")
    print(f"  ✅ Sensitivity: {threshold_result['sensitivity']:.3f}")
    
    print("\n[6/8] Test evaluation...")
    y_pred = (y_test_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    performance = {
        'auc': roc_auc_score(y_test, y_test_proba),
        'brier': brier_score_loss(y_test, y_test_proba),
        'sensitivity': float(tp / (tp + fn)),
        'specificity': float(tn / (tn + fp)),
        'fnr': float(fn / (tp + fn))
    }
    
    print(f"  AUC: {performance['auc']:.4f}")
    print(f"  Sensitivity: {performance['sensitivity']:.3f}")
    
    print("\n[7/8] BCa Bootstrap CIs...")
    ci_results = bootstrap_bca_ci(y_test, y_test_proba, threshold, CONFIG['n_bootstraps'])
    
    print("\n[8/8] CART risk stratification (3 groups)...")
    cart_model, leaf_to_risk, risk_stats = create_cart_3_groups(y_test, y_test_proba, CONFIG)
    
    print("\n  Saving model package...")
    package = {
        'version': '1.0_production',
        'timestamp': datetime.now().isoformat(),
        'model_type': best_model_name,
        'base_model': best_model,
        'calibrator': final_calibrator,
        'calibration_method': best_cal_method,
        'cart_model': cart_model,
        'leaf_to_risk': leaf_to_risk,  # ✅ SAVE THIS
        'threshold': threshold,
        'test_performance': performance,
        'confidence_intervals': ci_results,
        'risk_statistics': risk_stats,
        'config': CONFIG
    }
    
    joblib.dump(package, MODEL_DIR / 'model_package.pkl', compress=3)
    print("  ✅ Saved: model_package.pkl")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel: {best_model_name} + {best_cal_method}")
    print(f"Risk Groups: {list(risk_stats.keys())}")
    print(f"\nTest Performance:")
    for metric, result in ci_results.items():
        ci_str = "[NaN, NaN]" if np.isnan(result['ci_lower']) else f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"
        print(f"  {metric}: {result['mean']:.3f} {ci_str}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 To test predictions, run:")
    print("   python -m src.pipeline.predict_pipeline")
    print("="*80 + "\n")
    
    return package


if __name__ == "__main__":
    try:
        package = main()
        print("✅ Training successful! Model saved.\n")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
