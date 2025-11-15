# src/components/data_handler.py
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import logging

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataHandler:
    """Simplified DataHandler for heart disease preprocessing."""
    
    def __init__(self, project_root: Optional[Path] = None, random_state: int = 42):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.data_raw_dir = (self.project_root / "data" / "raw").resolve()
        self.data_processed_dir = (self.project_root / "data" / "processed").resolve()
        self.reports_dir = (self.project_root / "reports").resolve()
        
        for directory in [self.data_raw_dir, self.data_processed_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.df: Optional[pd.DataFrame] = None
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.random_state = int(random_state)
        np.random.seed(self.random_state)
        
        logger.info("DataHandler initialized (random_state=%s)", self.random_state)
    
    def load_data(self, file_path: str, file_type: str = "processed") -> pd.DataFrame:
        """Load data from parquet or CSV."""
        if file_type not in ("raw", "processed"):
            raise ValueError("file_type must be 'raw' or 'processed'")
        
        data_dir = self.data_raw_dir if file_type == "raw" else self.data_processed_dir
        full_path = data_dir / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Data file not found: {full_path}")
        
        suffix = full_path.suffix.lower()
        
        if suffix == ".parquet":
            self.df = pd.read_parquet(full_path)
        elif suffix == ".csv":
            self.df = pd.read_csv(full_path, low_memory=False)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        
        if self.df is None or self.df.shape[0] == 0:
            raise ValueError(f"Loaded data is empty from {full_path}")
        
        self.df = self.df.reset_index(drop=True)
        logger.info("Loaded data from %s (shape=%s)", full_path, self.df.shape)
        
        return self.df
    
    def handle_missing_values(self, strategy: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Handle missing values using MICE for numeric, mode for categorical."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        strategy = strategy or {"numeric": "mice", "categorical": "most_frequent"}
        numeric_strategy = strategy.get("numeric", "mice")
        categorical_strategy = strategy.get("categorical", "most_frequent")
        
        logger.info("Handling missing values (numeric=%s categorical=%s)", 
                   numeric_strategy, categorical_strategy)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Numeric (MICE)
        if numeric_cols and numeric_strategy == "mice":
            if self.df[numeric_cols].isnull().values.any():
                try:
                    imp = IterativeImputer(
                        max_iter=10, 
                        random_state=self.random_state, 
                        initial_strategy="median",
                        verbose=0
                    )
                    numeric_df = pd.DataFrame(
                        imp.fit_transform(self.df[numeric_cols]), 
                        columns=numeric_cols, 
                        index=self.df.index
                    )
                    self.df[numeric_cols] = numeric_df
                    self.imputers["mice"] = imp
                    logger.info("✓ Applied MICE to %d numeric columns", len(numeric_cols))
                except Exception as e:
                    logger.warning("MICE failed: %s. Using median.", e)
                    imp = SimpleImputer(strategy="median")
                    self.df[numeric_cols] = imp.fit_transform(self.df[numeric_cols])
                    self.imputers["numeric"] = imp
        
        # Categorical (mode)
        if categorical_cols:
            cols_to_impute = [c for c in categorical_cols if self.df[c].isnull().any()]
            if cols_to_impute:
                imp = SimpleImputer(strategy=categorical_strategy, fill_value="Unknown")
                self.df[cols_to_impute] = imp.fit_transform(self.df[cols_to_impute])
                self.imputers["categorical"] = imp
                logger.info("✓ Applied %s to %d categorical columns", 
                          categorical_strategy, len(cols_to_impute))
        
        missing_after = int(self.df.isnull().sum().sum())
        logger.info("Missing values after handling: %d", missing_after)
        
        return self.imputers
    
    def handle_outliers(self, method: str = "iqr", threshold: float = 1.5) -> Dict[str, int]:
        """Handle outliers using IQR capping."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "target" in numeric_cols:
            numeric_cols.remove("target")
        
        outliers_report: Dict[str, int] = {}
        total_handled = 0
        
        for col in numeric_cols:
            series = pd.to_numeric(self.df[col], errors="coerce").astype(float)
            
            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR == 0:
                    outliers_report[col] = 0
                    continue
                
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                mask = (series < lower) | (series > upper)
                count = int(mask.sum())
                
                if count:
                    clipped = np.clip(series[mask], lower, upper)
                    self.df.at[series[mask].index, col] = clipped.values
                
                outliers_report[col] = count
                total_handled += count
        
        outliers_report["total_handled"] = total_handled
        logger.info("Handled %d outliers using %s", total_handled, method)
        return outliers_report
    
    def scale_features(self, exclude_cols: Optional[list] = None) -> Dict[str, Any]:
        """Scale numeric features using RobustScaler."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        exclude_cols = list(exclude_cols or [])
        
        # Always exclude target
        if "target" not in exclude_cols:
            exclude_cols.append("target")
        
        numeric_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns.tolist() 
                       if c not in exclude_cols]
        
        if not numeric_cols:
            logger.warning("No numeric columns to scale")
            return self.scalers
        
        scaler = RobustScaler()
        
        # Fill NaNs before scaling
        tmp = self.df[numeric_cols].copy()
        for c in tmp.columns:
            if tmp[c].isnull().any():
                tmp[c] = tmp[c].fillna(tmp[c].median())
        
        scaled = scaler.fit_transform(tmp)
        self.df[numeric_cols] = scaled
        self.scalers["robust"] = scaler
        
        logger.info("Scaled %d numeric columns using RobustScaler", len(numeric_cols))
        return self.scalers
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate data quality report."""
        if self.df is None:
            return {"error": "No data loaded"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": {
                "total": int(self.df.isnull().sum().sum()),
                "by_column": {col: int(cnt) for col, cnt in self.df.isnull().sum().items()}
            },
            "memory_mb": float(self.df.memory_usage(deep=True).sum() / 1024**2)
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = "data_quality_report.json") -> Path:
        """Save report to JSON file."""
        out = self.reports_dir / filename
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Saved report to %s", out)
        return out
    
    def run_pipeline(self, file_path: str, file_type: str = "processed", 
                    save_report: bool = True) -> Dict[str, Any]:
        """Run preprocessing pipeline."""
        artifacts = {
            "imputers": {},
            "scalers": {},
            "outliers_report": {},
            "report": {}
        }
        
        # Load
        self.load_data(file_path, file_type=file_type)
        
        # Process
        artifacts["imputers"] = self.handle_missing_values()
        artifacts["outliers_report"] = self.handle_outliers()
        
        # Exclude engineered features from scaling
        exclude = ["target", "Systolic_Age_risk", "Diastolic_Age_risk", 
                  "Metabolic_Syndrome_Score", "Low_Risk_Paradox_Score"]
        artifacts["scalers"] = self.scale_features(exclude_cols=exclude)
        
        # Report
        if save_report:
            artifacts["report"] = self.generate_report()
            self.save_report(artifacts["report"])
        
        artifacts["data"] = self.df
        logger.info("Pipeline completed successfully")
        
        return artifacts
    
    def save_processed_data(self, filename: str = "processed_data.parquet") -> Path:
        """Save processed data."""
        if self.df is None:
            raise ValueError("No data to save")
        
        out = self.data_processed_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.df.to_parquet(out, index=False)
            logger.info("Saved to %s", out)
        except Exception as e:
            logger.warning("Parquet failed (%s). Using CSV.", e)
            out = out.with_suffix(".csv")
            self.df.to_csv(out, index=False)
            logger.info("Saved to %s", out)
        
        return out


if __name__ == "__main__":
    handler = DataHandler(random_state=42)
    
    try:
        artifacts = handler.run_pipeline(
            "processed_data.parquet", 
            file_type="processed",
            save_report=True
        )
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        print(f"Shape: {artifacts['data'].shape}")
        print(f"Columns: {list(artifacts['data'].columns)}")
        print(f"Missing: {artifacts['report']['missing_values']['total']}")
        print(f"Outliers handled: {artifacts['outliers_report']['total_handled']}")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)