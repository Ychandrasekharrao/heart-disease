# data_handler.py

"""
Data Handler Component
Comprehensive data preprocessing pipeline for heart disease dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
from scipy import stats
import glob
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataHandler:
    """
    Comprehensive data preprocessing handler for heart disease dataset
    """
    
    def __init__(self, project_root=None, random_state=42):
        """
        Initialize DataHandler
        
        Args:
            project_root (Path/str): Project root directory
            random_state (int): Random seed for reproducibility
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.data_raw_dir = self.project_root / 'data' / 'raw'
        self.data_processed_dir = self.project_root / 'data' / 'processed'
        self.artifacts_dir = self.project_root / 'artifacts'
        self.reports_dir = self.project_root / 'reports'
        
        # Create directories
        for directory in [self.data_raw_dir, self.data_processed_dir, self.artifacts_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Configuration with your model weights for feature importance
        self.config = {
            'missing_value_strategy': {
                'numeric': 'median',  # 'mean', 'median', 'knn'
                'categorical': 'most_frequent'
            },
            'outlier_strategy': {
                'method': 'iqr',  # 'iqr', 'zscore'
                'threshold': 1.5
            },
            'scaling_strategy': 'robust',  # 'standard', 'robust', 'minmax'
            'categorical_encoding': 'label',  # 'label', 'onehot'
            'feature_weights': {
                'Systolic_BP': 0.466143,
                'Diastolic_BP': 0.399695,
                'Age_Years': 0.361310,
                'Pulse_Pressure': 0.347211,
                'BMI': 0.179270,
                'Cholesterol_Level': 0.111671,
                'Physical_Activity': -0.086953,
                'Alcohol_Intake': -0.035564,
                'Smoking_Status': -0.032207,
                'Glucose_Level': -0.019720,
                'Sex': 0.004828,
                'BP_level': -0.001939
            }
        }
        
        logger.info(f"DataHandler initialized with random_state={random_state}")
    
    def _find_file_case_insensitive(self, filename, directory):
        """Find file with case-insensitive matching"""
        pattern = str(directory / '*')
        files = glob.glob(pattern, recursive=False)
        filename_lower = filename.lower()
        
        for file_path in files:
            if Path(file_path).name.lower() == filename_lower:
                return Path(file_path)
        return None
    
    def load_data(self, file_path, file_type='raw'):
        """
        Load data from various file formats with case-insensitive matching
        
        Args:
            file_path (str): Path to data file
            file_type (str): 'raw' or 'processed'
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if file_type == 'raw':
                data_dir = self.data_raw_dir
            else:
                data_dir = self.data_processed_dir
            
            # Try exact match first
            full_path = data_dir / file_path
            
            if not full_path.exists():
                # Try case-insensitive matching
                found_path = self._find_file_case_insensitive(file_path, data_dir)
                if found_path:
                    full_path = found_path
                    logger.info(f"Found file with case-insensitive matching: {found_path}")
                else:
                    raise FileNotFoundError(f"Data file not found: {full_path}")
            
            # Determine file type and load
            if full_path.suffix.lower() == '.parquet':
                self.df = pd.read_parquet(full_path)
            elif full_path.suffix.lower() == '.csv':
                self.df = pd.read_csv(full_path)
            else:
                raise ValueError(f"Unsupported file format: {full_path.suffix}")
            
            logger.info(f"✅ Loaded data from {full_path} - Shape: {self.df.shape}")
            
            # Apply initial transformations
            self._apply_initial_transformations()
            
            return self.df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def _apply_initial_transformations(self):
        """Apply initial data transformations after loading"""
        # Age normalization: Convert from days to years if needed
        if 'age' in self.df.columns and self.df['age'].max() > 1000:
            logger.info("Converting age from days to years")
            self.df['Age_Years'] = (self.df['age'] / 365.25).round().astype(int)
            # Keep original age column for reference
            self.df['age_days'] = self.df['age']
            self.df = self.df.drop('age', axis=1)
        
        # Sex decoding
        if 'gender' in self.df.columns or 'sex' in self.df.columns:
            col_name = 'gender' if 'gender' in self.df.columns else 'sex'
            if self.df[col_name].dtype in [np.int64, np.float64]:
                logger.info("Decoding sex/gender to categorical")
                sex_mapping = {1: 'Male', 2: 'Female', 0: 'Unknown'}
                self.df[col_name] = self.df[col_name].map(sex_mapping).fillna('Unknown')
        
        logger.info("Applied initial transformations")
    
    def handle_missing_values(self, strategy=None):
        """
        Handle missing values in the dataset with reproducibility
        
        Args:
            strategy (dict): Override default missing value strategy
            
        Returns:
            dict: Imputer objects for model pipeline
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        strategy = strategy or self.config['missing_value_strategy']
        
        logger.info("Handling missing values...")
        
        # Report initial missing values
        missing_before = self.df.isnull().sum().sum()
        missing_report = {}
        
        if missing_before > 0:
            logger.info(f"Missing values before handling: {missing_before}")
            missing_by_col = self.df.isnull().sum()
            cols_with_missing = missing_by_col[missing_by_col > 0]
            missing_report['before'] = dict(cols_with_missing)
            logger.info(f"Columns with missing values: {dict(cols_with_missing)}")
        
        # Separate numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numeric missing values
        if numeric_cols and strategy['numeric'] != 'none':
            if strategy['numeric'] == 'knn':
                imputer = KNNImputer(n_neighbors=5, random_state=self.random_state)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
                self.imputers['numeric_knn'] = imputer
            else:
                imputer = SimpleImputer(strategy=strategy['numeric'], random_state=self.random_state)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
                self.imputers['numeric'] = imputer
            logger.info(f"Applied {strategy['numeric']} imputation to numeric columns")
        
        # Handle categorical missing values
        if categorical_cols and strategy['categorical'] != 'none':
            imputer = SimpleImputer(strategy=strategy['categorical'], fill_value='Unknown')
            self.df[categorical_cols] = imputer.fit_transform(self.df[categorical_cols])
            self.imputers['categorical'] = imputer
            logger.info(f"Applied {strategy['categorical']} imputation to categorical columns")
        
        # Report results
        missing_after = self.df.isnull().sum().sum()
        missing_report['after'] = dict(self.df.isnull().sum())
        missing_report['total_handled'] = missing_before - missing_after
        
        logger.info(f"Missing values after handling: {missing_after}")
        
        return self.imputers
    
    def handle_outliers(self, strategy=None):
        """
        Handle outliers in numeric columns
        
        Args:
            strategy (dict): Override default outlier strategy
            
        Returns:
            dict: Outlier handling statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        strategy = strategy or self.config['outlier_strategy']
        
        logger.info("Handling outliers...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from outlier handling if it exists
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        outliers_report = {}
        outliers_handled = 0
        
        for col in numeric_cols:
            col_outliers = 0
            
            if strategy['method'] == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - strategy['threshold'] * IQR
                upper_bound = Q3 + strategy['threshold'] * IQR
                
                # Cap outliers instead of removing them
                outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                    outliers_handled += outliers_count
                    col_outliers = outliers_count
                    logger.debug(f"Capped {outliers_count} outliers in {col}")
            
            elif strategy['method'] == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col]))
                outliers_mask = z_scores > strategy['threshold']
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    # Replace with median
                    median_val = self.df[col].median()
                    self.df.loc[outliers_mask, col] = median_val
                    outliers_handled += outliers_count
                    col_outliers = outliers_count
                    logger.debug(f"Replaced {outliers_count} outliers in {col} with median")
            
            outliers_report[col] = col_outliers
        
        outliers_report['total_handled'] = outliers_handled
        logger.info(f"Handled {outliers_handled} outliers using {strategy['method']} method")
        
        return outliers_report
    
    def encode_categorical_variables(self, strategy=None):
        """
        Encode categorical variables
        
        Args:
            strategy (str): Override default encoding strategy
            
        Returns:
            dict: Encoder objects for model pipeline
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        strategy = strategy or self.config['categorical_encoding']
        
        logger.info("Encoding categorical variables...")
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if strategy == 'label':
                encoder = LabelEncoder()
                self.df[col] = encoder.fit_transform(self.df[col])
                self.encoders[col] = encoder
                logger.debug(f"Label encoded {col}")
            
            elif strategy == 'onehot':
                # For one-hot encoding, we'll create new columns
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
                logger.debug(f"One-hot encoded {col} into {len(dummies.columns)} columns")
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns using {strategy} encoding")
        return self.encoders
    
    def scale_features(self, strategy=None, exclude_cols=None):
        """
        Scale numeric features
        
        Args:
            strategy (str): Scaling strategy
            exclude_cols (list): Columns to exclude from scaling
            
        Returns:
            dict: Scaler objects for model pipeline
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        strategy = strategy or self.config['scaling_strategy']
        exclude_cols = exclude_cols or ['target'] if 'target' in self.df.columns else []
        
        logger.info(f"Scaling features using {strategy} strategy...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not scale_cols:
            logger.warning("No numeric columns to scale")
            return {}
        
        if strategy == 'standard':
            scaler = StandardScaler()
        elif strategy == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {strategy}")
        
        self.df[scale_cols] = scaler.fit_transform(self.df[scale_cols])
        self.scalers[strategy] = scaler
        
        logger.info(f"Scaled {len(scale_cols)} numeric columns using {strategy} scaler")
        return self.scalers
    
    def generate_features(self):
        """
        Generate new features from existing ones
        
        Returns:
            dict: Generated features information
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Generating new features...")
        
        generated_features = {}
        
        # BMI Categories (if height and weight exist)
        if all(col in self.df.columns for col in ['height', 'weight']):
            self.df['BMI'] = self.df['weight'] / ((self.df['height'] / 100) ** 2)
            
            # BMI Categories
            bmi_bins = [0, 18.5, 25, 30, 35, float('inf')]
            bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II+']
            self.df['BMI_Category'] = pd.cut(self.df['BMI'], bins=bmi_bins, labels=bmi_labels)
            generated_features['BMI'] = 'Generated BMI from height/weight'
            generated_features['BMI_Category'] = 'BMI risk categories'
            logger.info("Generated BMI and BMI_Category features")
        
        # Blood Pressure Classification
        bp_cols = ['ap_hi', 'Systolic_BP']
        systolic_col = next((col for col in bp_cols if col in self.df.columns), None)
        diastolic_col = 'ap_lo' if 'ap_lo' in self.df.columns else 'Diastolic_BP'
        
        if systolic_col and diastolic_col in self.df.columns:
            conditions = [
                (self.df[systolic_col] < 120) & (self.df[diastolic_col] < 80),
                (self.df[systolic_col].between(120, 129)) & (self.df[diastolic_col] < 80),
                (self.df[systolic_col].between(130, 139)) | (self.df[diastolic_col].between(80, 89)),
                (self.df[systolic_col] >= 140) | (self.df[diastolic_col] >= 90)
            ]
            choices = ['Normal', 'Elevated', 'Stage 1', 'Stage 2+']
            self.df['BP_Label'] = np.select(conditions, choices, default='Unknown')
            generated_features['BP_Label'] = 'Blood pressure classification'
            logger.info("Generated BP_Label feature")
        
        # Pulse Pressure and risk bands
        if systolic_col and diastolic_col in self.df.columns:
            self.df['Pulse_Pressure'] = self.df[systolic_col] - self.df[diastolic_col]
            
            # Pulse Pressure risk bands
            pp_bins = [0, 40, 60, float('inf')]
            pp_labels = ['Low', 'Normal', 'High']
            self.df['Pulse_Pressure_Risk'] = pd.cut(
                self.df['Pulse_Pressure'], bins=pp_bins, labels=pp_labels
            )
            generated_features['Pulse_Pressure'] = 'Systolic - Diastolic BP'
            generated_features['Pulse_Pressure_Risk'] = 'Pulse pressure risk bands'
            logger.info("Generated Pulse_Pressure features with risk bands")
        
        # Age Groups
        age_col = 'Age_Years' if 'Age_Years' in self.df.columns else 'age'
        if age_col in self.df.columns:
            age_bins = [0, 40, 50, 60, 70, 100]
            age_labels = ['<40', '40-49', '50-59', '60-69', '70+']
            self.df['Age_Group'] = pd.cut(self.df[age_col], bins=age_bins, labels=age_labels)
            generated_features['Age_Group'] = 'Age categories'
            logger.info("Generated Age_Group feature")
        
        # Risk Score (weighted by model coefficients)
        risk_score = 0
        weight_sum = 0
        
        for feature, weight in self.config['feature_weights'].items():
            if feature in self.df.columns and self.df[feature].dtype in [np.int64, np.float64]:
                # Normalize feature first
                feature_normalized = (self.df[feature] - self.df[feature].mean()) / self.df[feature].std()
                risk_score += feature_normalized * weight
                weight_sum += abs(weight)
        
        if weight_sum > 0:
            self.df['Weighted_Risk_Score'] = risk_score / weight_sum
            generated_features['Weighted_Risk_Score'] = 'Weighted risk score using model coefficients'
            logger.info("Generated Weighted_Risk_Score using model coefficients")
        
        logger.info(f"Total features after generation: {self.df.shape[1]}")
        return generated_features
    
    def generate_report(self):
        """
        Generate comprehensive data quality report
        
        Returns:
            dict: Report data
        """
        if self.df is None:
            return {"error": "No data available for report"}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': self.df.shape,
            'data_info': self.get_data_info(),
            'missing_values': {
                'total': self.df.isnull().sum().sum(),
                'by_column': dict(self.df.isnull().sum())
            },
            'skewness': {},
            'feature_importance': self.config['feature_weights'],
            'data_quality_metrics': {}
        }
        
        # Calculate skewness for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report['skewness'][col] = {
                'skew': float(self.df[col].skew()),
                'interpretation': self._interpret_skewness(self.df[col].skew())
            }
        
        # Data quality metrics
        report['data_quality_metrics'] = {
            'completeness': 1 - (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])),
            'duplicates': self.df.duplicated().sum(),
            'constant_columns': len([col for col in self.df.columns if self.df[col].nunique() <= 1])
        }
        
        return report
    
    def _interpret_skewness(self, skew_value):
        """Interpret skewness value"""
        if abs(skew_value) < 0.5:
            return "Approximately symmetric"
        elif abs(skew_value) < 1:
            return "Moderately skewed"
        else:
            return "Highly skewed"
    
    def save_report(self, report, filename="data_quality_report.json"):
        """Save report to JSON file"""
        report_path = self.reports_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📊 Report saved to: {report_path}")
        return report_path
    
    def run_full_pipeline(self, file_path, file_type='raw', save_cleaned=True, generate_report=True):
        """
        Run complete data preprocessing pipeline
        
        Args:
            file_path (str): Path to data file
            file_type (str): 'raw' or 'processed'
            save_cleaned (bool): Whether to save cleaned data
            generate_report (bool): Whether to generate data quality report
            
        Returns:
            dict: Pipeline artifacts (transformers, report, data)
        """
        logger.info("🚀 Starting full data preprocessing pipeline...")
        
        pipeline_artifacts = {
            'transformers': {},
            'report': {},
            'data_info': {},
            'generated_features': {}
        }
        
        try:
            # 1. Load data
            self.load_data(file_path, file_type)
            
            # 2. Handle missing values
            pipeline_artifacts['transformers']['imputers'] = self.handle_missing_values()
            
            # 3. Handle outliers
            pipeline_artifacts['outliers_report'] = self.handle_outliers()
            
            # 4. Encode categorical variables
            pipeline_artifacts['transformers']['encoders'] = self.encode_categorical_variables()
            
            # 5. Generate new features
            pipeline_artifacts['generated_features'] = self.generate_features()
            
            # 6. Scale features (exclude target)
            pipeline_artifacts['transformers']['scalers'] = self.scale_features(exclude_cols=['target'])
            
            # 7. Generate report
            if generate_report:
                pipeline_artifacts['report'] = self.generate_report()
                self.save_report(pipeline_artifacts['report'])
            
            # 8. Save cleaned data
            if save_cleaned:
                output_path = self.save_cleaned_data()
                pipeline_artifacts['output_path'] = str(output_path)
                logger.info(f"✅ Pipeline completed. Cleaned data saved to: {output_path}")
            else:
                logger.info("✅ Pipeline completed.")
            
            pipeline_artifacts['data_info'] = self.get_data_info()
            pipeline_artifacts['data'] = self.df
            
            return pipeline_artifacts
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def save_cleaned_data(self, filename="cleaned_data.parquet"):
        """Save cleaned data to processed directory"""
        if self.df is None:
            raise ValueError("No data to save. Run preprocessing pipeline first.")
        
        output_path = self.data_processed_dir / filename
        self.df.to_parquet(output_path, index=False)
        
        logger.info(f"💾 Saved cleaned data to: {output_path}")
        logger.info(f"📊 Final data shape: {self.df.shape}")
        
        return output_path
    
    def get_data_info(self):
        """Get information about current dataset"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "data_types": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            "numeric_columns": self.df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
        }
        
        return info
    
    # Unit test hooks
    def test_missing_values(self):
        """Test that no missing values remain"""
        if self.df is None:
            return False, "No data loaded"
        
        missing_count = self.df.isnull().sum().sum()
        return missing_count == 0, f"Missing values: {missing_count}"
    
    def test_data_types(self):
        """Test that all data types are appropriate"""
        if self.df is None:
            return False, "No data loaded"
        
        # Check for object dtypes that should be numeric
        object_cols = self.df.select_dtypes(include=['object']).columns
        issues = []
        
        for col in object_cols:
            # Try to convert to numeric
            try:
                pd.to_numeric(self.df[col])
                issues.append(f"Column '{col}' could be numeric")
            except:
                pass
        
        return len(issues) == 0, f"Data type issues: {issues}"
    
    def test_feature_generation(self):
        """Test that expected features were generated"""
        expected_features = ['BMI', 'BMI_Category', 'BP_Label', 'Pulse_Pressure', 'Age_Group']
        generated = [f for f in expected_features if f in self.df.columns]
        
        return len(generated) > 0, f"Generated features: {generated}"


# Example usage and demonstration
if __name__ == "__main__":
    print("🔧 Enhanced Data Handler Component Demo")
    print("=" * 50)
    
    # Initialize handler with random seed
    handler = DataHandler(random_state=42)
    
    try:
        # Run full pipeline with all features
        print("\n🚀 Running full preprocessing pipeline...")
        artifacts = handler.run_full_pipeline(
            file_path="heart disease.parquet",  # Case-insensitive matching
            file_type='raw',
            save_cleaned=True,
            generate_report=True
        )
        
        # Display results
        print(f"\n✅ Preprocessing completed!")
        print(f"📊 Final data shape: {artifacts['data_info']['shape']}")
        print(f"🔢 Columns: {len(artifacts['data_info']['columns'])}")
        
        # Show generated features
        if artifacts['generated_features']:
            print(f"🎯 Generated features: {list(artifacts['generated_features'].keys())}")
        
        # Run unit tests
        print("\n🔍 Running unit tests...")
        missing_test, missing_msg = handler.test_missing_values()
        dtype_test, dtype_msg = handler.test_data_types()
        feature_test, feature_msg = handler.test_feature_generation()
        
        print(f"✅ Missing values test: {'PASS' if missing_test else 'FAIL'} - {missing_msg}")
        print(f"✅ Data types test: {'PASS' if dtype_test else 'FAIL'} - {dtype_msg}")
        print(f"✅ Feature generation test: {'PASS' if feature_test else 'FAIL'} - {feature_msg}")
        
        # Show pipeline artifacts
        print(f"\n🛠️  Pipeline artifacts:")
        print(f"   - Transformers: {list(artifacts['transformers'].keys())}")
        print(f"   - Report: {artifacts['report'].get('timestamp', 'Generated')}")
        print(f"   - Output: {artifacts.get('output_path', 'Not saved')}")
        
        print(f"\n💡 Enhanced usage examples:")
        print("""
        # Reproducible pipeline with random seed
        handler = DataHandler(random_state=42)
        artifacts = handler.run_full_pipeline('data.csv')
        
        # Access transformers for model pipeline
        imputers = artifacts['transformers']['imputers']
        scalers = artifacts['transformers']['scalers']
        
        # Generate custom report
        report = handler.generate_report()
        handler.save_report(report, 'custom_report.json')
        
        # Run unit tests
        handler.test_missing_values()
        handler.test_data_types()
        """)
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        print("\n💡 The handler will automatically try case-insensitive matching")
        
    except Exception as e:
        print(f"❌ Error: {e}")