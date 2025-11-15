Heart Disease Prediction

ML-powered cardiovascular disease risk prediction with explainable AI.

Dataset

Kaggle - Cardiovascular Disease Dataset
70,000 patient records
link - https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Performance

- AUC: 0.800
- Sensitivity: 98.1%
- False Negative Rate: 1.9%

Features

- XGBoost, LightGBM, CatBoost models
- SHAP explainability
- 3-tier risk stratification
- Gender-fair predictions

Installation

git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
conda env create -f environment.yaml
conda activate heart_disease

Usage

python -m src.pipeline.trainer_pipeline

Tech Stack

Python | scikit-learn | CatBoost | SHAP | Optuna

Limitations
⚠️ Model performs poorly on Normal/Elevated BP groups:

Normal BP: Recall 11.6%, FNR 88.4%

Elevated BP: Recall 27.3%, FNR 72.7%

Model is optimized for high-risk patients (Stage 2+ HTN) and may miss early-stage disease. Not suitable for screening low/normal BP populations.

Disclaimer

 Research tool only. Not for clinical use. Requires physician validation.

Contact

gmial - sekher157@gmail.com
linkedln - https://www.linkedin.com/in/chandra-sekhar-89159a266/
