# Heart Disease Risk Prediction

Calibrated CatBoost model for cardiovascular disease risk prediction with explainability, clinical evaluation, and fairness analysis.

## Summary
- CatBoost classifier + Beta Calibration
- SHAP-based explainability
- Threshold optimization (Max F1)
- Risk stratification (Low / Moderate / High)
- Fairness analysis by gender and Age × BP subgroups
- Reproducible pipeline and artifacts for training and inference

## Dataset
- Source: Kaggle — Cardiovascular Disease Dataset  
  https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset  
- ~70,000 patient records (BP, cholesterol, glucose, BMI, lifestyle, demographics)

## Key Features
- CatBoost classifier with Beta calibration
- Threshold optimization and clinical metrics (Sensitivity, Specificity, PPV, NPV)
- SHAP explainability and risk‑group analysis
- Fairness reporting (gender and Age × BP subgroup metrics)
- Modular pipeline for training, evaluation, and inference

## Performance (example / reported)
- AUC-ROC: 0.801  
- PR-AUC: 0.784  
- Brier score: 0.180 (Beta Calibration)
- Optimal threshold: 0.381
- Accuracy: 72.1% | F1: 0.742
- Sensitivity (Recall): 80.2% | Specificity: 64.0%
- PPV: 69.0% | NPV: 76.4%

Confusion matrix (example)
- TN: 6,717 | FP: 3,784
- FN: 2,075 | TP: 8,417

Risk-group notes
- Moderate group shows higher FP rates and lower specificity
- Low / Elevated BP subgroups show markedly reduced recall

## Fairness & Subgroup Performance
- Gender: small differences in sensitivity/specificity; PPV differences observed
- Age × BP:
  - Normal BP: AUC ≈ 0.66, Recall ≈ 11.6%
  - Elevated BP: AUC ≈ 0.71, Recall ≈ 27.3%
  - Stage-1/Stage-2 HTN: AUC 0.83–0.89+

## Limitations
- Poor recall in Normal / Elevated BP groups (not suitable for screening low-risk populations)
- Possible dataset biases and limited generalizability across regions
- Requires prospective, multi-center validation before clinical use

## Installation
git clone https://github.com/Ychandrasekharrao/heart-disease.git  
cd heart-disease-prediction

conda env create -f environment.yaml  
conda activate heart_disease

## Usage
Train:
python -m src.pipeline.trainer_pipeline

Predict:
python -m src.pipeline.predict_pipeline

## Repository structure
heart-disease/
├── src/
│   ├── components/      # data handlers, modeling, evaluation
│   ├── pipeline/        # training + inference pipelines
│   └── utils/           # logging, helpers
├── notebooks/           # EDA, modeling, RAI dashboard
├── reports/             # plots, metrics, fairness analysis
├── artifacts/           # trained models, splits, calibration files
├── data/
│   ├── raw/
│   └── processed/
├── environment.yaml
├── setup.py
└── README.md

## Disclaimer
Research tool only. Not for clinical use. Requires physician validation and regulatory approval for any clinical deployment.

## Contact
Email: sekher157@gmail.com  
LinkedIn: https://www.linkedin.com/in/chandra-sekhar-89159a266/  
GitHub: https://github.com/Ychandrasekharrao/heart-disease
