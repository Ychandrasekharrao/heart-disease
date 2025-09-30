# Heart Disease Prediction Project

## Overview
This project aims to predict heart disease using machine learning algorithms. The model is trained on clinical parameters to predict the likelihood of heart disease in patients.

## Project Organization
```
heart_disease_prediction/
├── data/                         # All project data
│   ├── raw/                      # Original, immutable data
│   ├── processed/                # Cleaned and processed data
│   └── external/                 # External source data if any
│
├── models/                       # Trained and serialized models
│
├── notebooks/                    # Jupyter notebooks for exploration & communication
│   ├── 1.0-data-exploration-and-features.ipynb
│   ├── 2.0-model-development.ipynb
│   └── 3.0-model-evaluation.ipynb
│
├── reports/                      # Generated analysis reports and visualizations
│   ├── eda/                      # EDA-specific plots and charts
│   ├── figures/                  # Other generated graphics and figures
│   ├── metrics/                  # Model performance metrics
│   └── final_report.pdf          # Optional: compiled final report
│
├── src/                          # Source code for use in this project
│   ├── data/                     # Scripts for data processing
│   ├── models/                   # Scripts for model training and prediction
│   └── visualization/            # Scripts for creating visualizations
│
├── tests/                        # Unit tests
├── environment.yml               # Conda environment file
├── setup.py                      # Make project pip installable
└── config.yaml                   # Configuration parameters
```

## Setup
1. Clone this repository
2. Create the conda environment:
```bash
conda env create -f environment.yml
```
3. Activate the environment:
```bash
conda activate heart_disease_env
```

## Usage
1. Place the raw data in `data/raw/`
2. Run the notebooks in order:
   - Data Exploration
   - Feature Engineering
   - Model Development

## Project Status
[In Development]
