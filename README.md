# Breast Cancer Prediction with XGBoost

## Description
This project implements a machine learning pipeline to predict breast cancer using the scikit-learn breast cancer dataset. 
The pipeline is modular, reproducible, and includes automatic evaluation with confusion matrix and ROC curve.

## Project Goal
- Build a reproducible ML pipeline for breast cancer prediction
- Use best practices: modular code, environment setup, and evaluation
- Demonstrate understanding of data preprocessing, model training, and evaluation metrics

## Project Structure
CodeAlpha_disease-prediction-ml/
│
├── data/ # Dataset (raw or processed)
├── src/ # Source code
│ ├── data/ # Preprocessing
│ │ └── preprocess.py
│ ├── models/ # Training and evaluation
│ │ ├── train.py
│ │ └── evaluate.py
│ └── utils/ # Utility functions
│ └── seed.py
├── notebooks/ # Experimental notebook
├── outputs/ # Generated outputs
│ ├── saved_models/ # Trained model files
│ └── figures/ # Confusion matrix and ROC curve
├── configs/ # Config files (optional)
├── experiments/ # Experiment logs (optional)
├── requirements.txt # Required Python packages
└── README.md # This file


## Installation

1. Clone the repository:
```bash
git clone https://github.com/NawresFekih/CodeAlpha_disease-prediction-ml.git
cd CodeAlpha_disease-prediction-ml
``` 

2. Create and activate a virtual environment:
```bash
python -m venv cfeproj
# Windows
cfeproj\Scripts\activate
# Linux / macOS
source cfeproj/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage

Train model
```bash
python -m src.models.train
```
Evaluate model
```bash
python -m src.models.evaluate
```

This will generate:

- Trained model in outputs/saved_models/
- Confusion matrix and ROC curve in outputs/figures/

## Expected Results

- Accuracy: ~0.97
- F1 Score: ~0.96
- Confusion matrix saved as outputs/figures/confusion_matrix.png
- ROC curve saved as outputs/figures/roc_curve.png (AUC ~0.97)

## Reproducibility

. Random seed is fixed using src/utils/seed.py
. Dependencies listed in requirements.txt
. Modular pipeline ensures consistent results across machines

## Contact
GitHub: [NawresFekih](https://github.com/NawresFekih)
Email: nawresfekih3@gmail.com