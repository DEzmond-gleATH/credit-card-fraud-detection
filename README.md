# Credit Card Fraud Detection (Machine Learning Project)

This project builds and evaluates machine learning models to detect fraudulent
credit card transactions on a highly imbalanced dataset.

## Objective

- Predict whether a transaction is **fraudulent (1)** or **legitimate (0)**.
- Handle **class imbalance** and compare different models for fraud detection.
- Show a clean, reproducible pipeline in both **notebook** and **Python script** form.

## Project Structure

```text
credit-card-fraud-detection/
├─ notebooks/
│   └─ credit_card_fraud_detection.ipynb    # end-to-end EDA + experiments
├─ src/
│   └─ train_models.py                      # reproducible training pipeline
├─ data/
│   └─ README.md                            # where to place `crowd.csv`
├─ reports/
│   └─ figures/                             # optional: saved plots
├─ requirements.txt
├─ README.md
└─ .gitignore

## Project Overview
The notebook (`Machine_learning_Final_Project.ipynb`) explores:
- Data loading and preprocessing (using `pandas` and `numpy`)
- Handling class imbalance (using `RandomOverSampler` from `imbalanced-learn`)
- Feature scaling (using `StandardScaler`)
- Model building and evaluation with:
  - Decision Tree
  - Random Forest
  - Bagging Classifier
  - XGBoost
- Metrics such as accuracy, classification report, and confusion matrix
- Basic clustering and visualization using `KMeans`, `matplotlib`, and `seaborn`

## How to Run
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
