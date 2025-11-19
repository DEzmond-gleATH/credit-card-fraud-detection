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
````

## Project Overview

The main notebook (`notebooks/credit_card_fraud_detection.ipynb`) explores:

* Data loading and preprocessing (using `pandas` and `numpy`)
* Handling class imbalance (using `RandomOverSampler` from `imbalanced-learn`)
* Feature scaling (using `StandardScaler`)
* Model building and evaluation with:

  * Decision Tree
  * Random Forest
  * Bagging Classifier
  * XGBoost (if installed)
* Metrics such as accuracy, classification report, and confusion matrix
* Basic clustering and visualization using `KMeans`, `matplotlib`, and `seaborn`

The script `src/train_models.py` contains a clean, reproducible training pipeline that:

* Loads the dataset from `data/crowd.csv`
* Splits into stratified train/test sets
* Scales features and applies oversampling
* Trains multiple models and prints evaluation metrics

## Dataset

The code expects a file named:

* `data/crowd.csv`

Place the file in the **data/** folder in the project root.

> **Note:** If this dataset comes from a course or a private source, it may not be committed to the repository. In that case, clone the repo and place your own `crowd.csv` in `data/`.

## How to Run

1. **Clone this repository**

   ```bash
   git clone https://github.com/DEzmond-gleATH/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **(Optional) Create and activate a virtual environment**

   ```bash
   python -m venv .venv

   # Windows:
   .venv\Scripts\activate

   # macOS / Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add the dataset**

   Place your `crowd.csv` file into the `data/` folder.

5. **Run the training script**

   ```bash
   python src/train_models.py
   ```

6. **Explore the notebook (optional)**

   ```bash
   jupyter notebook
   ```

   Then open:

   * `notebooks/credit_card_fraud_detection.ipynb`
