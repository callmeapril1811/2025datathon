# 2025datathon


# Credit Line Increase & Risk Segmentation Pipeline

This part builds a machine learning pipeline to classify accounts by risk and recommend personalized credit line increases. It is divided into two main stages:

---

## 1. Data Cleaning: `cleaning.py`

This script processes raw CSV files and prepares them for modeling. Each dataset is cleaned individually to remove noise, correct datatypes, and handle missing or anomalous values.

### Key Implementations:
- Drop irrelevant or duplicate columns.
- Convert date strings to datetime format.
- Map currency codes to country names.
- Remove or replace invalid placeholder values.
- Save cleaned datasets as new `.csv` files.

### Example Output:
- `account_dim_20250325_clean.csv`
- `fraud_claim_case_20250325_clean.csv`
- `fraud_claim_tran_20250325_clean.csv`
- `rams_batch_cur_20250325_clean.csv`
- `statement_fact_20250325_clean.csv`
- `syf_id_20250325_clean.csv`
- `transaction_fact_20250325_clean.csv`
- `wrld_stor_tran_fact_20250325_clean.csv`

Run this script once to generate all the cleaned files used in the modeling stage.

---

## 2. Modeling: `model.py`

This script performs feature engineering, trains two machine learning models, and evaluates their performance.

### Task 1: Classification
Segments each account into one of four risk categories:
- **0**: Low risk
- **1**: Moderate risk
- **2**: No increase needed
- **3**: High risk / Non-performing

*Model:* `XGBClassifier`

### Task 2: Regression
Estimates the optimal credit line increase amount for eligible accounts.

*Model:* `XGBRegressor`

### Key Implementation:
- Merge all cleaned datasets.
- Engineer date, delinquency, and fraud features.
- Define custom risk segmentation logic.
- Split data for training/testing and apply class balancing.
- Evaluate classification via accuracy score and confusion matrix.
- Evaluate regression via RMSE and scatter plot.
- Generate business metrics: revenue lift, fraud savings, and analyst capacity saved.

### Example Outputs:

**Classification Output:**
Classification Accuracy: 0.89 Classification Report: precision recall f1-score support

       0       0.92      0.90      0.91     12000
       1       0.88      0.87      0.87     10000
       2       0.85      0.88      0.86      8000
       3       0.90      0.89      0.89     11000

accuracy                           0.89     41000
Regression RMSE: 125.67

**Plot Output:**
After training and evaluating the models, several visualizations are generated to help you understand the performance and insights from the data. Below is a description of each plot and what you can learn from it.

### Feature Importance Plot
A bar chart ranking the top 10 features used by the XGBoost classifier.
*How to Interpret:*
Features with higher bars are more influential in the model’s decisions.

### Confusion Matrix Plot
What It Shows:
A heatmap comparing actual risk segments to predicted ones.
*How to Interpret:*
Diagonal cells represent correct predictions; off-diagonals indicate misclassifications.

### Actual vs. Predicted Scatter Plot (Regression)
A scatter plot of actual credit line increases versus predicted values with a 45-degree reference line.
*How to Interpret:*
Points near the line indicate accurate predictions; deviations show areas for improvement.

### Pie Chart of Account Risk Distribution
A pie chart displaying the percentage distribution of accounts across risk segments.
*How to Interpret:*
Use the percentages to understand the balance of low risk, moderate risk, no increase, and high risk accounts.


## How to Run
Run the data cleaning script:
$python cleaning.py

Once the cleaned files are saved, run each section in the modeling script:
Model.ipynb
