🏦 Credit Scoring Classification Model

This project builds a machine learning pipeline to assess **creditworthiness** of customers based on financial and behavioral features. It predicts the likelihood that a customer will **default** on a loan, using classification models like Logistic Regression, Decision Tree, and Random Forest.

## 📌 Project Overview

The goal is to predict whether a customer is creditworthy (label `1`) or not (label `0`). The dataset is synthetically generated and includes features such as income, total debt, credit history, and payment behavior. The project includes:

* Data preprocessing and feature engineering
* Model training and comparison
* Hyperparameter tuning using GridSearchCV
* Model evaluation with performance metrics
* Model saving (deployment-ready)
* Function for real-time prediction on new customer data

## 📊 Features Used

| Feature              | Description                           |
| -------------------- | ------------------------------------- |
| `income`             | Annual income of the customer         |
| `total_debt`         | Total outstanding debt                |
| `credit_age`         | Age of credit history in years        |
| `credit_inquiries`   | Number of credit inquiries made       |
| `on_time_payments`   | Number of on-time payments            |
| `late_payments`      | Number of late payments               |
| `credit_used`        | Current credit used                   |
| `credit_limit`       | Total credit limit available          |
| `employment_status`  | Categorical: employed/unemployed      |
| `home_ownership`     | Categorical: rent/own                 |
| `debt_to_income`     | Derived: total\_debt / income         |
| `payment_ratio`      | Derived: on\_time / (on\_time + late) |
| `credit_utilization` | Derived: credit\_used / credit\_limit |

## 🧠 ML Models Used

* **Logistic Regression**
* **Decision Tree**
* **Random Forest (Best Performer)**

The final model is tuned using `GridSearchCV` and evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC Score
* Confusion Matrix
* Feature Importance

## 📁 Project Structure
credit_scoring_model/
├── credit_model.py                # Main script
├── credit_scoring_model.pkl       # Saved best model with scaler and features
├── README.md                      # Project documentation

## 🧪 Sample Prediction Function

You can use the trained model to predict new customer creditworthiness like this:

python
from credit_model import predict_credit_worthiness

sample_customer = {
    'income': 80000,
    'total_debt': 30000,
    'credit_age': 6,
    'credit_inquiries': 2,
    'debt_to_income': 0.38,
    'payment_ratio': 0.92,
    'credit_utilization': 0.45,
    'employment_status_employed': 1,
    'employment_status_unemployed': 0,
    'home_ownership_own': 0,
    'home_ownership_rent': 1
}

print(predict_credit_worthiness(sample_customer))

This will return a **probability score** between 0 and 1. Higher scores indicate stronger creditworthiness.

---

## 📈 Model Performance (Sample Output)

Model Performance Comparison:
                   Accuracy  Precision  Recall  F1 Score  ROC AUC
Random Forest         1.0        1.0      1.0     1.0      1.0
Logistic Regression   0.9        0.8      1.0     0.89     0.98
Decision Tree         0.8        0.67     1.0     0.8      0.93




