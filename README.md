# CodeAlpha_Task
Task 1: **Character Recognition using CNN on MNIST dataset**

# 🧠 Handwritten Character Recognition using CNN (MNIST)

This project is a deep learning-based character recognition system that uses a Convolutional Neural Network (CNN) to classify handwritten digits from the popular **MNIST** dataset. It is built using **TensorFlow** and **Keras**, and achieves high accuracy in recognizing digits (0–9) from 28x28 grayscale images.

## 📌 Project Overview

Handwritten digit recognition is one of the fundamental problems in computer vision. The MNIST dataset is a benchmark dataset consisting of 70,000 grayscale images of handwritten digits (60,000 for training and 10,000 for testing). This project implements a CNN to train on this dataset and evaluate its performance on unseen data.

## 🛠️ Technologies Used

* Python 🐍
* TensorFlow / Keras 📦
* NumPy 📊
* Matplotlib 📈
* MNIST Dataset 📚

## 🧮 Model Architecture

The Convolutional Neural Network (CNN) model architecture:
Input Layer: 28x28x1 grayscale images

1. Conv2D (32 filters, 3x3 kernel, ReLU activation)
2. MaxPooling2D (2x2)
3. Conv2D (64 filters, 3x3 kernel, ReLU activation)
4. MaxPooling2D (2x2)
5. Flatten
6. Dense (64 neurons, ReLU activation)
7. Dense (10 neurons, Softmax activation - for 10 digit classes)

## 📊 Results

* The model achieves an accuracy of over **98%** on the test dataset.
* A sample prediction is visualized using `matplotlib`, displaying both the true and predicted label.
* The trained model is saved as `mnist_cnn_model.h5`.

Task2: **Credit Scoring Classification Model**
# 🏦 Credit Scoring Classification Model

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

```bash
Model Performance Comparison:
                   Accuracy  Precision  Recall  F1 Score  ROC AUC
Random Forest         1.0        1.0      1.0     1.0      1.0
Logistic Regression   0.9        0.8      1.0     0.89     0.98
Decision Tree         0.8        0.67     1.0     0.8      0.93
```

---



