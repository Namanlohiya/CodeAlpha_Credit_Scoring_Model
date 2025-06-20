import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, 
                             roc_auc_score, confusion_matrix,
                             classification_report)
import pickle
from io import StringIO

# ====================== 1. Create Synthetic Data ======================
credit_data = """income,total_debt,credit_age,credit_inquiries,on_time_payments,late_payments,credit_used,credit_limit,employment_status,home_ownership,default
75000,25000,5,2,45,5,4000,10000,employed,rent,0
50000,35000,3,4,30,10,6000,8000,employed,own,1
120000,15000,10,1,60,2,2000,20000,employed,own,0
85000,40000,7,3,50,8,8000,15000,unemployed,rent,1
60000,28000,4,5,35,12,5000,10000,employed,rent,0
150000,20000,12,0,70,1,3000,25000,employed,own,0
90000,50000,8,2,55,6,10000,18000,employed,own,0
55000,32000,2,6,25,15,7000,9000,unemployed,rent,1
130000,18000,15,1,65,3,2500,22000,employed,own,0
70000,38000,6,4,40,9,9000,12000,employed,rent,0
95000,42000,9,2,58,5,11000,17000,employed,own,0
45000,40000,1,7,20,18,7500,8500,unemployed,rent,1
110000,22000,11,1,62,4,3500,21000,employed,own,0
80000,35000,5,3,48,7,7000,14000,employed,rent,0
65000,30000,4,4,38,10,6000,11000,employed,rent,0
"""

# Load data from string
data = pd.read_csv(StringIO(credit_data))

# ====================== 2. Data Understanding ======================
print("\nData Overview:")
print(data.info())
print("\nData Statistics:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())

# ====================== 3. Feature Engineering ======================
print("\nPerforming feature engineering...")
# Create new features
data['debt_to_income'] = data['total_debt'] / data['income']
data['payment_ratio'] = data['on_time_payments'] / (data['on_time_payments'] + data['late_payments'])
data['credit_utilization'] = data['credit_used'] / data['credit_limit']

# Convert categorical variables
data = pd.get_dummies(data, columns=['employment_status', 'home_ownership'])

# Define target variable (1 = good credit, 0 = bad credit)
data['credit_worthy'] = np.where(data['default'] == 0, 1, 0)

# Select features and target
features = ['income', 'total_debt', 'credit_age', 'credit_inquiries',
            'debt_to_income', 'payment_ratio', 'credit_utilization',
            'employment_status_employed', 'employment_status_unemployed',
            'home_ownership_own', 'home_ownership_rent']
            
X = data[features]
y = data['credit_worthy']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================== 4. Model Building ======================
print("\nBuilding and evaluating models...")
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_prob)
    }

# Display results
results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df.sort_values(by='ROC AUC', ascending=False))

# ====================== 5. Model Evaluation ======================
print("\nEvaluating best model...")
best_model = RandomForestClassifier(n_estimators=100, max_depth=5)
best_model.fit(X_train_scaled, y_train)

# Feature importance
importances = best_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Confusion matrix
y_pred = best_model.predict(X_test_scaled)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ====================== 6. Model Optimization ======================
print("\nOptimizing model...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use 3 folds instead of 5 for such a small dataset
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("\nBest Parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]
print("Optimized ROC AUC:", roc_auc_score(y_test, y_prob))

# ====================== 7. Model Deployment ======================
print("\nSaving model for deployment...")
with open('credit_scoring_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_rf,
        'scaler': scaler,
        'features': features
    }, f)

def predict_credit_worthiness(customer_data):
    """Predict credit worthiness probability (0-1)"""
    with open('credit_scoring_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    input_df = pd.DataFrame([customer_data])
    input_df = input_df[artifacts['features']]
    input_scaled = artifacts['scaler'].transform(input_df)
    return artifacts['model'].predict_proba(input_scaled)[0, 1]

# Example prediction
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

print(f"\nSample prediction: {predict_credit_worthiness(sample_customer):.2f}")
