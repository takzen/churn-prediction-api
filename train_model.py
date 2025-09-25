# --- train_model.py ---
# This script trains the model and saves the artifacts (model and scaler).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

print("Starting model training process...")

# --- 1. Load Data ---
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# --- 2. Preprocessing Pipeline ---
# Drop customerID
df_processed = df.drop('customerID', axis=1)
# Clean TotalCharges
df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
df_processed.dropna(subset=['TotalCharges'], inplace=True)
# Convert Churn
df_processed['Churn'] = df_processed['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
# One-Hot Encoding
categorical_cols = df_processed.select_dtypes(include='object').columns
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

# --- 3. Define X and y, and Split Data ---
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Scale Data ---
scaler = StandardScaler()
# Important: We are fitting and transforming the entire training set for the final model.
# In a real scenario, you'd have a separate validation set, but for deployment, we use all train data.
X_train_scaled = scaler.fit_transform(X_train)

# --- 5. Train the Model ---
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("Model training complete.")

# --- 6. Save Artifacts ---
artifacts_dir = 'model_artifacts'
os.makedirs(artifacts_dir, exist_ok=True)

joblib.dump(model, os.path.join(artifacts_dir, 'model.joblib'))
joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.joblib'))

print(f"Model and scaler saved to '{artifacts_dir}' directory.")

# Save the column list from the training data for the API
joblib.dump(list(X_train.columns), os.path.join(artifacts_dir, 'training_columns.joblib'))

print("Training columns saved.")