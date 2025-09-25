# --- main.py ---
# This is our FastAPI application file.

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# --- 1. DEFINE A PYDANTIC MODEL FOR INPUT DATA ---
# This model defines the structure and data types of the input we expect.
# FastAPI will use this for automatic validation and documentation.
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# --- 2. CREATE THE FASTAPI APP OBJECT ---
app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API to predict customer churn using a Logistic Regression model.",
    version="0.1.0"
)

# --- 3. LOAD THE MODEL AND SCALER AT STARTUP ---
model = joblib.load('model_artifacts/model.joblib')
scaler = joblib.load('model_artifacts/scaler.joblib')

# We also need the column order from the training phase to create the dataframe correctly
TRAINING_COLUMNS = joblib.load('model_artifacts/training_columns.joblib')

# --- 4. DEFINE THE PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    """
    Receives customer data, preprocesses it, and returns a churn prediction.
    """
    # Convert the input Pydantic model to a dictionary
    input_data = customer_data.model_dump()
    
    # Convert the dictionary to a pandas DataFrame
    input_df = pd.DataFrame([input_data])
    
    # --- PREPROCESSING ---
    # One-Hot Encode the new data
    input_df_encoded = pd.get_dummies(input_df)
    
    # Align columns with the training data: add missing columns, remove extra ones
    input_df_aligned = input_df_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    
    # Scale the numerical features using the loaded scaler
    input_df_scaled = scaler.transform(input_df_aligned)
    
    # --- PREDICTION ---
    prediction = model.predict(input_df_scaled)
    probability = model.predict_proba(input_df_scaled)
    
    # Convert prediction to a human-readable format
    churn_status = "Yes" if prediction[0] == 1 else "No"
    churn_probability = probability[0][1] # Probability of the 'Yes' class
    
    return {
        "prediction": churn_status,
        "probability_of_churn": f"{churn_probability:.2%}"
    }

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Churn Prediction API!"}

