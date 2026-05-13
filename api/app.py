# api/app.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------
pipeline = joblib.load("artifacts/pipeline.pkl")

# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predict whether a telecom customer is likely to churn."
)

# --------------------------------------------------
# Request schema
# --------------------------------------------------
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

# --------------------------------------------------
# Health endpoint
# --------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(customer: CustomerData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([customer.model_dump()])

    # Predict label
    prediction = pipeline.predict(input_df)[0]

    # Predict probability
    probability = pipeline.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }