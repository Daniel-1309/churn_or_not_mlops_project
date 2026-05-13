# api/app.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

# Count predictions by class
prediction_counter = Counter(
    "churn_predictions_total",
    "Number of predictions by class",
    ["prediction"]
)

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

# Automatically collect metrics and expose /metrics
Instrumentator().instrument(app).expose(app)

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

    # Track prediction distribution
    prediction_counter.labels(prediction=str(int(prediction))).inc()

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }