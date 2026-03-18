from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Churn Prediction API")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: int
    gender: int = 0
    SeniorCitizen: int = 0
    Partner: int = 0
    Dependents: int = 0
    PhoneService: int = 1
    MultipleLines: int = 0
    InternetService: int = 1
    OnlineSecurity: int = 0
    OnlineBackup: int = 0
    DeviceProtection: int = 0
    TechSupport: int = 0
    StreamingTV: int = 0
    StreamingMovies: int = 0
    PaperlessBilling: int = 1
    PaymentMethod: int = 0

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(data: CustomerData):
    features = np.array([[
        data.gender, data.SeniorCitizen, data.Partner, data.Dependents,
        data.tenure, data.PhoneService, data.MultipleLines, data.InternetService,
        data.OnlineSecurity, data.OnlineBackup, data.DeviceProtection,
        data.TechSupport, data.StreamingTV, data.StreamingMovies,
        data.Contract, data.PaperlessBilling, data.PaymentMethod,
        data.MonthlyCharges, data.TotalCharges
    ]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return {
        "churn": bool(prediction),
        "churn_probability": round(float(probability), 3)
    }