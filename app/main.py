"""Churn prediction API. Accepts raw customer attributes, as they appear in the dataset."""
import json
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from churn.pipeline import FEATURES, METADATA_PATH, MODEL_PATH

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text())
THRESHOLD = metadata["threshold"]

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]

EXAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
}


class Customer(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE]})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(gt=0, le=500)


class Prediction(BaseModel):
    churn_probability: float
    high_risk: bool
    threshold: float


app = FastAPI(title="Churn Prediction API", version="2.0")


@app.get("/")
def root() -> dict:
    return {"message": "Churn Prediction API. Open /docs to try it."}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": metadata["model"], "test_roc_auc": metadata["metrics"]["test_roc_auc"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    row = pd.DataFrame([customer.model_dump()])[FEATURES]
    proba = float(model.predict_proba(row)[0, 1])
    return Prediction(churn_probability=round(proba, 4), high_risk=proba >= THRESHOLD, threshold=THRESHOLD)
