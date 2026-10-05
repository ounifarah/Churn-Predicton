import json
from pathlib import Path

import joblib
import pandas as pd

from preprocess import clean_data, add_features

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "models" / "churn_model.joblib")
CONFIG = json.loads((ROOT / "models" / "config.json").read_text())
THRESHOLD = CONFIG["threshold"]


def predict_churn(customer: dict) -> dict:
    """Takes a dict of raw customer details, returns probability and risk label."""
    df = pd.DataFrame([customer])
    df = add_features(clean_data(df))
    proba = float(MODEL.predict_proba(df)[0, 1])
    return {
        "probability": proba,
        "will_churn": proba >= THRESHOLD,
        "risk": "High" if proba >= THRESHOLD else ("Medium" if proba >= THRESHOLD / 2 else "Low"),
    }


if __name__ == "__main__":
    sample = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0,
    }
    print(predict_churn(sample))