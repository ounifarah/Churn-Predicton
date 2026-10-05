import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import clean_data, add_features

ROOT = Path(__file__).resolve().parents[1]

model = joblib.load(ROOT / "models" / "churn_model.joblib")
threshold = json.loads((ROOT / "models" / "config.json").read_text())["threshold"]

raw = pd.read_csv(ROOT / "data" / "raw" / "telco_churn.csv")
ids = raw["customerID"]
df = add_features(clean_data(raw))

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Same split as the notebook, so we can flag which customers the model never saw
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

score = model.predict_proba(X)[:, 1]

out = pd.DataFrame({
    "customerID": ids,
    "Contract": df["Contract"],
    "InternetService": df["InternetService"],
    "PaymentMethod": df["PaymentMethod"],
    "TechSupport": df["TechSupport"],
    "tenure": df["tenure"],
    "tenure_group": df["tenure_group"],
    "MonthlyCharges": df["MonthlyCharges"],
    "actual_churn": y,
    "risk_score": score.round(3),
    "predicted_churn": (score >= threshold).astype(int),
    "risk_level": pd.cut(score, bins=[-1, threshold / 2, threshold, 2],
                         labels=["Low", "Medium", "High"]).astype(str),
    "split": ["test" if i in X_test.index else "train" for i in X.index],
})

(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
out.to_csv(ROOT / "data" / "processed" / "predictions.csv", index=False)
print(out.shape)
print(out["risk_level"].value_counts())