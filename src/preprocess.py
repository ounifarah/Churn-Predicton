import pandas as pd

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: fix TotalCharges, drop the ID, encode the target if present."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"], errors="ignore")
    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features used by the model."""
    df = df.copy()
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    return df