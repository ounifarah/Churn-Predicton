import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from predict import predict_churn, THRESHOLD

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details to estimate their risk of leaving.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total charges", 0.0, 10000.0, float(monthly * max(tenure, 1)))

with col2:
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    no_net = "No internet service"
    opts = ["No", "Yes"] if internet != "No" else [no_net]
    security = st.selectbox("Online security", opts)
    backup = st.selectbox("Online backup", opts)
    protection = st.selectbox("Device protection", opts)
    support = st.selectbox("Tech support", opts)
    tv = st.selectbox("Streaming TV", opts)
    movies = st.selectbox("Streaming movies", opts)

if st.button("Predict", type="primary"):
    customer = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    result = predict_churn(customer)
    st.metric("Churn probability", f"{result['probability']:.0%}")
    if result["risk"] == "High":
        st.error("🔴 High risk: consider a retention offer (e.g. a discount for a 1-year contract).")
    elif result["risk"] == "Medium":
        st.warning("🟠 Medium risk: keep an eye on this customer.")
    else:
        st.success("🟢 Low risk.")
    st.caption(f"Decision threshold used: {THRESHOLD:.2f}")