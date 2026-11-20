import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from predict import predict_churn, THRESHOLD

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --app-bg: #0f1727;
        --sidebar-bg: #111827;
        --panel-bg: #1b2638;
        --panel-bg-hover: #1e293b;
        --border: #475569;
        --accent: #2563eb;
        --accent-dark: #1d4ed8;
        --text: #f8fafc;
    }
    .stApp { background: var(--app-bg); color: var(--text); }
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3, p, label, span { color: var(--text); }
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 100%; }
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--panel-bg-hover);
        color: var(--text);
        text-align: left;
        padding: 0.65rem 0.8rem;
        margin-bottom: 0.25rem;
    }
    div.stButton > button:hover {
        border-color: #60a5fa;
        background: var(--accent);
        color: white;
    }
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"]:hover {
        background: var(--accent-dark) !important;
        border-color: #60a5fa !important;
        color: white !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    input,
    textarea {
        background-color: var(--panel-bg) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        box-shadow: none !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[role="listbox"],
    li[role="option"] {
        background: var(--panel-bg) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background: var(--accent-dark) !important;
        color: white !important;
    }
    div[data-baseweb="tag"] {
        background: var(--accent-dark) !important;
        border: 1px solid #60a5fa !important;
        color: white !important;
    }
    div[data-baseweb="tag"] span,
    div[data-baseweb="tag"] svg {
        color: white !important;
        fill: white !important;
    }
    div[data-baseweb="radio"] > div,
    div[data-baseweb="radio"] div,
    [data-testid="stRadio"] div {
        color: var(--text) !important;
    }
    [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
        background: var(--panel-bg) !important;
        border-color: #94a3b8 !important;
    }
    [data-testid="stRadio"] [data-baseweb="radio"] [aria-checked="true"] > div:first-child {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: var(--accent) !important;
        border: 2px solid white !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div,
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] + div {
        background: var(--accent) !important;
    }
    [data-testid="stMetric"] {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--text) !important;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 12px;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        border: 1px solid #60a5fa;
        border-radius: 8px;
        padding: 12px 10px 14px;
        text-align: center;
        min-height: 92px;
    }
    .kpi-label { font-size: 16px; color: #eaf3ff; margin-bottom: 7px; }
    .kpi-value { font-size: 31px; font-weight: 700; color: white; }
    .panel {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 10px 4px;
    }
    [data-testid="stDataFrame"] {
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state.page = "Prediction"

with st.sidebar:
    st.markdown("## Customer Churn")
    if st.button(
        "Dashboard",
        use_container_width=True,
        type="primary" if st.session_state.page == "Dashboard" else "secondary",
    ):
        st.session_state.page = "Dashboard"
        st.rerun()
    if st.button(
        "Prediction",
        use_container_width=True,
        type="primary" if st.session_state.page == "Prediction" else "secondary",
    ):
        st.session_state.page = "Prediction"
        st.rerun()


def prediction_page() -> None:
    st.title("Customer Churn Predictor")
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
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
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
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        result = predict_churn(customer)
        st.metric("Churn probability", f"{result['probability']:.0%}")
        if result["risk"] == "High":
            st.error("🔴 High risk: consider a retention offer.")
        elif result["risk"] == "Medium":
            st.warning("🟠 Medium risk: keep an eye on this customer.")
        else:
            st.success("🟢 Low risk.")
        st.caption(f"Decision threshold used: {THRESHOLD:.2f}")


def style_chart(ax, title: str) -> None:
    ax.set_facecolor("#1b2638")
    ax.set_title(title, loc="left", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#7a879a")
    ax.grid(axis="x", color="#64748b", alpha=0.35)
    ax.grid(axis="y", color="#64748b", alpha=0.35)


def dashboard_page() -> None:
    st.markdown("## Customer Churn Dashboard")

    data_path = ROOT / "data" / "processed" / "predictions.csv"
    df = pd.read_csv(data_path)

    with st.sidebar:
        st.markdown("### Dashboard filters")
        contract_filter = st.multiselect(
            "Contract",
            sorted(df["Contract"].dropna().unique()),
            default=[],
        )
        risk_filter = st.multiselect(
            "Risk level",
            ["Low", "Medium", "High"],
            default=[],
        )
        internet_filter = st.multiselect(
            "Internet service",
            sorted(df["InternetService"].dropna().unique()),
            default=[],
        )
        tenure_filter = st.multiselect(
            "Tenure group",
            ["0-12", "13-24", "25-48", "49-72"],
            default=[],
        )

    filtered = df.copy()
    if contract_filter:
        filtered = filtered[filtered["Contract"].isin(contract_filter)]
    if risk_filter:
        filtered = filtered[filtered["risk_level"].isin(risk_filter)]
    if internet_filter:
        filtered = filtered[filtered["InternetService"].isin(internet_filter)]
    if tenure_filter:
        filtered = filtered[filtered["tenure_group"].isin(tenure_filter)]

    total_customers = len(filtered)
    high_risk = filtered[filtered["risk_level"] == "High"]
    revenue_at_risk = high_risk["MonthlyCharges"].sum()
    average_risk = filtered["risk_score"].mean() if total_customers else 0.0

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi-card"><div class="kpi-label">Total Customers</div><div class="kpi-value">{total_customers:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">Revenue at Risk</div><div class="kpi-value">${revenue_at_risk:,.0f}</div></div>
          <div class="kpi-card"><div class="kpi-label">High Risk Customers</div><div class="kpi-value">{len(high_risk):,}</div></div>
          <div class="kpi-card"><div class="kpi-label">Average Risk Score</div><div class="kpi-value">{average_risk:.2f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart1, chart2, chart3 = st.columns(3)

    with chart1:
        counts = high_risk["InternetService"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(4.2, 3.1), facecolor="#1b2638")
        ax.barh(counts.index, counts.values, color="#3b82f6")
        ax.set_xlim(left=0)
        style_chart(ax, "High-Risk Customers by Internet Service")
        st.pyplot(fig, clear_figure=True)

    with chart2:
        churn_rate = (
            filtered.assign(churned=filtered["actual_churn"].astype(str).str.lower().isin(["yes", "1"]))
            .groupby("tenure_group", observed=True)["churned"]
            .mean()
            .reindex(["0-12", "13-24", "25-48", "49-72"])
            .fillna(0)
        )
        fig, ax = plt.subplots(figsize=(4.2, 3.1), facecolor="#1b2638")
        ax.bar(churn_rate.index, churn_rate.values * 100, width=0.55, color="#3b82f6")
        ax.set_ylim(0, max(5, (churn_rate.max() * 100) * 1.15))
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
        style_chart(ax, "Churn rate by tenure group")
        st.pyplot(fig, clear_figure=True)

    with chart3:
        contract_risk = (
            filtered.groupby("Contract", observed=True)["risk_score"]
            .mean()
            .reindex(["Month-to-month", "One year", "Two year"])
            .dropna()
        )
        fig, ax = plt.subplots(figsize=(4.2, 3.1), facecolor="#1b2638")
        ax.barh(contract_risk.index, contract_risk.values, color="#3b82f6")
        ax.set_xlim(0, max(0.65, contract_risk.max() * 1.15))
        style_chart(ax, "Average risk score by contract")
        st.pyplot(fig, clear_figure=True)

    top20 = (
        filtered.sort_values("risk_score", ascending=False)
        .head(20)[["customerID", "Contract", "tenure", "MonthlyCharges", "risk_score"]]
        .rename(columns={
            "customerID": "Customer ID",
            "Contract": "Contract",
            "tenure": "Tenure",
            "MonthlyCharges": "Monthly charges",
            "risk_score": "Risk-Score",
        })
    )
    top20["Monthly charges"] = top20["Monthly charges"].map(lambda x: f"{x:.2f}")
    top20["Risk-Score"] = top20["Risk-Score"].map(lambda x: f"{x:.2f}")

    st.markdown("### Top 20 customers by risk score")
    st.dataframe(top20, use_container_width=True, hide_index=True)


if st.session_state.page == "Dashboard":
    dashboard_page()
else:
    prediction_page()
