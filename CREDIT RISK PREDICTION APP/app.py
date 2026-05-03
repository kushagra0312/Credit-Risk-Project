import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

st.set_page_config(page_title="Credit Risk System", layout="wide")

st.title("🏦 Credit Risk Prediction System")
st.write("End-to-end ML system with business decision + explainability")

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "best_credit_risk_model.pkl"))
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler

model, scaler = load_model()

# -------------------------------
# INPUT UI
# -------------------------------
st.sidebar.header("Applicant Details")

age = st.sidebar.number_input("Age", 18, 80, 30)
income = st.sidebar.number_input("Income", 10000, 5000000, 500000)
loan = st.sidebar.number_input("Loan Amount", 10000, 5000000, 200000)
credit = st.sidebar.slider("Credit Score", 300, 850, 650)
months = st.sidebar.number_input("Months Employed", 0, 480, 60)
credit_lines = st.sidebar.number_input("Credit Lines", 0, 20, 3)
rate = st.sidebar.slider("Interest Rate", 1.0, 35.0, 12.0)
term = st.sidebar.slider("Loan Term", 6, 360, 60)
dti = st.sidebar.slider("DTI Ratio", 0.0, 1.0, 0.3)

education = st.sidebar.selectbox("Education", ["Bachelor's", "High School", "Master's", "PhD"])
employment = st.sidebar.selectbox("Employment", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital = st.sidebar.selectbox("Marital Status", ["Divorced", "Married", "Single"])
mortgage = st.sidebar.selectbox("Has Mortgage", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
purpose = st.sidebar.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
cosigner = st.sidebar.selectbox("Co-Signer", ["No", "Yes"])

# -------------------------------
# BUILD INPUT
# -------------------------------
def build_input():
    data = {col: 0 for col in [
        "Age","Income","LoanAmount","CreditScore","MonthsEmployed",
        "NumCreditLines","InterestRate","LoanTerm","DTIRatio",
        "Education_High School","Education_Master's","Education_PhD",
        "EmploymentType_Part-time","EmploymentType_Self-employed","EmploymentType_Unemployed",
        "MaritalStatus_Married","MaritalStatus_Single",
        "HasMortgage_Yes","HasDependents_Yes",
        "LoanPurpose_Business","LoanPurpose_Education","LoanPurpose_Home","LoanPurpose_Other",
        "HasCoSigner_Yes"
    ]}

    data.update({
        "Age": age, "Income": income, "LoanAmount": loan,
        "CreditScore": credit, "MonthsEmployed": months,
        "NumCreditLines": credit_lines, "InterestRate": rate,
        "LoanTerm": term, "DTIRatio": dti
    })

    if education != "Bachelor's":
        data[f"Education_{education}"] = 1
    if employment != "Full-time":
        data[f"EmploymentType_{employment}"] = 1
    if marital != "Divorced":
        data[f"MaritalStatus_{marital}"] = 1
    if mortgage == "Yes":
        data["HasMortgage_Yes"] = 1
    if dependents == "Yes":
        data["HasDependents_Yes"] = 1
    if purpose != "Auto":
        data[f"LoanPurpose_{purpose}"] = 1
    if cosigner == "Yes":
        data["HasCoSigner_Yes"] = 1

    return pd.DataFrame([data])

# -------------------------------
# PREDICTION
# -------------------------------
if st.sidebar.button("Predict Risk"):

    X_df = build_input()
    X = scaler.transform(X_df) if scaler else X_df

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.error("❌ High Risk of Default")
    else:
        st.success("✅ Low Risk")

    st.write(f"Probability: **{prob:.2%}**")

    st.subheader("💼 Decision")
    if prob < 0.25:
        st.success("Approve Loan")
    elif prob < 0.5:
        st.warning("Approve with Higher Interest")
    else:
        st.error("Reject Loan")

    st.subheader("💰 Business Impact")
    if prob > 0.5:
        st.error(f"Loss Avoided: ₹{loan:,.0f}")
    elif prob > 0.25:
        st.warning(f"Extra Revenue: ₹{loan*0.02:,.0f}")
    else:
        st.success("Stable Customer")

    st.subheader("🔍 Key Drivers")
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_df)
        vals = shap_values.values[0]
        feats = X_df.columns
        top = sorted(zip(feats, vals), key=lambda x: abs(x[1]), reverse=True)[:5]
        for f, v in top:
            if v > 0:
                st.write(f"🔴 {f} ↑ Risk")
            else:
                st.write(f"🟢 {f} ↓ Risk")
    except:
        st.info("Explainability not available")

# -------------------------------
# BATCH
# -------------------------------
st.subheader("📁 Batch Prediction")
file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    X = scaler.transform(df) if scaler else df
    df["Prediction"] = model.predict(X)
    df["Probability"] = model.predict_proba(X)[:,1]
    st.dataframe(df)
    st.download_button("Download", df.to_csv(index=False), "results.csv")
