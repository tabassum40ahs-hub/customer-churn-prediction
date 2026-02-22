import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------
# Load Saved Files
# ---------------------------
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("This model predicts whether a customer is likely to churn.")

st.markdown("---")

# ---------------------------
# User Inputs
# ---------------------------

tenure = st.number_input("Tenure (Months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges (USD)")
total_charges = st.number_input("Total Charges (USD)")

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])

# ---------------------------
# Prediction Button
# ---------------------------

if st.button("Predict Churn"):

    # Create input dictionary
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct order
    input_df = input_df[feature_columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Apply threshold 0.5
    if probability >= 0.5:
        st.error(f"⚠️ High Risk of Churn\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Churn\n\nProbability: {probability:.2f}")