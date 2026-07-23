import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_data
from models.no_show import train_classifier
from models.forecasting import train_forecast

# -------------------------------
# Load dataset directly
# -------------------------------
DATA_PATH = "data/Medical_appointment_data.csv"

st.title("CER Rehabilitation Analytics Dashboard")

df = pd.read_csv(DATA_PATH)
df = preprocess_data(df)

# -------------------------------
# Classification
# -------------------------------
st.subheader("No-Show Risk Prediction")
clf, f1, roc = train_classifier(df)
st.write(f"Model Performance → F1: {f1:.2f}, ROC-AUC: {roc:.2f}")

age = st.number_input("Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", ["F","M"])
days_waiting = st.number_input("Days Waiting", min_value=0)
scholarship = st.selectbox("Scholarship", [0,1])
diabetes = st.selectbox("Diabetes", [0,1])
hypertension = st.selectbox("Hypertension", [0,1])
alcoholism = st.selectbox("Alcoholism", [0,1])
handicap = st.selectbox("Handcap", [0,1])

input_df = pd.DataFrame([[age, 0 if gender=="F" else 1, days_waiting, scholarship,
                          hypertension, diabetes, alcoholism, handicap,
                          hypertension+diabetes+alcoholism+handicap]],
                        columns=["age","gender","DaysWaiting","Scholarship",
                                 "Hipertension","Diabetes","Alcoholism","Handcap",
                                 "chronic_conditions"])

if st.button("Predict Risk"):
    risk_score = clf.predict_proba(input_df)[0][1]
    st.write(f"Predicted No-Show Risk: {risk_score:.2f}")

# -------------------------------
# Forecasting
# -------------------------------
st.subheader("Demand Forecasting")
model, forecast, mape, r2 = train_forecast(df)
st.write(f"Forecast Performance → MAPE: {mape:.2f}, R²: {r2:.2f}")
st.line_chart(forecast[["ds","yhat"]].set_index("ds"))
