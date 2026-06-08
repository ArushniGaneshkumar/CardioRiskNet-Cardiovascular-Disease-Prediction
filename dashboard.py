# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler

# Set Streamlit page config
st.set_page_config(page_title="CardioRiskNet", layout="centered")

# Load model
model = joblib.load("checkpoints/cardiorisknet_model.pkl")

# Load training data for SHAP scaler
data = pd.read_csv("data/final_input.csv")  # Ensure this matches the input used for training
labels = pd.read_csv("data/labels.csv")

# Sidebar input
st.sidebar.header("🩺 Enter Patient Info")

age = st.sidebar.slider("Age", 29, 77, 50)
sex = st.sidebar.selectbox("Sex", [0, 1])  # 0 = female, 1 = male
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting BP", 94, 200, 120)
chol = st.sidebar.slider("Cholesterol", 126, 564, 240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate", 71, 202, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
slope = st.sidebar.selectbox("Slope of ST", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Build input DataFrame
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

# Scale the input
scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)
scaled_input = scaler.transform(input_data)

# Predict
prediction = model.predict(scaled_input)[0]
risk_score = round(float(prediction), 2)

# Output
st.title("🫀 CardioRiskNet: Cardiovascular Risk Scoring System")
st.subheader("Predicted Risk Score (0 – 1):")
st.metric(label="Risk Score", value=risk_score)

if risk_score >= 0.5:
    st.error("⚠️ High Risk of Cardiovascular Disease")
else:
    st.success("✅ Low Risk of Cardiovascular Disease")

# SHAP Explainability
st.subheader("📊 Feature Contribution (SHAP)")

# SHAP setup
explainer = shap.Explainer(model.predict, scaled_data)
shap_values = explainer(scaled_input)

# Display SHAP Waterfall plot
fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(shap_values[0], max_display=13, show=False)
st.pyplot(fig)
