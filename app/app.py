"""
app.py
-------
Task 08 — AI Prototype Development
Owner: Tharindu

A Streamlit app that:
  1. Accepts patient information as user input.
  2. Applies the same preprocessing steps used in training.
  3. Loads the best model (models/best_model.pkl) and predicts disease_risk_level.
  4. Displays the prediction plus a short SHAP-based explanation.

Run with:  streamlit run app/app.py

NOTE: Adjust the input fields below to match the ACTUAL columns in your
processed dataset (check data/raw/smartcare_ai_dataset_data_dictionary.csv).
This is a working template, not the final field list.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from src import config
from src.utils import load_model
from src.explainability import build_explainer, get_top_features_for_patient

st.set_page_config(page_title="SmartCare Disease Risk Predictor", page_icon="🏥", layout="centered")

st.title("🏥 SmartCare Hospital — Disease Risk Prediction")
st.markdown(
    "Enter patient information below to predict disease risk level "
    "(**Low / Medium / High**), with an explanation of the key contributing factors."
)


@st.cache_resource
def get_model():
    return load_model(config.BEST_MODEL_FILE)


@st.cache_data
def get_reference_columns():
    """Load processed data just to get the exact column order the model expects."""
    df = pd.read_csv(config.TRAIN_FILE)
    return df.drop(columns=[config.TARGET_VARIABLE]).columns.tolist()


model = get_model()
feature_columns = get_reference_columns()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("patient_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_pressure = st.number_input("Blood Pressure (systolic)", min_value=60, max_value=250, value=120)
        blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=500, value=100)

    with col2:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
        previous_admissions = st.number_input("Previous Admissions", min_value=0, max_value=20, value=0)
        treatment_count = st.number_input("Treatment Count", min_value=0, max_value=50, value=1)

    submitted = st.form_submit_button("Predict Disease Risk")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    # Build a single-row dataframe matching the training feature set.
    # NOTE: This must apply the SAME encoding/scaling used in preprocessing.py —
    # for a real submission, save the fitted encoders/scaler from Task 03
    # (e.g. with joblib) and load them here instead of re-deriving values.
    input_dict = {col: 0 for col in feature_columns}  # default-fill any engineered columns

    input_dict.update({
        "age": age,
        "gender": 1 if gender == "Male" else 0,
        "blood_pressure": blood_pressure,
        "blood_sugar": blood_sugar,
        "cholesterol": cholesterol,
        "bmi": bmi,
        "previous_admissions": previous_admissions,
        "treatment_count": treatment_count,
    })

    input_df = pd.DataFrame([input_dict])[feature_columns]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")
    risk_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(prediction, "blue")
    st.markdown(f"### Predicted Risk Level: :{risk_color}[{prediction}]")

    prob_df = pd.DataFrame({"Class": model.classes_, "Probability": probabilities})
    st.bar_chart(prob_df.set_index("Class"))

    # --- Explainability ---
    st.subheader("Why this prediction? (Top contributing factors)")
    try:
        explainer = build_explainer(model, input_df)
        top_features = get_top_features_for_patient(model, input_df, explainer, top_n=5)
        st.bar_chart(top_features)
        st.caption("Larger bars = greater influence on this specific prediction (SHAP values).")
    except Exception as e:
        st.warning(f"Explainability output unavailable: {e}")

st.markdown("---")
st.caption("SmartCare Hospital AI Coursework — CCS3440 — Prototype for demonstration purposes only. "
           "Not for real clinical use.")
