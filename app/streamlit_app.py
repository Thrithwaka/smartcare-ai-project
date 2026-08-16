"""
Task 08 – SmartCare AI Streamlit Prototype (Redesigned UI)
Inference-only application over the existing Task 03–07 pipeline.
UI/UX only — prediction logic unchanged from the original implementation.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from utils.preprocessing import prepare_input, PreprocessingError
from utils.prediction import load_artifacts, predict, ArtifactLoadError, PredictionError
from utils.explainability import build_explainer, explain_instance, ExplainabilityError, friendly_label

st.set_page_config(page_title="SmartCare AI", page_icon="🏥", layout="centered")


def load_css():
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


load_css()


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_resource
def get_explainer(_model, _background):
    return build_explainer(_model, _background)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="sc-header">
        <div>
            <div class="sc-title">SmartCare AI</div>
            <div class="sc-subtitle">AI-Powered Disease Risk Assessment</div>
        </div>
    </div>
    <div class="sc-status"><span class="sc-status-dot"></span>AI Model Ready</div>
    <div class="sc-hero">Understand how our AI model evaluates patient information and classifies disease risk into Low, Medium, or High categories.</div>
    <div class="sc-hero-tags">66 Features &nbsp;·&nbsp; Logistic Regression &nbsp;·&nbsp; Explainable AI</div>
    """,
    unsafe_allow_html=True,
)

try:
    model, scaler, feature_columns, scaled_columns = get_artifacts()
    artifacts_ok = True
except ArtifactLoadError as e:
    st.error(f"Could not load required model artifacts.\n\n{e}")
    artifacts_ok = False

if not artifacts_ok:
    st.stop()

# ---------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------
st.markdown('<div class="sc-section-heading">How the Prediction Works</div>', unsafe_allow_html=True)
steps = [("01", "Patient Info"), ("02", "Validation"), ("03", "Feature Processing"),
         ("04", "66 Features"), ("05", "AI Model"), ("06", "Risk Prediction")]
cols = st.columns(len(steps))
for col, (num, label) in zip(cols, steps):
    with col:
        st.markdown(
            f'<div class="sc-step-card"><div class="sc-step-num">{num}</div>'
            f'<div class="sc-step-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# Patient Assessment Form
# ---------------------------------------------------------------------
st.markdown('<div class="sc-section-heading">Patient Assessment</div>', unsafe_allow_html=True)

st.markdown('<div class="sc-group-label first">Patient Profile</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
age = c1.number_input("Age", min_value=0, max_value=120, value=45)
gender = c2.selectbox("Gender", ["Female", "Male"])
blood_group = c3.selectbox("Blood Group", ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"])

st.markdown('<div class="sc-group-label">Clinical Measurements</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
systolic_bp = c1.number_input("Systolic BP", min_value=70, max_value=220, value=128)
diastolic_bp = c2.number_input("Diastolic BP", min_value=40, max_value=140, value=80)
bmi = c3.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
c1, c2 = st.columns(2)
blood_sugar_mg_dl = c1.number_input("Blood Sugar (mg/dL)", min_value=40, max_value=400, value=100)
cholesterol_mg_dl = c2.number_input("Cholesterol (mg/dL)", min_value=80, max_value=400, value=200)

st.markdown('<div class="sc-group-label">Medical & Admission Information</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
admitted_label = c1.selectbox("Admitted", ["No", "Yes"])
admitted = 1 if admitted_label == "Yes" else 0
room_type = c2.selectbox(
    "Room Type", ["Not Admitted", "General Ward", "ICU", "Private Room"],
    index=0 if admitted == 0 else 1, disabled=(admitted == 0),
)
if admitted == 0:
    room_type = "Not Admitted"
length_of_stay_days = c3.number_input(
    "Length of Stay (days)", min_value=0, max_value=90,
    value=0 if admitted == 0 else 3, disabled=(admitted == 0),
)
if admitted == 0:
    length_of_stay_days = 0

c1, c2, c3 = st.columns(3)
previous_admissions = c1.number_input("Previous Admissions", min_value=0, max_value=20, value=0)
previous_appointments = c2.number_input("Previous Appointments", min_value=0, max_value=50, value=2)
diagnosis = c3.selectbox(
    "Diagnosis",
    ["Asthma", "Back Pain", "Chest Pain", "Diabetes", "Fever", "Fracture",
     "Hypertension", "Kidney Infection", "Migraine", "Pneumonia"],
)

st.markdown('<div class="sc-group-label">Healthcare & Administrative Information</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
department = c1.selectbox(
    "Department",
    ["Cardiology", "General Medicine", "Laboratory Services", "Neurology",
     "Orthopedics", "Pediatrics", "Radiology"],
)
appointment_status = c2.selectbox("Appointment Status", ["Cancelled", "Completed", "No-Show", "Scheduled"])
payment_method = c3.selectbox("Payment Method", ["Card", "Cash", "Insurance", "Online"])
payment_status = st.selectbox("Payment Status", ["Paid", "Partially Paid", "Unpaid"])

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("Predict Disease Risk", type="primary")

# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
if predict_clicked:
    raw_input = {
        "age": age, "gender": gender, "blood_group": blood_group,
        "admitted": admitted, "length_of_stay_days": length_of_stay_days,
        "room_type": room_type, "previous_admissions": previous_admissions,
        "previous_appointments": previous_appointments, "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp, "blood_sugar_mg_dl": blood_sugar_mg_dl,
        "cholesterol_mg_dl": cholesterol_mg_dl, "bmi": bmi, "department": department,
        "diagnosis": diagnosis, "appointment_status": appointment_status,
        "payment_method": payment_method, "payment_status": payment_status,
    }

    with st.spinner("Analyzing patient information..."):
        try:
            final_features = prepare_input(raw_input, feature_columns, scaler, scaled_columns)
        except PreprocessingError as e:
            st.error(f"Input processing failed: {e}")
            st.stop()

        try:
            result = predict(model, final_features)
        except PredictionError as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    predicted = result["predicted_class"]
    risk_class_css = {"Low": "sc-risk-low", "Medium": "sc-risk-medium", "High": "sc-risk-high"}[predicted]
    risk_desc = {
        "Low": "Based on the information provided, the model classified this patient into the Low Risk category.",
        "Medium": "Based on the information provided, the model classified this patient into the Medium Risk category.",
        "High": "Based on the information provided, the model classified this patient into the High Risk category.",
    }[predicted]

    st.markdown('<div class="sc-section-heading">AI Assessment Result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sc-result-wrap {risk_class_css}">
            <div class="sc-result-eyebrow">Predicted Risk Level</div>
            <div class="sc-result-value">{predicted.upper()} RISK</div>
            <div class="sc-result-desc">{risk_desc}</div>
            <div class="sc-prob-title">Model prediction probability</div>
        """,
        unsafe_allow_html=True,
    )

    bar_colors = {"Low": "#1E8E3E", "Medium": "#F9AB00", "High": "#D93025"}
    for cls in ["Low", "Medium", "High"]:
        pct = result["probabilities"][cls] * 100
        st.markdown(
            f"""
            <div class="sc-prob-row">
                <div class="sc-prob-label">{cls}</div>
                <div class="sc-prob-bar-bg"><div class="sc-prob-bar-fill" style="width:{pct}%; background-color:{bar_colors[cls]};"></div></div>
                <div class="sc-prob-pct">{pct:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="sc-prob-note">These values represent the model\'s estimated probabilities for each risk class. They are not medical certainty.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # Factors Considered
    # -------------------------------------------------------------
    st.markdown('<div class="sc-section-heading">Factors Considered by the Model</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sc-factor-grid">
            <div class="sc-factor-card"><div class="sc-factor-card-title">Patient Profile</div><div class="sc-factor-card-desc">Age, gender, blood group</div></div>
            <div class="sc-factor-card"><div class="sc-factor-card-title">Clinical Measurements</div><div class="sc-factor-card-desc">Blood pressure, blood sugar, cholesterol, BMI</div></div>
            <div class="sc-factor-card"><div class="sc-factor-card-title">Medical Information</div><div class="sc-factor-card-desc">Admissions, appointments, diagnosis</div></div>
            <div class="sc-factor-card"><div class="sc-factor-card-title">Healthcare Information</div><div class="sc-factor-card-desc">Department, appointment status, payment information</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # Explainability
    # -------------------------------------------------------------
    st.markdown('<div class="sc-section-heading">Why Did the Model Make This Prediction?</div>', unsafe_allow_html=True)
    try:
        root = Path(__file__).resolve().parents[1]
        X_train_sample = pd.read_csv(root / "data" / "processed" / "X_train.csv").sample(n=100, random_state=42)
        explainer = get_explainer(model, X_train_sample)
        contributions = explain_instance(explainer, final_features, result["predicted_class_index"], feature_columns)

        for c in contributions:
            arrow = "↑" if c["direction"] == "up" else "↓"
            css_class = "sc-contrib-up" if c["direction"] == "up" else "sc-contrib-down"
            st.markdown(
                f'<div class="sc-contrib-row"><span class="sc-contrib-name">{friendly_label(c["feature"])}</span>'
                f'<span class="sc-contrib-value {css_class}">{arrow} {c["shap_value"]:+.2f}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="sc-explain-note">Positive contributions pushed the model toward the predicted class, '
            'while negative contributions pushed it away. These values describe model behaviour and do not '
            'establish medical causation.</div>',
            unsafe_allow_html=True,
        )
    except (ExplainabilityError, FileNotFoundError, Exception):
        st.info("The optional explanation component is currently unavailable for this prediction.")

# ---------------------------------------------------------------------
# About the AI Model
# ---------------------------------------------------------------------
with st.expander("About the AI Model"):
    st.markdown(
        """
        <div class="sc-metric-grid">
            <div class="sc-metric-card"><div class="sc-metric-value">92.00%</div><div class="sc-metric-label">Accuracy</div></div>
            <div class="sc-metric-card"><div class="sc-metric-value">91.40%</div><div class="sc-metric-label">Macro F1</div></div>
            <div class="sc-metric-card"><div class="sc-metric-value">94.06%</div><div class="sc-metric-label">Macro Precision</div></div>
            <div class="sc-metric-card"><div class="sc-metric-value">89.44%</div><div class="sc-metric-label">Macro Recall</div></div>
            <div class="sc-metric-card"><div class="sc-metric-value">99.14%</div><div class="sc-metric-label">Macro ROC-AUC</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Model: Logistic Regression · 66 features · Task 06 held-out test set evaluation. These are not the confidence of the current prediction.")

with st.expander("Technical Details"):
    st.markdown(
        """
        - **Target:** disease_risk_level
        - **Classes:** Low / Medium / High
        - **Feature count:** 66
        - **Model:** Logistic Regression
        - **Preprocessing:** Task 03 leakage-safe pipeline
        - **Explainability:** SHAP (LinearExplainer)
        - **Dataset:** Synthetic SmartCare AI dataset
        """
    )

st.markdown(
    """
    <div class="sc-disclaimer">
        <div class="sc-disclaimer-title">Educational AI prototype</div>
        SmartCare AI is not a medical diagnostic system and should not be used as a substitute for
        professional medical advice. The dataset is synthetic, predictions may be inaccurate, and all
        outputs require human interpretation.
    </div>
    """,
    unsafe_allow_html=True,
)