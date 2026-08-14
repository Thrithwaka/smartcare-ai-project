"""
SmartCare AI - Disease Risk Assessment Prototype (Task 08)

A Streamlit clinical decision-support prototype for Option C: Disease Risk
Classification. Loads the model Task 06 selected as best-performing,
reproduces Task 03's exact feature engineering for a new patient, and
surfaces a simplified SHAP explanation alongside every prediction -
carrying the transparency and ethical framing established in Task 07
directly into the delivered tool, not just the report.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import plotly.graph_objects as go

from predict import (
    load_model_and_scaler, load_feature_columns, build_shap_explainer, predict,
    GENDER_OPTIONS, BLOOD_GROUP_OPTIONS, DEPARTMENT_OPTIONS, DIAGNOSIS_OPTIONS,
    ROOM_TYPE_OPTIONS, PAYMENT_METHOD_OPTIONS, PAYMENT_STATUS_OPTIONS,
    APPOINTMENT_STATUS_OPTIONS, RISK_ORDER,
)

st.set_page_config(
    page_title="SmartCare AI",
    page_icon="\u2295",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Design tokens
# ------------------------------------------------------------------
COLOR_BG = "#F5F5F7"
COLOR_SURFACE = "#FFFFFF"
COLOR_TEXT = "#1D1D1F"
COLOR_TEXT_SECONDARY = "#6E6E73"
COLOR_DIVIDER = "#D2D2D7"
COLOR_ACCENT = "#0071E3"
COLOR_LOW = "#30D158"
COLOR_MEDIUM = "#FF9F0A"
COLOR_HIGH = "#FF3B30"

RISK_COLORS = {"Low": COLOR_LOW, "Medium": COLOR_MEDIUM, "High": COLOR_HIGH}

# ------------------------------------------------------------------
# Global styling
# ------------------------------------------------------------------
# NOTE: cards are now real st.container(key=...) blocks (see form below),
# not manually opened/closed <div> tags. Each container gets a stable
# `st-key-<key>` class from Streamlit that we target here directly, so
# widgets placed inside `with st.container(key=...):` actually render
# nested inside the styled card instead of floating next to an empty box.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI',
                 Roboto, Helvetica, Arial, sans-serif;
}}

.stApp {{
    background-color: {COLOR_BG};
}}

#MainMenu, header, footer {{visibility: hidden;}}

.block-container {{
    max-width: 1080px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}}

.sc-hero {{
    text-align: center;
    margin-bottom: 2.5rem;
}}
.sc-hero h1 {{
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: {COLOR_TEXT};
    margin-bottom: 0.3rem;
}}
.sc-hero p {{
    font-size: 1.05rem;
    color: {COLOR_TEXT_SECONDARY};
    font-weight: 400;
    margin: 0;
}}
.sc-badge {{
    display: inline-block;
    margin-top: 0.9rem;
    padding: 0.35rem 0.9rem;
    background: rgba(0,113,227,0.08);
    color: {COLOR_ACCENT};
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}}

/* Card styling targets the container itself (NOT its children -
   targeting children made every widget/markdown call inside get its
   own separate white box instead of one merged card). */
div[class*="st-key-demo_card"],
div[class*="st-key-vitals_card"],
div[class*="st-key-history_card"],
div[class*="st-key-results_card"] {{
    background: {COLOR_SURFACE} !important;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    box-shadow: 0 2px 24px rgba(0,0,0,0.06);
    margin-bottom: 1.4rem;
    border: 1px solid rgba(0,0,0,0.02);
}}

/* The app's base theme is dark, so default widget label text is light
   and disappears against the white cards above. Force readable text
   color on every widget label (Age, Gender, Blood group, etc). */
[data-testid="stWidgetLabel"] p {{
    color: {COLOR_TEXT} !important;
}}

.sc-section-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: {COLOR_TEXT};
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem;
}}
.sc-section-sub {{
    font-size: 0.82rem;
    color: {COLOR_TEXT_SECONDARY};
    margin-bottom: 1.1rem;
}}

div[data-testid="stButton"] > button {{
    background: {COLOR_ACCENT};
    color: white;
    border: none;
    border-radius: 999px;
    padding: 0.65rem 1.6rem;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
    transition: opacity 0.15s ease;
}}
div[data-testid="stButton"] > button:hover {{
    opacity: 0.85;
    color: white;
}}

.sc-result-placeholder {{
    text-align: center;
    padding: 3.5rem 1.5rem;
    color: {COLOR_TEXT_SECONDARY};
}}
.sc-result-placeholder .sc-icon {{
    font-size: 2.2rem;
    margin-bottom: 0.6rem;
    opacity: 0.5;
}}

.sc-risk-label {{
    text-align: center;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.6rem 0 0.1rem 0;
}}
.sc-risk-caption {{
    text-align: center;
    font-size: 0.85rem;
    color: {COLOR_TEXT_SECONDARY};
    margin-bottom: 1.4rem;
}}

.sc-factor-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid {COLOR_DIVIDER};
    font-size: 0.88rem;
}}
.sc-factor-row:last-child {{ border-bottom: none; }}
.sc-factor-name {{
    color: {COLOR_TEXT};
    font-weight: 500;
}}
.sc-factor-dir {{
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
}}

.sc-disclaimer {{
    text-align: center;
    font-size: 0.78rem;
    color: {COLOR_TEXT_SECONDARY};
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid {COLOR_DIVIDER};
    line-height: 1.5;
}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    model, scaler, selection, metadata = load_model_and_scaler()
    feature_columns = load_feature_columns()
    explainer = build_shap_explainer(model)
    return model, scaler, selection, metadata, feature_columns, explainer


model, scaler, selection, metadata, feature_columns, explainer = get_pipeline()

# ------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------
st.markdown(f"""
<div class="sc-hero">
    <h1>SmartCare AI</h1>
    <p>Disease risk assessment for clinical decision support</p>
    <div class="sc-badge">Model: {selection['best_model']} &middot; Macro F1 {selection['macro_f1']:.3f}</div>
</div>
""", unsafe_allow_html=True)

col_form, col_result = st.columns([1.15, 1], gap="large")

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with col_form:
    with st.container(key="demo_card"):
        st.markdown('<div class="sc-section-title">Patient Demographics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sc-section-sub">Basic identifying and biological information</div>', unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        with d1:
            age = st.slider("Age", 1, 100, 45)
            gender = st.selectbox("Gender", GENDER_OPTIONS)
        with d2:
            blood_group = st.selectbox("Blood group", BLOOD_GROUP_OPTIONS)

    with st.container(key="vitals_card"):
        st.markdown('<div class="sc-section-title">Vitals and Lab Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="sc-section-sub">Clinical measurements from the current visit</div>', unsafe_allow_html=True)

        v1, v2 = st.columns(2)
        with v1:
            systolic_bp = st.slider("Systolic BP (mmHg)", 80, 200, 120)
            blood_sugar_mg_dl = st.slider("Blood sugar (mg/dL)", 60, 400, 100)
            bmi = st.slider("BMI", 12.0, 50.0, 24.0, step=0.1)
        with v2:
            diastolic_bp = st.slider("Diastolic BP (mmHg)", 40, 130, 80)
            cholesterol_mg_dl = st.slider("Cholesterol (mg/dL)", 100, 400, 190)

    with st.container(key="history_card"):
        st.markdown('<div class="sc-section-title">Visit and History</div>', unsafe_allow_html=True)
        st.markdown('<div class="sc-section-sub">Department, admission status, and appointment history</div>', unsafe_allow_html=True)

        h1, h2 = st.columns(2)
        with h1:
            department = st.selectbox("Department", DEPARTMENT_OPTIONS)
            diagnosis = st.selectbox("Diagnosis", DIAGNOSIS_OPTIONS)
            appointment_status = st.selectbox("Appointment status", APPOINTMENT_STATUS_OPTIONS)
            previous_appointments = st.number_input("Previous appointments", 0, 50, 3)
        with h2:
            admitted = st.toggle("Currently admitted", value=False)
            room_type = st.selectbox("Room type", ROOM_TYPE_OPTIONS, disabled=not admitted)
            length_of_stay_days = st.number_input("Length of stay (days)", 0, 60, 0, disabled=not admitted)
            previous_admissions = st.number_input("Previous admissions", 0, 20, 0)

        if previous_appointments > 0:
            missed_previous_appointments = st.slider(
                "Missed previous appointments", 0, int(previous_appointments), 0
            )
        else:
            st.markdown('<div class="sc-section-sub" style="margin-bottom:0;">Missed previous appointments: 0 (no previous appointments recorded)</div>', unsafe_allow_html=True)
            missed_previous_appointments = 0

        p1, p2 = st.columns(2)
        with p1:
            payment_method = st.selectbox("Payment method", PAYMENT_METHOD_OPTIONS)
        with p2:
            payment_status = st.selectbox("Payment status", PAYMENT_STATUS_OPTIONS)

    predict_clicked = st.button("Predict risk level", use_container_width=True)

# ------------------------------------------------------------------
# Results panel
# ------------------------------------------------------------------
with col_result:
    with st.container(key="results_card"):
        if not predict_clicked and "last_result" not in st.session_state:
            st.markdown("""
            <div class="sc-result-placeholder">
                <div class="sc-icon">&#8853;</div>
                <div>Fill in patient details and select<br><b>Predict risk level</b> to see the assessment.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if predict_clicked:
                patient = {
                    "age": age, "gender": gender, "blood_group": blood_group,
                    "department": department, "diagnosis": diagnosis,
                    "admitted": admitted, "room_type": room_type,
                    "previous_appointments": previous_appointments,
                    "missed_previous_appointments": missed_previous_appointments,
                    "previous_admissions": previous_admissions,
                    "length_of_stay_days": length_of_stay_days,
                    "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
                    "blood_sugar_mg_dl": blood_sugar_mg_dl,
                    "cholesterol_mg_dl": cholesterol_mg_dl, "bmi": bmi,
                    "payment_method": payment_method, "payment_status": payment_status,
                    "appointment_status": appointment_status,
                }
                st.session_state["last_result"] = predict(
                    patient, model, scaler, feature_columns, explainer
                )

            result = st.session_state["last_result"]
            predicted = result["predicted_class"]
            color = RISK_COLORS[predicted]

            # Apple Health-style ring chart showing class probabilities
            fig = go.Figure(data=[go.Pie(
                labels=list(result["probabilities"].keys()),
                values=list(result["probabilities"].values()),
                hole=0.72,
                marker=dict(colors=[RISK_COLORS[k] for k in result["probabilities"].keys()]),
                textinfo="none",
                sort=False,
                direction="clockwise",
            )])
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                annotations=[dict(
                    text=f"<b>{result['confidence']:.0%}</b>", x=0.5, y=0.5,
                    font=dict(size=28, color=color, family="Inter"), showarrow=False
                )],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown(f"""
            <div class="sc-risk-label" style="color:{color};">{predicted} Risk</div>
            <div class="sc-risk-caption">Predicted with {result['confidence']:.1%} confidence</div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sc-section-title" style="margin-top:0.5rem;">Top contributing factors</div>', unsafe_allow_html=True)
            for factor in result["top_factors"]:
                dir_color = COLOR_LOW if factor["direction"] == "supports" else COLOR_HIGH
                dir_bg = "rgba(48,209,88,0.12)" if factor["direction"] == "supports" else "rgba(255,59,48,0.10)"
                st.markdown(f"""
                <div class="sc-factor-row">
                    <span class="sc-factor-name">{factor['feature'].replace('_', ' ').title()}</span>
                    <span class="sc-factor-dir" style="color:{dir_color}; background:{dir_bg};">
                        {factor['direction']}
                    </span>
                </div>
                """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Disclaimer
# ------------------------------------------------------------------
st.markdown("""
<div class="sc-disclaimer">
    SmartCare AI is a clinical decision-support prototype built for the CCS3440
    coursework. Predictions are generated from a synthetic training dataset and
    are intended to support, not replace, professional clinical judgment.
    This tool should not be used for actual patient diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)