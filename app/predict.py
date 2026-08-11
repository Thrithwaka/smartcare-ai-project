"""
Prediction pipeline for the SmartCare AI prototype (Task 08).

This module reproduces, for a single new patient, exactly the same
feature engineering and encoding steps Task 03 applied to the full
dataset, then applies the exact fitted scaler and the Task 06-selected
model. Kept separate from app.py so the pipeline can be unit-tested
without launching Streamlit.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

NUMERIC_SCALED_COLS = [
    "age", "systolic_bp", "diastolic_bp", "blood_sugar_mg_dl",
    "cholesterol_mg_dl", "bmi", "previous_admissions",
]

RISK_ORDER = ["Low", "Medium", "High"]

GENDER_OPTIONS = ["Female", "Male"]
BLOOD_GROUP_OPTIONS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
DEPARTMENT_OPTIONS = [
    "Cardiology", "General Medicine", "Laboratory Services",
    "Neurology", "Orthopedics", "Pediatrics", "Radiology",
]
DIAGNOSIS_OPTIONS = [
    "Asthma", "Back Pain", "Chest Pain", "Diabetes", "Fever",
    "Fracture", "Hypertension", "Kidney Infection", "Migraine", "Pneumonia",
]
ROOM_TYPE_OPTIONS = ["General Ward", "Private Room", "ICU"]
PAYMENT_METHOD_OPTIONS = ["Card", "Cash", "Insurance", "Online"]
PAYMENT_STATUS_OPTIONS = ["Paid", "Partially Paid", "Unpaid"]
APPOINTMENT_STATUS_OPTIONS = ["Scheduled", "Completed", "Cancelled", "No-Show"]


def load_model_and_scaler():
    """Load the Task 06-selected model and the Task 03 feature scaler."""
    with open(MODELS_DIR / "final_model_selection.json") as f:
        selection = json.load(f)

    model = joblib.load(MODELS_DIR / selection["best_model_file"])
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    with open(MODELS_DIR / "model_metadata.json") as f:
        model_metadata = json.load(f)

    return model, scaler, selection, model_metadata


def load_feature_columns():
    """Feature column order used to train the model - must match exactly."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    return list(X_train.columns)


def build_shap_explainer(model):
    """Fit a fast, exact SHAP LinearExplainer once, reused across predictions."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv").astype("float64")
    background = shap.sample(X_train, 100, random_state=42)
    return shap.LinearExplainer(model, background)


def _bp_category(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    else:
        return "Hypertensive"


def _bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def _blood_sugar_category(bs):
    if bs < 100:
        return "Normal"
    elif bs < 126:
        return "Prediabetic"
    else:
        return "Diabetic"


def _age_group(age):
    if age < 13:
        return "Child"
    elif age < 30:
        return "Young Adult"
    elif age < 60:
        return "Adult"
    else:
        return "Senior"


def build_feature_row(patient: dict, feature_columns: list) -> pd.DataFrame:
    """
    Convert a raw patient input dict into a single-row DataFrame matching
    the model's exact training feature columns, applying the same feature
    engineering Task 03 used (categorical banding + one-hot encoding).
    """
    row = {col: 0.0 for col in feature_columns}

    # Direct numeric passthrough fields
    row["age"] = float(patient["age"])
    row["previous_appointments"] = float(patient["previous_appointments"])
    row["missed_previous_appointments"] = float(min(
        patient["missed_previous_appointments"], patient["previous_appointments"]
    ))
    row["admitted"] = 1.0 if patient["admitted"] else 0.0
    row["length_of_stay_days"] = float(patient["length_of_stay_days"]) if patient["admitted"] else 0.0
    row["previous_admissions"] = float(patient["previous_admissions"])
    row["systolic_bp"] = float(patient["systolic_bp"])
    row["diastolic_bp"] = float(patient["diastolic_bp"])
    row["blood_sugar_mg_dl"] = float(patient["blood_sugar_mg_dl"])
    row["cholesterol_mg_dl"] = float(patient["cholesterol_mg_dl"])
    row["bmi"] = float(patient["bmi"])
    row["health_burden_score"] = row["previous_admissions"] + row["previous_appointments"]

    # One-hot categorical fields
    def set_onehot(prefix, value):
        col = f"{prefix}_{value}"
        if col in row:
            row[col] = 1.0

    set_onehot("gender", patient["gender"])
    set_onehot("blood_group", patient["blood_group"])
    set_onehot("department", patient["department"])
    set_onehot("diagnosis", patient["diagnosis"])
    set_onehot("payment_method", patient["payment_method"])
    set_onehot("payment_status", patient["payment_status"])
    set_onehot("appointment_status", patient["appointment_status"])

    room_type = patient["room_type"] if patient["admitted"] else "Not Admitted"
    set_onehot("room_type", room_type)

    # Engineered categorical bands (Task 03 logic, reproduced exactly)
    set_onehot("bp_category", _bp_category(patient["systolic_bp"], patient["diastolic_bp"]))
    set_onehot("bmi_category", _bmi_category(patient["bmi"]))
    set_onehot("blood_sugar_category", _blood_sugar_category(patient["blood_sugar_mg_dl"]))
    set_onehot("age_group", _age_group(patient["age"]))

    return pd.DataFrame([row], columns=feature_columns)


def predict(patient: dict, model, scaler, feature_columns: list, explainer=None):
    """
    Run the full pipeline for one patient: feature engineering, encoding,
    scaling, prediction, and (optionally) a SHAP explanation for the
    predicted class.
    """
    X_row = build_feature_row(patient, feature_columns)
    X_row[NUMERIC_SCALED_COLS] = scaler.transform(X_row[NUMERIC_SCALED_COLS])
    X_row = X_row.astype("float64")

    proba = model.predict_proba(X_row)[0]
    predicted_idx = int(np.argmax(proba))
    predicted_label = RISK_ORDER[predicted_idx]

    result = {
        "predicted_class": predicted_label,
        "predicted_index": predicted_idx,
        "probabilities": {RISK_ORDER[i]: float(proba[i]) for i in range(3)},
        "confidence": float(proba[predicted_idx]),
    }

    if explainer is not None:
        shap_values = explainer(X_row)
        contributions = shap_values.values[0, :, predicted_idx]
        top_idx = np.argsort(np.abs(contributions))[::-1][:5]
        # Direction is phrased relative to the predicted class itself, not
        # a generic "risk" axis - a positive SHAP value on a "Low" prediction
        # means the feature supports the Low classification, not that it
        # increases danger. Phrasing it as "increases risk" regardless of
        # which class was predicted would be actively misleading for Low
        # and Medium predictions.
        result["top_factors"] = [
            {
                "feature": feature_columns[i],
                "shap_value": float(contributions[i]),
                "direction": "supports" if contributions[i] > 0 else "argues against",
            }
            for i in top_idx
        ]

    return result
