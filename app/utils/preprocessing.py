"""
Task 08 – Preprocessing module.
Reproduces the Task 03 feature-engineering and encoding pipeline for a
single-instance inference input. No fitting occurs here — only transform
using already-fitted artifacts (scaler, feature_columns).
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Task 03 feature-engineering thresholds (exact reuse — do not modify)
# ---------------------------------------------------------------------

def _bp_category(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return 'Normal'
    elif systolic < 130 and diastolic < 80:
        return 'Elevated'
    return 'Hypertensive'


def _bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    return 'Obese'


def _blood_sugar_category(bs):
    if bs < 100:
        return 'Normal'
    elif bs < 126:
        return 'Prediabetic'
    return 'Diabetic'


def _age_group(age):
    if age < 13:
        return 'Child'
    elif age < 30:
        return 'Young Adult'
    elif age < 60:
        return 'Adult'
    return 'Senior'


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

REQUIRED_FIELDS = [
    'age', 'gender', 'blood_group', 'admitted', 'length_of_stay_days',
    'room_type', 'previous_admissions', 'previous_appointments',
    'systolic_bp', 'diastolic_bp', 'blood_sugar_mg_dl', 'cholesterol_mg_dl',
    'bmi', 'department', 'diagnosis', 'appointment_status',
    'payment_method', 'payment_status',
]

NUMERIC_FIELDS = [
    'age', 'length_of_stay_days', 'previous_admissions', 'previous_appointments',
    'systolic_bp', 'diastolic_bp', 'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi',
]


class PreprocessingError(Exception):
    """Raised for any validation or feature-construction failure."""
    pass


def validate_raw_input(raw_input: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in raw_input or raw_input[f] is None]
    if missing:
        raise PreprocessingError(f"Missing required fields: {', '.join(missing)}")

    for f in NUMERIC_FIELDS:
        val = raw_input[f]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise PreprocessingError(f"Field '{f}' must be numeric, got {type(val).__name__}.")
        if np.isnan(val):
            raise PreprocessingError(f"Field '{f}' cannot be NaN.")

    if raw_input['admitted'] not in (0, 1):
        raise PreprocessingError("Field 'admitted' must be 0 or 1.")


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def engineer_features(raw_input: dict) -> dict:
    """Adds derived fields to a copy of raw_input. Returns the enriched dict."""
    enriched = dict(raw_input)

    enriched['health_burden_score'] = (
        raw_input['previous_admissions'] + raw_input['previous_appointments']
    )
    enriched['bp_category'] = _bp_category(raw_input['systolic_bp'], raw_input['diastolic_bp'])
    enriched['bmi_category'] = _bmi_category(raw_input['bmi'])
    enriched['blood_sugar_category'] = _blood_sugar_category(raw_input['blood_sugar_mg_dl'])
    enriched['age_group'] = _age_group(raw_input['age'])

    return enriched


# ---------------------------------------------------------------------
# Encoding + alignment
# ---------------------------------------------------------------------

NOMINAL_FIELDS = [
    'gender', 'blood_group', 'department', 'diagnosis', 'room_type',
    'payment_method', 'payment_status', 'appointment_status',
    'bp_category', 'bmi_category', 'blood_sugar_category', 'age_group',
]

NUMERIC_MODEL_FIELDS = [
    'age', 'admitted', 'length_of_stay_days', 'previous_admissions',
    'systolic_bp', 'diastolic_bp', 'blood_sugar_mg_dl', 'cholesterol_mg_dl',
    'bmi', 'health_burden_score',
]


def build_raw_row(enriched: dict) -> pd.DataFrame:
    """Single-row DataFrame with raw + engineered values, pre-encoding."""
    return pd.DataFrame([enriched])


def encode_and_align(raw_row: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    One-hot encodes nominal fields, then aligns strictly to feature_columns:
    missing expected columns are added as 0, unexpected columns are dropped,
    final column order matches feature_columns exactly.
    """
    encoded = pd.get_dummies(raw_row, columns=NOMINAL_FIELDS)

    aligned = pd.DataFrame(0, index=encoded.index, columns=feature_columns)
    common_cols = [c for c in encoded.columns if c in feature_columns]
    aligned[common_cols] = encoded[common_cols]

    for col in NUMERIC_MODEL_FIELDS:
        if col in aligned.columns:
            aligned[col] = encoded[col].values if col in encoded.columns else aligned[col]

    if aligned.shape[1] != len(feature_columns):
        raise PreprocessingError(
            f"Feature construction produced {aligned.shape[1]} columns, expected {len(feature_columns)}."
        )
    if list(aligned.columns) != list(feature_columns):
        raise PreprocessingError("Constructed feature order does not match feature_columns.pkl.")

    return aligned


# ---------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------

ASSUMED_SCALED_COLUMNS = [
    'age', 'systolic_bp', 'diastolic_bp', 'blood_sugar_mg_dl',
    'cholesterol_mg_dl', 'bmi', 'previous_admissions', 'health_burden_score',
]


def apply_scaling(aligned_row: pd.DataFrame, scaler, scaled_columns: list) -> pd.DataFrame:
    """Applies an already-fitted scaler.transform() to the given columns only."""
    result = aligned_row.copy()
    missing = [c for c in scaled_columns if c not in result.columns]
    if missing:
        raise PreprocessingError(f"Scaler expects columns not present in constructed features: {missing}")

    result[scaled_columns] = scaler.transform(result[scaled_columns].astype(float))
    return result


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def prepare_input(raw_input: dict, feature_columns: list, scaler, scaled_columns: list) -> pd.DataFrame:
    """
    Full pipeline: validate -> engineer -> build row -> encode/align -> scale.
    Returns a single-row DataFrame with exactly len(feature_columns) columns,
    in the exact order required by the model.
    """
    validate_raw_input(raw_input)
    enriched = engineer_features(raw_input)
    raw_row = build_raw_row(enriched)
    aligned = encode_and_align(raw_row, feature_columns)
    final = apply_scaling(aligned, scaler, scaled_columns)
    return final