"""
Task 08 – SHAP explanation for a single prediction.
Uses LinearExplainer since the final model is Logistic Regression.
Any failure here must never break the core prediction flow.
"""

import numpy as np
import pandas as pd


class ExplainabilityError(Exception):
    pass


def build_explainer(model, background_data: pd.DataFrame):
    """background_data: a sample of already-preprocessed training rows (X_train)."""
    import shap
    return shap.LinearExplainer(model, background_data)


def explain_instance(explainer, final_features: pd.DataFrame, predicted_class_index: int,
                      feature_names: list, top_n: int = 6) -> list:
    """
    Returns a list of dicts: [{"feature": str, "shap_value": float, "direction": "up"/"down"}, ...]
    sorted by absolute contribution, for the predicted class only.
    SHAP computation is unchanged from the original implementation.
    """
    shap_values = explainer.shap_values(final_features)

    if isinstance(shap_values, list):
        values_for_class = np.asarray(shap_values[predicted_class_index])[0]
    elif np.asarray(shap_values).ndim == 3:
        values_for_class = np.asarray(shap_values)[0, :, predicted_class_index]
    else:
        values_for_class = np.asarray(shap_values)[0]

    order = np.argsort(np.abs(values_for_class))[::-1][:top_n]

    results = []
    for idx in order:
        val = float(values_for_class[idx])
        results.append({
            "feature": feature_names[idx],
            "shap_value": val,
            "direction": "up" if val > 0 else "down",
        })
    return results


# ---------------------------------------------------------------------
# Display-only label mapping — UI presentation only, does not affect
# any SHAP computation, ordering, or values above.
# ---------------------------------------------------------------------

_LABEL_MAP = {
    "age": "Age", "admitted": "Admitted", "length_of_stay_days": "Length of Stay",
    "previous_admissions": "Previous Admissions", "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP", "blood_sugar_mg_dl": "Blood Sugar",
    "cholesterol_mg_dl": "Cholesterol", "bmi": "BMI", "health_burden_score": "Health Burden Score",
}


def friendly_label(raw_name: str) -> str:
    """Converts a raw model feature name into a human-readable label for the UI."""
    if raw_name in _LABEL_MAP:
        return _LABEL_MAP[raw_name]

    for prefix, base_label in [
        ("gender_", "Gender"), ("blood_group_", "Blood Group"), ("department_", "Department"),
        ("diagnosis_", "Diagnosis"), ("room_type_", "Room Type"), ("payment_method_", "Payment Method"),
        ("payment_status_", "Payment Status"), ("appointment_status_", "Appointment Status"),
        ("bp_category_", "BP Category"), ("bmi_category_", "BMI Category"),
        ("blood_sugar_category_", "Blood Sugar Category"), ("age_group_", "Age Group"),
    ]:
        if raw_name.startswith(prefix):
            value = raw_name[len(prefix):]
            return f"{base_label}: {value}"

    return raw_name.replace("_", " ").title()