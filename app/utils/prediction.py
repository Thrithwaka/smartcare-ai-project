"""
Task 08 – Model/artifact loading and prediction.
Loads already-trained artifacts only. Never fits or trains anything.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High"}


class ArtifactLoadError(Exception):
    pass


class PredictionError(Exception):
    pass


def _project_root() -> Path:
    # app/utils/prediction.py -> app/utils -> app -> project root
    return Path(__file__).resolve().parents[2]


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise ArtifactLoadError(f"{label} not found at expected path: {path}")
    return path


def load_artifacts():
    """
    Loads model, scaler, feature_columns. Verifies the scaler's fitted
    feature names against the assumed scaled-columns list; raises a clear
    error if they diverge, rather than silently mis-scaling.
    Returns: (model, scaler, feature_columns, scaled_columns)
    """
    from .preprocessing import ASSUMED_SCALED_COLUMNS

    root = _project_root()
    model_path = _require_file(root / "models" / "logistic_regression_tuned.pkl", "Model")
    scaler_path = _require_file(root / "models" / "scaler.pkl", "Scaler")
    columns_path = _require_file(root / "models" / "feature_columns.pkl", "feature_columns.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(columns_path)

    if len(feature_columns) != 66:
        raise ArtifactLoadError(f"Expected 66 feature columns, found {len(feature_columns)}.")

    if hasattr(scaler, "feature_names_in_"):
        actual_scaled_cols = list(scaler.feature_names_in_)
        if set(actual_scaled_cols) != set(ASSUMED_SCALED_COLUMNS):
            raise ArtifactLoadError(
                "Scaler's fitted feature names do not match the assumed Task 03 "
                f"scaled columns.\nScaler expects: {actual_scaled_cols}\n"
                f"Assumed: {ASSUMED_SCALED_COLUMNS}\n"
                "Update ASSUMED_SCALED_COLUMNS in preprocessing.py to match, then restart."
            )
        scaled_columns = actual_scaled_cols
    else:
        # Older sklearn scaler without feature_names_in_ — fall back to assumption
        scaled_columns = ASSUMED_SCALED_COLUMNS

    return model, scaler, feature_columns, scaled_columns


def get_class_mapping(model) -> dict:
    """Prefers model.classes_ if available; falls back to the verified project mapping."""
    if hasattr(model, "classes_"):
        return {int(c): CLASS_LABELS[int(c)] for c in model.classes_}
    return CLASS_LABELS


def predict(model, final_features: pd.DataFrame) -> dict:
    """
    Runs prediction + probabilities on a single-row, fully preprocessed
    DataFrame. Verifies probabilities sum to ~1 before returning.
    """
    try:
        pred_class = model.predict(final_features)[0]
        proba = model.predict_proba(final_features)[0]
    except Exception as e:
        raise PredictionError(f"Model prediction failed: {e}") from e

    if not np.isclose(proba.sum(), 1.0, atol=1e-3):
        raise PredictionError(f"Predicted probabilities do not sum to 1 (sum={proba.sum():.4f}).")

    mapping = get_class_mapping(model)
    class_order = list(model.classes_) if hasattr(model, "classes_") else [0, 1, 2]

    return {
        "predicted_class": mapping[int(pred_class)],
        "predicted_class_index": int(pred_class),
        "probabilities": {mapping[int(c)]: float(p) for c, p in zip(class_order, proba)},
    }