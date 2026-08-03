"""
explainability.py
-------------------
Task 07 — Explainable AI Analysis
Owner: Tharindu

Reads  : models/best_model.pkl        (Ramda's Task 06 output)
         data/processed/test.csv
Writes : reports/figures/shap/*.png

Uses SHAP to interpret the best model's predictions for disease_risk_level.
SHAP is chosen over LIME here because it gives both global (which features
matter overall) and local (why THIS patient got THIS prediction) explanations
in one consistent framework — useful for both the report and the prototype.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src import config
from src.utils import load_model

SHAP_FIGURES_DIR = os.path.join(config.FIGURES_DIR, "shap")


def load_best_model_and_test_data():
    model = load_model(config.BEST_MODEL_FILE)
    test_df = pd.read_csv(config.TEST_FILE)
    X_test = test_df.drop(columns=[config.TARGET_VARIABLE])
    return model, X_test


def build_explainer(model, X_background: pd.DataFrame):
    """
    Build a SHAP explainer. TreeExplainer is used for tree-based models
    (Random Forest / XGBoost) as it's exact and fast; falls back to a
    model-agnostic Explainer for other model types (e.g. Logistic Regression).
    """
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_background)
    return explainer


def global_feature_importance(explainer, X_test: pd.DataFrame, max_display: int = 15):
    """
    Global explanation: SHAP summary/beeswarm plot showing which features
    matter most across ALL predictions, and in which direction.
    """
    shap_values = explainer.shap_values(X_test)

    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)

    os.makedirs(SHAP_FIGURES_DIR, exist_ok=True)
    fig.savefig(
        os.path.join(SHAP_FIGURES_DIR, "shap_summary_global.png"),
        bbox_inches="tight", dpi=150,
    )
    plt.show()
    return shap_values


def local_explanation(explainer, X_test: pd.DataFrame, row_index: int, class_index: int = None):
    """
    Local explanation: why did the model predict this specific patient's risk
    level? Use a force plot or waterfall plot for a single row.
    `class_index` selects which class's SHAP values to show for multi-class
    models (0=Low, 1=Medium, 2=High, matching config.CLASS_LABELS order).
    """
    row = X_test.iloc[[row_index]]
    shap_values = explainer.shap_values(row)

    fig = plt.figure()
    if isinstance(shap_values, list) and class_index is not None:
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[class_index][0],
                base_values=explainer.expected_value[class_index],
                data=row.iloc[0],
                feature_names=row.columns.tolist(),
            ),
            show=False,
        )
    else:
        shap.waterfall_plot(shap_values[0], show=False)

    os.makedirs(SHAP_FIGURES_DIR, exist_ok=True)
    fig.savefig(
        os.path.join(SHAP_FIGURES_DIR, f"shap_local_row_{row_index}.png"),
        bbox_inches="tight", dpi=150,
    )
    plt.show()


def get_top_features_for_patient(model, X_row: pd.DataFrame, explainer, top_n: int = 5) -> pd.Series:
    """
    Used by the Streamlit prototype (app/app.py) to show a short, readable
    explanation alongside each prediction — e.g. "Top factors: high blood
    sugar, elevated BMI, previous admissions."
    """
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        pred_class = model.predict(X_row)[0]
        class_idx = config.CLASS_LABELS.index(pred_class) if pred_class in config.CLASS_LABELS else 0
        values = shap_values[class_idx][0]
    else:
        values = shap_values[0]

    contributions = pd.Series(values, index=X_row.columns)
    return contributions.abs().sort_values(ascending=False).head(top_n)


if __name__ == "__main__":
    model, X_test = load_best_model_and_test_data()
    explainer = build_explainer(model, X_test)
    global_feature_importance(explainer, X_test)
    local_explanation(explainer, X_test, row_index=0)
