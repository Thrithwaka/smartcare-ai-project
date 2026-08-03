"""
evaluate.py
------------
Task 06 — Model Evaluation
Owner: Ramda

Reads  : models/*.pkl               (Thrithwaka's Task 05 output)
         data/processed/test.csv    (Chanuu's Task 03 output)
Writes : reports/model_comparison_table.csv
         reports/figures/evaluation/*.png
         models/best_model.pkl      (copy of the winning model, for Tharindu's app)

Since disease_risk_level is multi-class (Low/Medium/High), the required
metrics are: Accuracy, Precision, Recall, F1 (macro-averaged to treat all
three classes equally regardless of class imbalance), and Confusion Matrix.
"""

import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src import config
from src.utils import load_model, save_metadata

EVAL_FIGURES_DIR = os.path.join(config.FIGURES_DIR, "evaluation")


def load_test_set():
    test_df = pd.read_csv(config.TEST_FILE)
    X_test = test_df.drop(columns=[config.TARGET_VARIABLE])
    y_test = test_df[config.TARGET_VARIABLE]
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute the full multi-class metric set for one model.
    macro-average is used (not weighted) so a minority class like "High risk"
    doesn't get overshadowed by "Low risk" if classes are imbalanced —
    important for a healthcare use case where missing "High risk" is costly.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    plot_confusion_matrix(y_test, y_pred, model_name)

    return metrics


def plot_confusion_matrix(y_test, y_pred, model_name: str):
    """Save a labeled confusion matrix heatmap for one model."""
    cm = confusion_matrix(y_test, y_pred, labels=config.CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_LABELS, yticklabels=config.CLASS_LABELS, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")

    os.makedirs(EVAL_FIGURES_DIR, exist_ok=True)
    fig.savefig(
        os.path.join(EVAL_FIGURES_DIR, f"confusion_matrix_{model_name}.png"),
        bbox_inches="tight", dpi=150,
    )
    plt.show()


def compare_all_models() -> pd.DataFrame:
    """
    Load every model saved by Thrithwaka, evaluate each on the held-out test
    set, and produce a single comparison table — the key Task 06 deliverable.
    """
    X_test, y_test = load_test_set()
    all_metrics = []

    for model_name, filepath in config.MODEL_FILES.items():
        if not os.path.exists(filepath):
            print(f"Skipping {model_name} — file not found at {filepath}")
            continue
        model = load_model(filepath)
        metrics = evaluate_model(model, X_test, y_test, model_name)
        all_metrics.append(metrics)

    comparison_df = pd.DataFrame(all_metrics).sort_values("f1_macro", ascending=False)

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    comparison_df.to_csv(config.MODEL_COMPARISON_TABLE, index=False)
    print(f"\nSaved comparison table to {config.MODEL_COMPARISON_TABLE}")
    print(comparison_df)

    return comparison_df


def select_and_save_best_model(comparison_df: pd.DataFrame):
    """
    Copy the highest macro-F1 model to models/best_model.pkl — this is the
    single file Tharindu's prototype (app/app.py) and explainability.py load.
    Justify the choice of macro-F1 (over accuracy) in the report: it's the
    fairer metric when classes aren't perfectly balanced.
    """
    best_row = comparison_df.iloc[0]
    best_model_name = best_row["model"]
    source_path = config.MODEL_FILES[best_model_name]
    shutil.copy(source_path, config.BEST_MODEL_FILE)

    save_metadata(
        {
            "best_model": best_model_name,
            "metrics": best_row.to_dict(),
            "selection_criterion": "highest macro-F1 on held-out test set",
        },
        os.path.join(config.MODELS_DIR, "best_model_selection.json"),
    )
    print(f"\nBest model: {best_model_name} -> saved as {config.BEST_MODEL_FILE}")
    return best_model_name


if __name__ == "__main__":
    comparison = compare_all_models()
    select_and_save_best_model(comparison)
