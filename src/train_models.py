"""
train_models.py
-----------------
Task 05 — Machine Learning Model Development
Owner: Thrithwaka

Reads  : data/processed/train.csv, data/processed/test.csv  (Chanuu's output)
Writes : models/logistic_regression.pkl
         models/random_forest.pkl
         models/xgboost_model.pkl

Trains at least THREE classification models for the multi-class
disease_risk_level problem, with hyperparameter tuning via GridSearchCV.
Ramda's evaluate.py then loads these saved models to score them independently.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src import config
from src.utils import save_model, save_metadata


def load_train_test():
    """Load the pre-split train/test sets produced by Chanuu's pipeline."""
    train_df = pd.read_csv(config.TRAIN_FILE)
    test_df = pd.read_csv(config.TEST_FILE)

    X_train = train_df.drop(columns=[config.TARGET_VARIABLE])
    y_train = train_df[config.TARGET_VARIABLE]
    X_test = test_df.drop(columns=[config.TARGET_VARIABLE])
    y_test = test_df[config.TARGET_VARIABLE]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model 1: Logistic Regression (baseline, interpretable)
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train, y_train):
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    }
    grid = GridSearchCV(
        LogisticRegression(multi_class="multinomial", random_state=config.RANDOM_SEED),
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Logistic Regression best params: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------------------
# Model 2: Random Forest (non-linear, handles mixed feature types well)
# ---------------------------------------------------------------------------
def train_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=config.RANDOM_SEED, class_weight="balanced"),
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Random Forest best params: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------------------
# Model 3: XGBoost (gradient boosting — usually strongest baseline)
# ---------------------------------------------------------------------------
def train_xgboost(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    }
    grid = GridSearchCV(
        XGBClassifier(
            objective="multi:softprob",
            num_class=len(config.CLASS_LABELS),
            eval_metric="mlogloss",
            random_state=config.RANDOM_SEED,
        ),
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"XGBoost best params: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------------------
# Train all models and save
# ---------------------------------------------------------------------------
def train_all_models():
    X_train, X_test, y_train, y_test = load_train_test()

    results = {}

    lr_model, lr_params = train_logistic_regression(X_train, y_train)
    save_model(lr_model, config.MODEL_FILES["logistic_regression"])
    results["logistic_regression"] = lr_params

    rf_model, rf_params = train_random_forest(X_train, y_train)
    save_model(rf_model, config.MODEL_FILES["random_forest"])
    results["random_forest"] = rf_params

    xgb_model, xgb_params = train_xgboost(X_train, y_train)
    save_model(xgb_model, config.MODEL_FILES["xgboost"])
    results["xgboost"] = xgb_params

    save_metadata(
        {"models_trained": list(results.keys()), "best_params": results},
        config.MODEL_METADATA_FILE,
    )

    print("\nAll models trained and saved. Hand off to Ramda for evaluation.")
    return results


if __name__ == "__main__":
    train_all_models()
