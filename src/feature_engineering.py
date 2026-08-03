"""
feature_engineering.py
------------------------
Task 03 — Data Preprocessing & Feature Engineering (continued)
Owner: Chanuu

Separated from preprocessing.py so that "cleaning" and "creating new features"
are clearly distinct steps in the report and easy to discuss separately in the
viva. Core feature creation logic lives in preprocessing.engineer_features();
this file holds feature SELECTION helpers (correlation-based, variance-based)
used to justify which engineered/raw features actually make it into the model.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def correlation_with_target(df: pd.DataFrame, target_col: str, top_n: int = 15) -> pd.Series:
    """
    Rank numeric features by absolute correlation with a (numerically-encoded)
    target. Useful as a first-pass filter before modeling — discuss any
    surprising results in the report (e.g. a clinical variable with weak
    correlation might still be clinically meaningful).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        raise ValueError(f"'{target_col}' must be numeric-encoded before calling this.")
    corr = numeric_df.corr()[target_col].drop(target_col)
    return corr.abs().sort_values(ascending=False).head(top_n)


def mutual_information_ranking(X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> pd.Series:
    """
    Rank features by mutual information with the (multi-class) target.
    Better than Pearson correlation here since disease_risk_level is
    categorical/ordinal, not continuous — MI captures non-linear relationships.
    """
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)
    return mi_series.sort_values(ascending=False).head(top_n)


def drop_low_variance_features(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Drop numeric columns with variance below `threshold` — they carry ~no signal."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    variances = df[numeric_cols].var()
    low_var_cols = variances[variances < threshold].index.tolist()
    if low_var_cols:
        print(f"Dropping low-variance columns: {low_var_cols}")
    return df.drop(columns=low_var_cols)


def drop_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Drop one feature from each pair of numeric features correlated above
    `threshold` — reduces redundancy/multicollinearity (relevant for
    Logistic Regression interpretability).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        print(f"Dropping highly correlated columns: {to_drop}")
    return df.drop(columns=to_drop)
