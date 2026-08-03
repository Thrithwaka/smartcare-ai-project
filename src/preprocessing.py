"""
preprocessing.py
-----------------
Task 03 — Data Preprocessing & Feature Engineering
Owner: Chanuu

Reads   : data/raw/smartcare_ai_dataset_1000.csv
Writes  : data/processed/processed_full.csv
          data/processed/train.csv
          data/processed/test.csv

This module is intentionally split into small, single-purpose functions so that:
  1. Each preprocessing decision can be justified individually in the report/viva.
  2. Avishka (EDA) and Thrithwaka (modeling) can import individual functions
     rather than re-writing cleaning logic.

Run directly with:  python -m src.preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src import config


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_raw_data(filepath: str = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw SmartCare dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------------
def report_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column — use this to decide strategy."""
    missing = df.isnull().sum()
    return missing[missing > 0].sort_values(ascending=False)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values.
    Strategy (justify in report):
      - Numeric columns  -> median (robust to outliers, e.g. Blood Pressure, BMI)
      - Categorical columns -> mode (most frequent category, e.g. Blood Group)
    Adjust per-column if EDA reveals a better strategy for a specific field.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ---------------------------------------------------------------------------
# 3. Duplicates
# ---------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and drop exact duplicate rows (log how many were removed)."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows")
    return df


# ---------------------------------------------------------------------------
# 4. Outliers
# ---------------------------------------------------------------------------
def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a boolean mask of rows that are outliers for `column` using the IQR rule."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (df[column] < lower) | (df[column] > upper)


def cap_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Cap (winsorize) outliers to the IQR bounds rather than dropping rows —
    preserves sample size, which matters with only 1000 records.
    """
    df = df.copy()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ---------------------------------------------------------------------------
# 5. Encoding
# ---------------------------------------------------------------------------
def encode_categorical_features(df: pd.DataFrame, columns: list) -> tuple:
    """
    Label-encode categorical columns (e.g. Gender, Blood Group, Department, Room Type).
    Returns the encoded dataframe AND the fitted encoders (needed later so the
    Streamlit prototype can encode new user input the same way).
    """
    df = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


# ---------------------------------------------------------------------------
# 6. Scaling
# ---------------------------------------------------------------------------
def scale_numeric_features(df: pd.DataFrame, columns: list) -> tuple:
    """
    Standardize numeric columns (mean=0, std=1). Required for models sensitive
    to feature scale (Logistic Regression, SVM, KNN) — tree-based models don't
    strictly need it, but consistent input is simpler for the app/prototype.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


# ---------------------------------------------------------------------------
# 7. Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features that may help predict disease_risk_level.
    Examples (adjust once EDA/data dictionary is reviewed):
      - bmi_category        : underweight / normal / overweight / obese, from BMI
      - bp_flag             : 1 if blood pressure above a clinical threshold
      - total_charges       : sum of consultation + lab + room + medicine charges
      - admissions_ratio    : previous_admissions / (age + 1), rough risk proxy
    """
    df = df.copy()

    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, np.inf],
            labels=["underweight", "normal", "overweight", "obese"],
        )

    if "blood_pressure" in df.columns:
        df["bp_flag"] = (df["blood_pressure"] > 140).astype(int)

    charge_cols = [
        c for c in
        ["consultation_charges", "laboratory_charges", "room_charges", "medicine_charges"]
        if c in df.columns
    ]
    if charge_cols:
        df["total_charges_calculated"] = df[charge_cols].sum(axis=1)

    if "previous_admissions" in df.columns and "age" in df.columns:
        df["admissions_ratio"] = df["previous_admissions"] / (df["age"] + 1)

    return df


# ---------------------------------------------------------------------------
# 8. Feature selection
# ---------------------------------------------------------------------------
def select_features(df: pd.DataFrame, target_col: str, drop_cols: list = None) -> pd.DataFrame:
    """
    Drop identifier/leakage columns that shouldn't be fed into the model
    (e.g. Patient ID) and any columns explicitly excluded by the team.
    """
    drop_cols = drop_cols or []
    id_like = [c for c in df.columns if "id" in c.lower() and c != target_col]
    cols_to_drop = set(drop_cols + id_like)
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


# ---------------------------------------------------------------------------
# 9. Train/test split
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame, target_col: str = config.TARGET_VARIABLE):
    """Stratified split so class proportions (Low/Medium/High) are preserved."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline():
    """
    Full Task 03 pipeline, run end to end.
    Saves processed_full.csv, train.csv, test.csv to data/processed/.
    """
    df = load_raw_data()

    print("\nMissing values before cleaning:")
    print(report_missing_values(df))

    df = remove_duplicates(df)
    df = handle_missing_values(df)

    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    if config.TARGET_VARIABLE in numeric_candidates:
        numeric_candidates.remove(config.TARGET_VARIABLE)
    df = cap_outliers_iqr(df, numeric_candidates)

    df = engineer_features(df)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if config.TARGET_VARIABLE in categorical_cols:
        categorical_cols.remove(config.TARGET_VARIABLE)
    df, encoders = encode_categorical_features(df, categorical_cols)

    df = select_features(df, target_col=config.TARGET_VARIABLE)

    import os
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    df.to_csv(config.PROCESSED_FULL_FILE, index=False)
    print(f"\nSaved processed data to {config.PROCESSED_FULL_FILE}")

    X_train, X_test, y_train, y_test = split_data(df)
    train_df = X_train.copy()
    train_df[config.TARGET_VARIABLE] = y_train
    test_df = X_test.copy()
    test_df[config.TARGET_VARIABLE] = y_test

    train_df.to_csv(config.TRAIN_FILE, index=False)
    test_df.to_csv(config.TEST_FILE, index=False)
    print(f"Saved train/test splits: {train_df.shape}, {test_df.shape}")

    return df


if __name__ == "__main__":
    run_preprocessing_pipeline()
