"""
config.py
Shared project configuration: file paths, constants, random seed.
Import this in every module so paths are consistent across the whole team.
"""

import os

# --- Base paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, "smartcare_ai_dataset_1000.csv")
DATA_DICTIONARY_FILE = os.path.join(DATA_RAW_DIR, "smartcare_ai_dataset_data_dictionary.csv")

TRAIN_FILE = os.path.join(DATA_PROCESSED_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_PROCESSED_DIR, "test.csv")
PROCESSED_FULL_FILE = os.path.join(DATA_PROCESSED_DIR, "processed_full.csv")

MODEL_FILES = {
    "logistic_regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
    "random_forest": os.path.join(MODELS_DIR, "random_forest.pkl"),
    "xgboost": os.path.join(MODELS_DIR, "xgboost_model.pkl"),
}
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.pkl")
MODEL_METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")
MODEL_COMPARISON_TABLE = os.path.join(REPORTS_DIR, "model_comparison_table.csv")

# --- Task selection: OPTION C selected ---
TARGET_VARIABLE = "disease_risk_level"
CLASS_LABELS = ["Low", "Medium", "High"]
PROBLEM_TYPE = "multiclass"

# --- Reproducibility ---
RANDOM_SEED = 42
TEST_SIZE = 0.2
