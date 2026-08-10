"""
Centralized paths and configuration for the SmartCare AI project.
All notebooks and scripts should import paths from here rather than
hardcoding relative paths, so the project works the same way regardless
of which machine or working directory it is run from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATASET_PATH = RAW_DATA_DIR / "smartcare_ai_dataset_1000.csv"
CLEAN_DATASET_PATH = PROCESSED_DATA_DIR / "smartcare_clean_dataset.csv"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

RANDOM_STATE = 42

RISK_ORDER = ["Low", "Medium", "High"]
RISK_LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

for directory in [MODELS_DIR, FIGURES_DIR / "models"]:
    directory.mkdir(parents=True, exist_ok=True)
