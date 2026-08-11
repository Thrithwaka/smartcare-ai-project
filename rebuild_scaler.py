import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
numeric_cols_to_scale = ['age', 'systolic_bp', 'diastolic_bp', 'blood_sugar_mg_dl',
                          'cholesterol_mg_dl', 'bmi', 'previous_admissions']

# Reload the unscaled clean dataset
df_final = pd.read_csv(PROCESSED_DIR / "smartcare_clean_dataset.csv")

X = df_final.drop(columns=['disease_risk_level'])
y = df_final['disease_risk_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Fit scaler exactly as notebook 01 did
scaler = StandardScaler()
X_train[numeric_cols_to_scale] = scaler.fit_transform(X_train[numeric_cols_to_scale])

# --- Sanity check: this X_train should match the already-saved X_train.csv ---
existing_X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
match = (X_train.reset_index(drop=True).round(6) == existing_X_train.round(6)).all().all()
print("Reconstructed X_train matches saved X_train.csv:", match)

if match:
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("scaler.pkl saved to", MODELS_DIR / "scaler.pkl")
else:
    print("MISMATCH — do not trust this scaler. See notes below.")