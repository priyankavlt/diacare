from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "kaggle_diabetes.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "Outcome"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DPF",
    "Age",
]
# A value of 0 is physiologically impossible for these, so it marks a missing reading.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
