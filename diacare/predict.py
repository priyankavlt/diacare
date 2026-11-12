"""Predict diabetes risk for one patient.

Usage:
  python -m diacare.predict --pregnancies 2 --glucose 120 --bp 70 --skin 20 \
      --insulin 80 --bmi 25 --dpf 0.5 --age 35
"""
import argparse
from functools import lru_cache

import joblib
import pandas as pd

from .config import FEATURES, MODEL_PATH


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(patient: dict) -> tuple[bool, float]:
    """Return (is_diabetic, probability) for a dict keyed by FEATURES."""
    bundle = load_model()
    row = pd.DataFrame([[float(patient[f]) for f in FEATURES]], columns=FEATURES)
    prob = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return prob >= bundle["threshold"], prob


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pregnancies", type=int, required=True)
    p.add_argument("--glucose", type=float, required=True)
    p.add_argument("--bp", type=float, required=True)
    p.add_argument("--skin", type=float, required=True)
    p.add_argument("--insulin", type=float, required=True)
    p.add_argument("--bmi", type=float, required=True)
    p.add_argument("--dpf", type=float, required=True)
    p.add_argument("--age", type=int, required=True)
    a = p.parse_args()

    positive, prob = predict(
        {
            "Pregnancies": a.pregnancies,
            "Glucose": a.glucose,
            "BloodPressure": a.bp,
            "SkinThickness": a.skin,
            "Insulin": a.insulin,
            "BMI": a.bmi,
            "DPF": a.dpf,
            "Age": a.age,
        }
    )
    label = "Diabetes likely" if positive else "Diabetes unlikely"
    print(f"{label} (probability {prob:.1%})")


if __name__ == "__main__":
    main()
