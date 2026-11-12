"""Retrain the diabetes model.

Compares several classifiers with stratified cross-validation on the training split,
refits the best one, evaluates it once on a held-out test split, and saves the full
pipeline (preprocessing + model) plus metrics.

Usage: python -m diacare.train
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .config import METRICS_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from .preprocessing import build_preprocessor, load_data, split_xy

CANDIDATES = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"model__C": [0.01, 0.1, 1, 10]},
        True,
    ),
    "RandomForest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [300],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        False,
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3],
        },
        False,
    ),
    "SVM": (
        CalibratedClassifierCV(SVC(class_weight="balanced"), ensemble=False),
        {"model__estimator__C": [0.1, 1, 10], "model__estimator__gamma": ["scale", 0.01, 0.1]},
        True,
    ),
}


def pick_threshold(model, X, y, cv) -> float:
    """Threshold maximising F1 on out-of-fold training predictions (never touches test)."""
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, oof)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(model, X, y, threshold) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main():
    df = load_data()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Unique rows: {len(df)}  train: {len(X_train)}  test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    searches = {}
    rows = []
    for name, (estimator, grid, scale) in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor(scale=scale)), ("model", estimator)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        searches[name] = search
        rows.append({"model": name, "cv_roc_auc": search.best_score_, "best_params": search.best_params_})

    leaderboard = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    print("\nCross-validated ROC-AUC (train split):")
    print(leaderboard.to_string(index=False))

    best_name = leaderboard.iloc[0]["model"]
    best = searches[best_name].best_estimator_
    threshold = pick_threshold(best, X_train, y_train, cv)
    test_metrics = evaluate(best, X_test, y_test, threshold)
    print(f"\nSelected: {best_name}  (decision threshold {threshold:.3f})")
    print("Held-out test metrics:", {k: round(v, 4) for k, v in test_metrics.items() if k != "confusion_matrix"})
    test_pred = (best.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    print(classification_report(y_test, test_pred, digits=3))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": best, "threshold": threshold, "model_name": best_name}, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sklearn_version": sklearn.__version__,
                "n_unique_rows": len(df),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "selected_model": best_name,
                "best_params": searches[best_name].best_params_,
                "cv_leaderboard": leaderboard.to_dict(orient="records"),
                "test_metrics": test_metrics,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Saved model to {MODEL_PATH} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
