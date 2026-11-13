<img src="assets/icon.svg" width="64" height="64" alt="">

# DiaCare

Diabetes risk prediction with a Flask web app, rebuilt from
[shsarv/Machine-Learning-Projects – Diabetes Prediction [END 2 END]](https://github.com/shsarv/Machine-Learning-Projects/tree/main/Diabetes%20Prediction%20%5BEND%202%20END%5D)
with a retrained, leak-free model.

> Educational demo only. This is not a medical diagnosis tool.

## Quick start

```bash
uv venv && uv pip install -r requirements.txt     # or: python -m venv .venv && pip install -r requirements.txt
python -m diacare.train                           # retrain -> model/diabetes_model.joblib + model/metrics.json
python -m flask --app app.app run                 # http://127.0.0.1:5000
python -m diacare.predict --pregnancies 2 --glucose 120 --bp 70 --skin 20 \
    --insulin 80 --bmi 25 --dpf 0.5 --age 35      # CLI prediction
pytest                                            # tests
```

Deploy (Heroku/Render style): `gunicorn app.app:app` (see `Procfile`).

## What changed vs. the original

| Issue in original | Fix |
|---|---|
| `kaggle_diabetes.csv` has 2000 rows but only **744 unique**; random splits put identical patients in train *and* test, so the reported ~99% accuracy was leakage | Exact duplicates are dropped before splitting |
| Deployed training script imputed zeros into `df_copy` but trained on the raw `df` | Zero → NaN → median imputation lives inside the model pipeline |
| Imputation statistics computed on the full dataset (test leakage) | Imputer/scaler are fit on training folds only |
| `train.py`/`evaluate.py` unpack 5 values from a function returning 4 (crash) | Single working `diacare.train` entry point |
| Single untuned Random Forest, `pickle` | Logistic Regression, Random Forest, Gradient Boosting and calibrated SVM compared via 5-fold stratified CV + grid search; best pipeline saved with `joblib` |
| Fixed 0.5 threshold | Decision threshold chosen to maximise F1 on out-of-fold training predictions |
| Free-text inputs, no validation | Numeric inputs with range validation; result shows probability |

## Current model

Selected by CV ROC-AUC: **calibrated RBF SVM** (C=1, gamma=0.1), threshold 0.276.

| Model | CV ROC-AUC |
|---|---|
| SVM (calibrated) | 0.836 |
| Logistic Regression | 0.831 |
| Random Forest | 0.824 |
| Gradient Boosting | 0.820 |

Held-out test set (149 unique patients, never seen during selection):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.718 | 0.563 | 0.784 | 0.656 | 0.818 |

The threshold favours recall (catching ~78% of diabetic patients), which suits screening.
These numbers are in line with published results on the Pima Indians dataset; full details are in `model/metrics.json`.

## Layout

```
data/kaggle_diabetes.csv     dataset
diacare/                     config, preprocessing, train, predict
model/                       trained pipeline + metrics
app/                         Flask app, templates, static images
tests/                       pytest
```

## License

Licensed under the [Apache License 2.0](LICENSE).

### Third-party material

The Apache license covers the original work in this repository. These parts come from other sources,
remain under their authors' terms, and are **not** covered by the Apache license:

- `data/kaggle_diabetes.csv` – Kaggle version of the Pima Indians Diabetes Database
  (originally from the National Institute of Diabetes and Digestive and Kidney Diseases),
  as included in [shsarv/Machine-Learning-Projects](https://github.com/shsarv/Machine-Learning-Projects)
- `app/static/` images (`diabetes.webp`, `no-diabetes.webp`, `wall.jpg`, `wallpaper.jpg`) –
  from [shsarv/Machine-Learning-Projects](https://github.com/shsarv/Machine-Learning-Projects)
  by Sarvesh Kumar Sharma

The web app's layout and the overall approach are adapted from that same project.
