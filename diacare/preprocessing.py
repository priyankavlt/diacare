import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import DATA_PATH, FEATURES, TARGET, ZERO_AS_MISSING


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the dataset and drop exact duplicate rows.

    The Kaggle CSV has 2000 rows but only 744 are unique. Keeping duplicates lets
    identical patients land in both train and test, which inflates test accuracy.
    """
    df = pd.read_csv(path).rename(columns={"DiabetesPedigreeFunction": "DPF"})
    return df.drop_duplicates().reset_index(drop=True)


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]


def _zeros_to_nan(X):
    X = np.asarray(X, dtype=float).copy()
    X[X == 0] = np.nan
    return X


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Zero -> NaN -> median impute (fit on training data only) -> optional scaling."""
    missing_steps = [
        ("zeros_to_nan", FunctionTransformer(_zeros_to_nan, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
    ]
    other_steps = []
    if scale:
        missing_steps.append(("scale", StandardScaler()))
        other_steps.append(("scale", StandardScaler()))
    other = [f for f in FEATURES if f not in ZERO_AS_MISSING]
    return ColumnTransformer(
        [
            ("missing", Pipeline(missing_steps), ZERO_AS_MISSING),
            ("other", Pipeline(other_steps) if other_steps else "passthrough", other),
        ]
    )
