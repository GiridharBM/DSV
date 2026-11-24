"""Preprocessing pipeline construction: imputation + one-hot encoding + scaling."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
    ])
    return ColumnTransformer(
        [("num", numeric, config.NUMERIC_FEATURES), ("cat", categorical, config.CATEGORICAL_FEATURES)],
        remainder="drop",
        sparse_threshold=1.0,
    )
