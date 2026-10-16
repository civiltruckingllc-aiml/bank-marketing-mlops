"""Validated, non-mutating feature preparation and sklearn transformers."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import features


def prepare_features(frame, cfg):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("Input dataframe must not be empty")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique")
    missing = set(features(cfg)).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    result = frame.loc[:, features(cfg)].copy(deep=True)
    for column in cfg["data"]["numeric_features"]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        if np.isinf(result[column]).any():
            raise ValueError(f"Infinite values in {column}")
    for column in cfg["data"]["categorical_features"]:
        result[column] = result[column].astype(object).replace({"unknown": np.nan, "": np.nan})
    return result


def build_preprocessor(cfg):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["preprocessing"]["numeric_imputation"], keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["preprocessing"]["categorical_imputation"], keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown=cfg["preprocessing"]["unknown_categories"], sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, cfg["data"]["numeric_features"]),
        ("categorical", categorical, cfg["data"]["categorical_features"]),
    ], remainder="drop")
