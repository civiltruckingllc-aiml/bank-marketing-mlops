import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.preprocessing import build_preprocessor, prepare_features


def test_numeric_missing_values_are_imputed(sample, config):
    X = prepare_features(sample, config)
    X.loc[X.index[:10], "age"] = np.nan
    transformer = build_preprocessor(config)
    transformed = transformer.fit_transform(X)
    assert np.isfinite(transformed).all()
    # Imputation must learn the training median, not a constant fill.
    learned = transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
    assert learned == pytest.approx(X["age"].median())


def test_categorical_missing_values_are_imputed(sample, config):
    X = prepare_features(sample, config)
    X.loc[X.index[:15], "job"] = np.nan
    transformer = build_preprocessor(config)
    transformed = transformer.fit_transform(X)
    assert np.isfinite(transformed).all()
    cat = transformer.named_transformers_["categorical"]
    assert cat.named_steps["imputer"].statistics_[0] == X["job"].mode().iloc[0]


def test_categorical_variables_use_one_hot_encoding(sample, config):
    X = prepare_features(sample, config)
    transformer = build_preprocessor(config)
    transformer.fit(X)
    encoded = transformer.named_transformers_["categorical"].transform(X[config["data"]["categorical_features"]])
    assert set(np.unique(encoded)).issubset({0.0, 1.0})
    assert np.all(encoded.sum(axis=1) == len(config["data"]["categorical_features"]))
    assert encoded.shape[1] > len(config["data"]["categorical_features"])


def test_preprocessing_does_not_modify_original(sample, config):
    before = sample.copy(deep=True)
    X = prepare_features(sample, config)
    build_preprocessor(config).fit_transform(X)
    assert_frame_equal(sample, before)
    X.iloc[0, 0] = -999
    assert_frame_equal(sample, before)


def test_missing_required_column_raises(sample, config):
    with pytest.raises(ValueError, match="Missing required features"):
        prepare_features(sample.drop(columns="age"), config)


def test_non_dataframe_input_raises(config):
    with pytest.raises(TypeError, match="DataFrame"):
        prepare_features([[1, 2]], config)


def test_empty_dataframe_raises(sample, config):
    with pytest.raises(ValueError, match="empty"):
        prepare_features(sample.iloc[:0], config)


def test_unseen_category_does_not_break_inference(sample, config):
    X = prepare_features(sample, config)
    transformer = build_preprocessor(config).fit(X)
    unknown = X.iloc[:2].copy()
    unknown["job"] = "brand-new-occupation"
    assert transformer.transform(unknown).shape == transformer.transform(X.iloc[:2]).shape


def test_invalid_numeric_input_raises(sample, config):
    bad = sample.copy()
    bad["age"] = bad["age"].astype(object)
    bad.loc[bad.index[0], "age"] = "not-a-number"
    with pytest.raises(ValueError):
        prepare_features(bad, config)


def test_infinite_numeric_input_raises(sample, config):
    sample.loc[sample.index[0], "balance"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        prepare_features(sample, config)


def test_validation_transform_reuses_training_median(sample, config):
    X = prepare_features(sample, config)
    transformer = build_preprocessor(config).fit(X)
    median = transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_.copy()
    current = X.iloc[:3].copy()
    current["age"] = [np.nan, 999, 999]
    transformer.transform(current)
    np.testing.assert_array_equal(transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_, median)
