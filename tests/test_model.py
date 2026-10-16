from copy import deepcopy
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from src.data import split_dataset
from src.evaluate import enforce_thresholds, evaluate_model
from src.modeling import build_model
from src.preprocessing import prepare_features


@pytest.fixture(scope="module")
def fitted(config, dataset):
    cfg = deepcopy(config)
    cfg["model"]["params"].update(n_estimators=60, max_depth=8, min_samples_leaf=8)
    # Select a small stratified subset of training data only.
    train, validation, _ = split_dataset(dataset, cfg)
    small, _ = train_test_split(train, train_size=5000, random_state=42, stratify=train["y"])
    model = build_model(cfg)
    model.fit(prepare_features(small, cfg), small["y"])
    return model, prepare_features(validation, cfg), validation["y"], cfg


def test_predictions_have_correct_shape_type_and_probabilities(fitted):
    model, X, _, _ = fitted
    prediction = model.predict(X)
    probability = model.predict_proba(X)
    assert prediction.shape == (len(X),)
    assert np.issubdtype(prediction.dtype, np.integer)
    assert set(np.unique(prediction)).issubset({0, 1})
    assert probability.shape == (len(X), 2)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)


def test_model_exceeds_minimum_performance_on_known_validation_set(fitted):
    model, X, y, cfg = fitted
    metrics = evaluate_model(model, X, y)
    enforce_thresholds(metrics, cfg)


@pytest.mark.parametrize("metrics", [
    {"roc_auc": 0.50, "average_precision": 0.5},
    {"roc_auc": 0.85, "average_precision": 0.05},
    {"roc_auc": float("nan"), "average_precision": 0.5},
])
def test_performance_gate_rejects_bad_or_nonfinite_metrics(metrics, config):
    with pytest.raises(ValueError, match="Performance gate failed"):
        enforce_thresholds(metrics, config)
