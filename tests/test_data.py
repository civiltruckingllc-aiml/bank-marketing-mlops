from src.data import data_version, features, load_dataset, split_dataset
from pandas.testing import assert_frame_equal


def test_expected_columns_and_required_size(dataset, config):
    assert set(dataset.columns) == set(features(config) + [config["data"]["target"]])
    assert len(dataset) == 45211
    assert len(features(config)) >= 8
    assert "duration" not in dataset.columns


def test_target_contains_only_expected_values(dataset, config):
    target = dataset[config["data"]["target"]]
    assert not target.isna().any()
    assert set(target.unique()) == {0, 1}
    assert 0.10 < target.mean() < 0.15


def test_numeric_features_are_in_expected_ranges(dataset, config):
    ranges = {"age": (18, 100), "balance": (-10000, 110000), "day": (1, 31),
              "campaign": (1, 65), "pdays": (-1, 900), "previous": (0, 300)}
    for column in config["data"]["numeric_features"]:
        low, high = ranges[column]
        assert dataset[column].dropna().between(low, high).all(), column


def test_dataset_has_both_types_and_real_missing_values(dataset, config):
    assert dataset[config["data"]["numeric_features"]].isna().any().any()
    assert dataset[config["data"]["categorical_features"]].isna().any().any()
    assert dataset["job"].dtype == object


def test_data_loading_is_deterministic(dataset, config):
    assert_frame_equal(dataset, load_dataset(config))
    assert len(data_version(config)) == 32


def test_train_validation_test_are_disjoint_and_complete(dataset, config):
    train, validation, test = split_dataset(dataset, config)
    a, b, c = set(train.index), set(validation.index), set(test.index)
    assert not a & b and not a & c and not b & c
    assert a | b | c == set(dataset.index)
