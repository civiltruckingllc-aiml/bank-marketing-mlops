"""Read the DVC-tracked UCI dataset and make deterministic data splits."""
import hashlib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.config import project_path


def features(cfg):
    return cfg["data"]["numeric_features"] + cfg["data"]["categorical_features"]


def data_version(cfg):
    path = project_path(cfg["paths"]["dataset"])
    pointer = project_path(cfg["paths"]["dvc_pointer"])
    if not path.exists():
        raise FileNotFoundError("Dataset missing. Run 'dvc pull' from the project root.")
    with pointer.open(encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    # Deferred imports pin their source ETag. An independent content MD5
    # also protects training against any accidental/malicious source changes.
    expected = metadata["outs"][0].get("md5", cfg["data"]["expected_dataset_md5"])
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Dataset hash does not match the pinned data version")
    return actual


def load_dataset(cfg):
    data_version(cfg)
    frame = pd.read_csv(project_path(cfg["paths"]["dataset"]), sep=";", na_values=["unknown"])
    expected = features(cfg) + [cfg["data"]["target"]]
    missing = set(expected).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    frame = frame.loc[:, expected].copy(deep=True)
    if not frame[cfg["data"]["target"]].isin(["yes", "no"]).all():
        raise ValueError("Target must contain only yes/no")
    frame[cfg["data"]["target"]] = frame[cfg["data"]["target"]].map({"no": 0, "yes": 1})
    rng = np.random.default_rng(cfg["project"]["random_state"])
    for column in cfg["data"]["missing_numeric_features"]:
        mask = rng.random(len(frame)) < cfg["data"]["simulated_missing_fraction"]
        frame.loc[mask, column] = np.nan
    return frame


def split_dataset(frame, cfg):
    """Return train/validation/test; fit preprocessors only on train later."""
    target = cfg["data"]["target"]
    train_val, test = train_test_split(
        frame, test_size=cfg["data"]["test_size"],
        random_state=cfg["project"]["random_state"], stratify=frame[target],
    )
    relative_val_size = cfg["data"]["validation_size"] / (1 - cfg["data"]["test_size"])
    train, validation = train_test_split(
        train_val, test_size=relative_val_size,
        random_state=cfg["project"]["random_state"], stratify=train_val[target],
    )
    return train.copy(), validation.copy(), test.copy()
