"""Configuration and project-relative paths, independent of current directory."""
from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path="configs/train.yaml"):
    with project_path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for key in ("project", "paths", "data", "preprocessing", "model", "evaluation", "monitoring"):
        if key not in cfg:
            raise ValueError(f"Missing configuration section: {key}")
    data = cfg["data"]
    if not 0 < data["test_size"] < 1 or not 0 < data["validation_size"] < 1:
        raise ValueError("Split fractions must be between zero and one")
    if data["test_size"] + data["validation_size"] >= 1:
        raise ValueError("Validation and test fractions must sum to less than one")
    if not 0 <= data["simulated_missing_fraction"] < 1:
        raise ValueError("Missing fraction must be in [0, 1)")
    if cfg["model"]["type"] not in ("random_forest", "logistic_regression"):
        raise ValueError("Unsupported model type")
    return cfg


def experiment_config(base, override):
    """Replace the entire model block; avoid RF parameters leaking into logistic."""
    cfg = deepcopy(base)
    cfg["model"] = deepcopy(override["model"])
    return cfg


def flatten(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = str(value) if isinstance(value, (list, tuple)) else value
    return result
