from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor


def build_model(cfg):
    params = dict(cfg["model"]["params"])
    params["random_state"] = cfg["project"]["random_state"]
    kind = cfg["model"]["type"]
    if kind == "random_forest":
        model = RandomForestClassifier(**params)
    elif kind == "logistic_regression":
        model = LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model: {kind}")
    return Pipeline([("preprocessor", build_preprocessor(cfg)), ("classifier", model)])
