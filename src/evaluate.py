"""Metrics, gates, and a once-per-selection final holdout evaluation."""
import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, X, y, threshold=0.5):
    probability = model.predict_proba(X)[:, 1]
    predictions = (probability >= threshold).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
    }


def enforce_thresholds(metrics, cfg):
    failures = []
    for metric in ("roc_auc", "average_precision"):
        value = metrics.get(metric, float("nan"))
        minimum = cfg["evaluation"][f"min_{metric}"]
        if not np.isfinite(value) or value < minimum:
            failures.append(f"{metric}={value:.4f} < {minimum:.4f}")
    if failures:
        raise ValueError("Performance gate failed: " + "; ".join(failures))


def main():
    import mlflow
    from src.config import load_config, project_path
    from src.data import data_version, load_dataset, split_dataset
    from src.preprocessing import prepare_features
    from src.tracking import configure_tracking
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_tracking(cfg)
    run_id = args.run_id
    if run_id is None:
        selection = json.loads(project_path("reports/best_run.json").read_text())
        run_id = selection["run_id"]
    run = mlflow.get_run(run_id)
    if run.data.params["data_version"] != data_version(cfg):
        raise ValueError("Selected model was trained on a different data version")
    # Use the training run's exact configuration to reproduce split and features.
    effective_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="effective_config.json")
    effective = json.loads(open(effective_path, encoding="utf-8").read())
    _, _, test = split_dataset(load_dataset(effective), effective)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    metrics = evaluate_model(model, prepare_features(test, effective), test[effective["data"]["target"]],
                             effective["evaluation"]["decision_threshold"])
    result = {"run_id": run_id, "rows": len(test), "metrics": metrics,
              "note": "Held out from training and experiment selection; simulated project evaluation only."}
    output = project_path(cfg["paths"]["reports_dir"]) / "holdout_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        mlflow.log_artifact(str(output), artifact_path="evaluation")
    print(json.dumps(result, indent=2))
    enforce_thresholds(metrics, effective)


if __name__ == "__main__":
    main()
