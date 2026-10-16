"""Config-driven training with complete MLflow logging and enforced gates."""
import argparse
import hashlib
import json
import subprocess
import uuid
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from src.config import ROOT, flatten, load_config, project_path
from src.data import data_version, load_dataset, split_dataset
from src.evaluate import enforce_thresholds, evaluate_model
from src.modeling import build_model
from src.preprocessing import prepare_features
from src.tracking import configure_tracking


def train(cfg, run_name="configured-training", batch_id=None):
    configure_tracking(cfg)
    frame = load_dataset(cfg)
    train_frame, validation, _ = split_dataset(frame, cfg)
    X_train = prepare_features(train_frame, cfg)
    X_validation = prepare_features(validation, cfg)
    target = cfg["data"]["target"]
    version = data_version(cfg)
    fingerprint = hashlib.sha256(json.dumps({
        "data_version": version, "data": cfg["data"], "seed": cfg["project"]["random_state"]
    }, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
    tags = {"data_version": version, "data_fingerprint": fingerprint, "git_commit": commit,
            "batch_id": batch_id or uuid.uuid4().hex, "purpose": "validation-selection"}
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params(flatten(cfg))
        mlflow.log_param("data_version", version)
        mlflow.log_param("train_rows", len(train_frame))
        mlflow.log_param("validation_rows", len(validation))
        mlflow.log_dict(cfg, "effective_config.json")
        mlflow.log_artifact(str(project_path(cfg["paths"]["dvc_pointer"])), artifact_path="data_version")
        model = build_model(cfg)
        model.fit(X_train, train_frame[target])
        metrics = evaluate_model(model, X_validation, validation[target], cfg["evaluation"]["decision_threshold"])
        mlflow.log_metrics({f"val_{key}": value for key, value in metrics.items()})
        # Full preprocessing + estimator is the deployable model artifact.
        mlflow.sklearn.log_model(
            model, artifact_path="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(3),
            pip_requirements=["scikit-learn==1.6.1", "pandas==2.2.3", "numpy==1.26.4", "scipy==1.15.2"],
        )
        passed = True
        try:
            enforce_thresholds(metrics, cfg)
        except ValueError:
            passed = False
        mlflow.set_tag("performance_gate", "passed" if passed else "failed")
        model_dir = project_path(cfg["paths"]["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        if passed:
            joblib.dump(model, model_dir / f"{run.info.run_id}.joblib")
        result = {"run_id": run.info.run_id, "run_name": run_name, "metrics": metrics,
                  "data_version": version, "batch_id": tags["batch_id"], "gate_passed": passed}
        print(json.dumps(result, indent=2))
        # Raising inside the run records FAILED and makes CI exit nonzero.
        enforce_thresholds(metrics, cfg)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--run-name", default="configured-training")
    args = parser.parse_args()
    train(load_config(args.config), args.run_name)


if __name__ == "__main__":
    main()
