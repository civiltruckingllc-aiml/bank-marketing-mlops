"""Query real MLflow runs and select on validation ROC-AUC, never test metrics."""
import argparse
import json
import mlflow
from src.config import load_config, project_path
from src.tracking import configure_tracking


def compare(cfg, batch_id=None):
    experiment_id = configure_tracking(cfg)
    primary = cfg["evaluation"]["primary_metric"]
    metric = f"metrics.val_{primary}"
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED' and tags.performance_gate = 'passed'",
        order_by=[f"{metric} DESC"],
    )
    if batch_id is not None:
        runs = runs.loc[runs["tags.batch_id"] == batch_id]
    if runs.empty or metric not in runs:
        raise ValueError("No successful runs; run python -m src.run_experiments first")
    # Comparing different data versions or split/feature configurations is invalid.
    if runs["tags.data_fingerprint"].nunique() != 1:
        raise ValueError("Incompatible datasets/splits; pass --batch-id to select one cohort")
    columns = ["run_id", "tags.mlflow.runName", "params.model.type", "metrics.val_roc_auc",
               "metrics.val_average_precision", "metrics.val_accuracy", "metrics.val_f1", "tags.batch_id"]
    table = runs.loc[:, columns].sort_values(metric, ascending=False)
    reports = project_path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "experiment_comparison.csv", index=False)
    best = {"run_id": table.iloc[0]["run_id"], "run_name": table.iloc[0]["tags.mlflow.runName"],
            "primary_metric": f"val_{primary}", "score": float(table.iloc[0][metric]),
            "data_version": runs.iloc[0]["params.data_version"], "runs_compared": len(table),
            "batch_id": table.iloc[0]["tags.batch_id"]}
    (reports / "best_run.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print("Best validation run:", json.dumps(best, indent=2))
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    compare(load_config(args.config), args.batch_id)
