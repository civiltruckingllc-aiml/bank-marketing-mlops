"""Local SQLite tracking with portable artifact paths for downloaded evidence."""
import sqlite3
import mlflow
from src.config import project_path


def configure_tracking(cfg):
    root = project_path(cfg["paths"]["tracking_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    database = root / "mlflow.db"
    artifact_root = root / "runs"
    # MLflow persists absolute file URIs. Rebase only this project's local store.
    # This allows opening the bundled or Actions-downloaded evidence on Windows.
    if database.exists():
        with sqlite3.connect(database) as connection:
            has_tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'"
            ).fetchone()
            if has_tables:
                for experiment_id, in connection.execute("SELECT experiment_id FROM experiments").fetchall():
                    location = (artifact_root / str(experiment_id)).as_uri()
                    connection.execute("UPDATE experiments SET artifact_location=? WHERE experiment_id=?", (location, experiment_id))
                for run_id, experiment_id in connection.execute("SELECT run_uuid, experiment_id FROM runs").fetchall():
                    uri = (artifact_root / str(experiment_id) / run_id / "artifacts").as_uri()
                    connection.execute("UPDATE runs SET artifact_uri=? WHERE run_uuid=?", (uri, run_id))
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    experiment = mlflow.get_experiment_by_name(cfg["project"]["experiment_name"])
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            cfg["project"]["experiment_name"], artifact_location=artifact_root.as_uri()
        )
        # Keep all experiments under <runs>/<experiment_id> for portability.
        with sqlite3.connect(database) as connection:
            connection.execute("UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
                               ((artifact_root / experiment_id).as_uri(), experiment_id))
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def main():
    import argparse
    from src.config import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_tracking(cfg)
    if args.serve:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", mlflow.get_tracking_uri(),
                        "--host", "127.0.0.1", "--port", "5000"], check=True)
    else:
        print("Tracking store ready:", mlflow.get_tracking_uri())


if __name__ == "__main__":
    main()
