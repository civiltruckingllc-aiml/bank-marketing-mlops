"""Feature-wise Evidently monitoring; exit 1 on a configurable drift alert."""
import argparse
import json
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from src.config import load_config, project_path
from src.data import features, load_dataset, split_dataset
from src.preprocessing import prepare_features


def make_current_data(frame, cfg, mode="drifted"):
    """Use only the disjoint holdout pool as a simulated unlabeled batch."""
    settings = cfg["monitoring"]
    current = frame.sample(n=min(settings["current_size"], len(frame)),
                           random_state=settings["random_state"]).copy(deep=True)
    if mode == "baseline":
        return current
    if mode != "drifted":
        raise ValueError("Mode must be baseline or drifted")
    shifts = settings["shifts"]
    current["age"] = (current["age"] + shifts["age_add"]).clip(upper=100)
    current["balance"] = current["balance"] + shifts["balance_add"]
    current["campaign"] = current["campaign"] + shifts["campaign_add"]
    count = int(len(current) * shifts["contact_fraction"])
    selected = current.sample(n=count, random_state=settings["random_state"]).index
    current.loc[selected, "contact"] = shifts["contact_value"]
    return current


def run_monitoring(cfg, mode="drifted", current_path=None):
    settings = cfg["monitoring"]
    threshold = settings["drift_share_threshold"]
    if not 0 <= threshold <= 1:
        raise ValueError("drift_share_threshold must be in [0, 1]")
    train, _, holdout = split_dataset(load_dataset(cfg), cfg)
    reference = prepare_features(train, cfg)
    if current_path:
        current = prepare_features(pd.read_csv(project_path(current_path)), cfg)
        label = "external"
    else:
        current = prepare_features(make_current_data(holdout, cfg, mode), cfg)
        label = mode
    # Detect missingness shifts separately in production. Here drop NA only
    # within each statistical test (Evidently behavior), retaining true inputs.
    mapping = ColumnMapping(
        target=None, prediction=None,
        numerical_features=cfg["data"]["numeric_features"],
        categorical_features=cfg["data"]["categorical_features"],
    )
    report = Report(metrics=[DataDriftPreset(
        # One feature increments drift_share by 1/N. This midpoint implements
        # strictly greater than the CLI threshold even when k/N == threshold.
        drift_share=min(1.01, (int(threshold * len(features(cfg))) + 0.5) / len(features(cfg))),
        num_stattest=settings["numeric_method"],
        cat_stattest=settings["categorical_method"],
        num_stattest_threshold=settings["statistical_threshold"],
        cat_stattest_threshold=settings["statistical_threshold"],
    )])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)
    result = report.as_dict()
    drift_table = next(metric["result"] for metric in result["metrics"] if metric["metric"] == "DataDriftTable")
    per_feature = {}
    for column in features(cfg):
        item = drift_table["drift_by_columns"][column]
        per_feature[column] = {
            "drift_detected": bool(item["drift_detected"]),
            "score": float(item["drift_score"]),
            "method": item["stattest_name"],
            "reference_missing_share": float(reference[column].isna().mean()),
            "current_missing_share": float(current[column].isna().mean()),
        }
    drifted = [column for column, info in per_feature.items() if info["drift_detected"]]
    share = len(drifted) / len(features(cfg))
    alert = share > threshold
    summary = {"mode": label, "reference_rows": len(reference), "current_rows": len(current),
               "features_checked": len(features(cfg)), "drifted_features": drifted,
               "drift_share": share, "threshold": threshold, "alert": alert,
               "rule": "alert if drift_share > threshold", "per_feature": per_feature,
               "caution": "Input drift is not proof of performance decline; simulated labels are not valid after feature edits."}
    reports = project_path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    stem = "drift" if label == "drifted" else f"drift_{label}"
    report.save_html(str(reports / f"{stem}_report.html"))
    report.save_json(str(reports / f"{stem}_report.json"))
    (reports / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Reference rows: {len(reference):,}; current rows: {len(current):,}")
    for column, info in per_feature.items():
        print(f"{column:12s} drift={str(info['drift_detected']):5s} score={info['score']:.4f} ({info['method']})")
    print(f"Drift share: {len(drifted)}/{len(features(cfg))} = {share:.2%}; threshold: {threshold:.2%}")
    print("Drifted features:", ", ".join(drifted) or "none")
    print("ALERT" if alert else "OK", f"HTML report: {reports / f'{stem}_report.html'}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--mode", choices=["baseline", "drifted"], default="drifted")
    parser.add_argument("--current-data", help="Optional real batch CSV with the configured input features")
    args = parser.parse_args()
    summary = run_monitoring(load_config(args.config), args.mode, args.current_data)
    raise SystemExit(1 if summary["alert"] else 0)


if __name__ == "__main__":
    main()
