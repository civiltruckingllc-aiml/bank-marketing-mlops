# Bank Marketing MLOps Pipeline

[![MLOps pipeline](https://github.com/civiltruckingllc-aiml/bank-marketing-mlops/actions/workflows/mlops.yml/badge.svg)](https://github.com/civiltruckingllc-aiml/bank-marketing-mlops/actions/workflows/mlops.yml)

A reproducible classification pipeline demonstrating Git, DVC, MLflow, pytest,
GitHub Actions, and Evidently. Predict whether a bank customer will subscribe
to a term deposit using information available before a marketing call.

The focus is reliable ML operations. This is an educational production
foundation, not a deployed banking decision system.

## Dataset

Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing),
by Sérgio Moro, Paulo Cortez, and Paulo Rita.
Dataset DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306).
The UCI listing licenses the dataset under CC BY 4.0.

The DVC-tracked download is `bank-full.csv`, retrieved from a fixed-commit
[Packt publisher mirror](https://raw.githubusercontent.com/PacktWorkshops/The-Machine-Learning-Workshop/7be1026fb864828ab302074b1f12ab4c25ad73c7/Chapter06/bank-full.csv).
Its 4,610,348 bytes were verified byte-for-byte identical to the file inside
UCI's original archive. Content MD5: `e6b0ca77f3f200ec5428e04dd104da53`.
Raw data is excluded from Git.

| Property | Value |
| --- | --- |
| Rows | 45,211 |
| Model input features | 15: 6 numeric and 9 categorical |
| Target | `y`, mapped from no/yes to 0/1 |
| Numeric | age, balance, day, campaign, pdays, previous |
| Categorical | job, marital, education, default, housing, loan, contact, month, poutcome |
| Missing values | UCI `unknown` values become missing; additionally, 2% missingness is simulated independently for age and balance using seed 42 |
| Excluded | duration, because it is only known after the call |

Data simulation does not alter labels. Missingness is deterministic and its
configuration is logged with the dataset MD5. `pdays = -1` means the customer
was not previously contacted and is preserved as a valid domain value.

## Structure

| Path | Purpose |
| --- | --- |
| `src/config.py` | YAML loading and path resolution |
| `src/data.py` | Version verification, source CSV loading, reproducible splits |
| `src/preprocessing.py` | Input validation, imputation, encoding, scaling |
| `src/modeling.py` | Sklearn preprocessing/estimator pipeline |
| `src/train.py` | Training, MLflow logging, performance gates |
| `src/evaluate.py` | Metric calculation and selected-model holdout evaluation |
| `src/tracking.py` | SQLite tracking and portable artifact paths |
| `src/run_experiments.py` | Five configured experiments |
| `compare_experiments.py` | `mlflow.search_runs()` comparison and best-run selection |
| `src/monitor_drift.py` | Baseline, simulated drift, or external-batch monitoring |
| `configs/` | Training, experiments, and monitoring settings |
| `tests/` | Unit, real-data, and model-validation tests |
| `.github/workflows/mlops.yml` | Test → train workflow |
| `data/raw/bank-full.csv.dvc` | Versioned source dataset pointer |
| `reports/` | Small committed evidence summaries; generated HTML excluded from Git |
| `artifacts/` | Ignored MLflow database, run artifacts, and model files |
| `MONITORING.md` | Measured drift results, interpretation, and recommended action |
| `START_HERE.md` | Step-by-step Windows/VS Code instructions |

## Setup

Use **Python 3.12** and Git. Python 3.14 is not supported by these pins.
Clone this repository, change into its root, then create a virtual environment.

```bash
git clone https://github.com/civiltruckingllc-aiml/bank-marketing-mlops.git
cd bank-marketing-mlops
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip check
dvc pull
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip check
dvc pull
```

For ZIP downloads, run `git init -b main` before using DVC. Do not initialize
DVC again: `.dvc/` and the pointer are already included.

The public dataset is about 4.6 MB; an internet connection is needed for the
initial dependency installation and data download. No API keys, paid services,
or Kaggle account are needed.

`requirements.txt` pins the project dependencies and uses `constraints.txt`
to constrain their transitive dependencies to the versions from the verified
Python 3.12 environment. Platform-specific dependencies remain subject to
their package environment markers.

## DVC data restoration and remote

DVC is initialized and the default **local** remote is `.dvc/local-remote/`.
This remote, raw CSV, and cache are ignored by Git. The dataset is tracked
with `dvc import-url --no-download`: a deferred import stores the source's
ETag and immutable commit URL in the pointer. A fresh `dvc pull` downloads
from that source without needing another person's local remote. Training
also independently checks the actual content against the MD5 pinned in YAML.
This source-backed DVC pointer intentionally has no cached-output MD5.

Run `dvc pull` to restore the dataset and inspect its status:

```bash
dvc status
```

The README's verification section records whether clean-clone restoration was
actually tested. A local remote alone is not a shared remote: the original
public URL enables source restoration. For longer-lived production operation,
replace the deferred import with a cached `dvc add` output, configure durable
shared object storage, and run `dvc push`, protecting against source removal.
Do not run `dvc add` over the deferred import without first removing its stage.

If deliberately accepting a new upstream version, use `dvc update
data/raw/bank-full.csv.dvc --no-download`, inspect the result, update the
content MD5 in YAML only after verifying the new data, rerun validation and
experiments, and commit the updated pointer/config. To select a different
fixed commit URL, recreate the deferred import explicitly. Do not silently
update the source inside CI.

## Run tests

```bash
pytest tests/ -v
```

Tests load the actual DVC-restored dataset and fail if it is absent. They
check numeric and categorical imputation, one-hot behavior, unknown categories,
non-mutating inputs, invalid inputs, train-only imputer statistics, schema,
target values, numeric ranges, split disjointness, prediction shape/types,
and real validation performance. Failure-gate tests also reject bad or NaN
metrics.

## Training and experiment tracking

One configured training run:

```bash
python -m src.train --config configs/train.yaml
```

Run all five experiments and select the best:

```bash
python -m src.run_experiments
python compare_experiments.py
python -m src.evaluate
```

The experiments compare two logistic-regression regularization settings and
three random-forest depths/tree counts. Every experiment uses the same
stratified 60% train / 20% validation / 20% test allocation and random seed.
The preprocessing pipeline is fitted only on training rows.

MLflow logs every configuration value, model hyperparameters, source MD5,
data/split fingerprint, Git commit, effective configuration, validation ROC-AUC,
average precision, accuracy, F1, precision, recall, and the complete trained
pipeline through `mlflow.sklearn.log_model()` with a signature and input sample.

The primary selection metric is **validation ROC-AUC**. Average precision is
also gated because the positive class is relatively rare; high accuracy alone
would not establish useful performance. Gates are configurable in YAML:
ROC-AUC ≥ 0.70 and average precision ≥ 0.20. Training exits nonzero and marks
its MLflow run FAILED when a gate fails. Models passing the gate are saved by
run ID, preventing accidental overwriting by a later run.

`compare_experiments.py` calls `mlflow.search_runs()`, considers only finished
runs that passed the gate, rejects incompatible data/split cohorts, ranks by
validation ROC-AUC, and writes `reports/experiment_comparison.csv` and
`reports/best_run.json`. Use `--batch-id <id>` to compare one experiment batch
if several incompatible datasets were tried.

`src.evaluate` restores the selected MLflow model and its effective training
configuration, then evaluates the reserved holdout exactly as configured.
Test metrics are not used for model selection. Repeated evaluation on this
fixed educational holdout is possible; use a new, independently collected
holdout before making real deployment decisions.

Open the tracking dashboard:

```bash
python -m src.tracking --serve
```

Visit http://127.0.0.1:5000 and choose **bank-marketing-mlops**. This command
rebases local artifact locations when moving a downloaded evidence folder to
a different machine. Stop with Ctrl+C.

## Drift monitoring

```bash
python -m src.monitor_drift --mode baseline
python -m src.monitor_drift --mode drifted
```

The reference is the training split. Simulated production rows are sampled
from the disjoint test split and labels are excluded from drift analysis.
All 15 input features are checked with explicit numeric/categorical mapping.
Evidently 0.6.7 is intentionally pinned because the code uses its Report API;
do not upgrade Evidently alone without migrating and testing its API.

Numeric features use normalized Wasserstein distance; categorical features
use Jensen–Shannon distance, both with a configured distance threshold of 0.10.
These distances are not p-values. The global alert triggers only when the
share of drifting features is **greater than** the configured 0.20 threshold.

The shifted simulation increases age by 20 years (capped at 100), balance by
10,000, campaign contacts by 8, and sets 90% of contact channels to telephone.
Baseline makes no edits. The two commands save interactive HTML reports,
full Evidently JSON, and concise machine-readable summaries in `reports/`.
**The shifted command exiting 1 is the intended alert, not a broken script.**
See [MONITORING.md](MONITORING.md) for measured features and conclusions.

For a real production batch with the same raw feature names:

```bash
python -m src.monitor_drift --current-data path/to/batch.csv
```

Missingness shares are included in the summary because distribution-distance
tests alone may not detect a rise in missing data. A production system should
also have dedicated missingness and schema alerts, scheduled execution, and
labeled performance monitoring.

## CI/CD and reviewing evidence

Pushes to main and pull requests targeting main trigger two jobs:

1. **test** installs dependencies, restores DVC data, and runs the full suite.
2. **train** depends on test success, restores data independently, trains all
   five experiments, enforces validation and holdout performance thresholds,
   checks baseline monitoring, and verifies that the drifted simulation returns
   exactly exit code 1. Other errors are not swallowed.

The workflow also supports a manual run. It uploads pytest XML and an
**mlops-training-evidence** artifact containing the MLflow database, complete
run/model artifacts, comparison tables, and drift HTML reports. They are not
committed as data/model files in Git. Download and extract the training artifact
into the repository root so its `artifacts/` and `reports/` folders are restored,
then use `python -m src.tracking --serve` to inspect all five runs. Actions
artifacts expire after 30 days; download or regenerate them when needed.

This is an automated validation and training pipeline. It does not deploy a
public prediction service or automatically retrain when drift is detected.

## Final submission

Public repository: **https://github.com/civiltruckingllc-aiml/bank-marketing-mlops**

Submit that repository URL. The local project archive is an optional download;
the public repository is the requested deliverable. Review the live status
badge above or the [Actions history](https://github.com/civiltruckingllc-aiml/bank-marketing-mlops/actions)
for hosted verification. Each successful training job makes its five MLflow
runs, model artifacts, and drift reports available in the workflow artifacts.

Verified successful hosted run:
[MLOps pipeline #1](https://github.com/civiltruckingllc-aiml/bank-marketing-mlops/actions/runs/38840303736).
Both **test** and **train** passed, including all 22 tests, five training
experiments, holdout gates, and both monitoring checks. Documentation-only
publication updates use `[skip ci]`; the linked run tested the pipeline code.

## Verification evidence

| Local check | Measured result |
| --- | --- |
| Fresh Git clone → `dvc pull` | Passed without an existing cache or remote directory |
| Full pytest suite | 22 passed |
| MLflow experiments | 5 finished runs with complete model artifacts |
| Selected validation ROC-AUC | 0.7776 |
| Selected holdout ROC-AUC | 0.7846 |
| Selected holdout average precision | 0.4342 |
| Baseline monitoring | 0/15 drifting features; exit 0 |
| Shifted monitoring | 4/15 drifting features (26.67%); exit 1 |
| Public GitHub repository / Actions run | Published; both jobs passed in linked hosted run #1 |

Measured local results are recorded in `reports/verification.json`,
`reports/experiment_comparison.csv`, `reports/holdout_metrics.json`, and
`MONITORING.md`. Local verification and GitHub-hosted Actions results are
separate evidence; a local pass does not prove a green GitHub run.

## Limits and next steps

The random stratified split evaluates a reproducible historical example; it
does not prove generalization to future campaigns. Before production, use
time-aware evaluation, inspect subgroup errors and calibration, determine
business thresholds/costs, address privacy and fairness, secure model storage,
and add an inference service, authorization, scheduling, alert routing, and
rollback. Drift alone is an investigation signal, not proof that retraining
will help. No financial or marketing decisions should be made from this demo.
