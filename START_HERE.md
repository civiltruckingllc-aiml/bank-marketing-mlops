# Start here — Windows and VS Code

This is a Python project with multiple files. Run it in the VS Code terminal;
do not paste the project into one Jupyter cell.

1. Extract the project ZIP and open the **bank-marketing-mlops** folder in VS Code.
2. Open **Terminal → New Terminal** and use PowerShell.
3. Confirm Python 3.12 is installed with `py -3.12 --version`.
   These pinned libraries target Python 3.12, not Python 3.14.
4. Copy the following commands into the terminal, one line at a time:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip check
```

5. If you downloaded a ZIP rather than cloning GitHub, initialize Git first:

```powershell
git init -b main
```

The ZIP already includes the DVC configuration. Do not run `dvc init` again.

6. Restore and verify the dataset, then run all tests:

```powershell
dvc pull
pytest tests/ -v
```

7. Train and compare all five experiments, then evaluate the selected model:

```powershell
python -m src.run_experiments
python -m src.evaluate
```

8. Open MLflow's experiment dashboard:

```powershell
python -m src.tracking --serve
```

Open **http://127.0.0.1:5000**. Click **bank-marketing-mlops** to inspect
parameters, metrics, and each model artifact. Use **Ctrl+C** to stop the server
before continuing.

9. Test both monitoring conditions:

```powershell
python -m src.monitor_drift --mode baseline
$LASTEXITCODE
python -m src.monitor_drift --mode drifted
$LASTEXITCODE
```

Baseline should exit **0**. Drifted should exit **1**, which is the intended
alert. Open `reports/drift_report.html` by double-clicking it.

10. Your project is published at
    https://github.com/civiltruckingllc-aiml/bank-marketing-mlops . Review its
    Actions tab and README. Submit this public repository URL; a downloaded
    ZIP alone is not the required submission.
