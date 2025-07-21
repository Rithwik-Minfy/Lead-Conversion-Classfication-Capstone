import pandas as pd
import os
import mlflow
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import json
from pathlib import Path
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from data_loader import load_data_from_postgres
from sklearn.model_selection import train_test_split

def split_data(df, target_column="Converted"):
    """
    Splits the input DataFrame into train, validation, and test sets (X and y).
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # First split: Train vs Temp (Val+Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # Second split: Val vs Test (from temp)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def log_evidently_report(reference_data, current_data, dataset_name="train_vs_test"):
    
    #  Align columns: use only the intersection to avoid partial-column errors
    common_cols = set(reference_data.columns).intersection(current_data.columns)
    if not common_cols:
        print(f"⚠️ No common columns between reference and {dataset_name}; skipping Evidently report.")
        return
    ref = reference_data[sorted(common_cols)]
    cur = current_data[sorted(common_cols)]

    #  Run the Evidently report (drift + summary)
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    #  Ensure local save directory exists
    save_dir = Path.cwd() / "evidently_reports"
    save_dir.mkdir(parents=True, exist_ok=True)

    #  Save HTML and JSON
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    html_path = save_dir / f"evidently_{dataset_name}_{ts}.html"
    json_path = save_dir / f"evidently_{dataset_name}_{ts}.json"
    result.save_html(str(html_path))
    with open(json_path, "w", encoding="utf-8") as fp:
        fp.write(result.json())

    #  Log artifacts to MLflow
    mlflow.log_artifact(str(html_path), artifact_path="evidently")
    mlflow.log_artifact(str(json_path), artifact_path="evidently")
    print(f"📄 Logged HTML: {html_path.name}")
    print(f"🗄️  Logged JSON: {json_path.name}")

    #  Load JSON and extract metrics list
    with open(json_path, "r", encoding="utf-8") as fp:
        report_json = json.load(fp)
    metrics_list = report_json.get("metrics", [])

    #  Overall drifted columns metrics
    drift_entry = next((m for m in metrics_list if m.get("metric_id", "").startswith("DriftedColumnsCount")), None)
    if drift_entry:
        count = drift_entry["value"]["count"]
        share = drift_entry["value"]["share"]
        mlflow.log_metric("drifted_columns_count", float(count))
        mlflow.log_metric("drifted_columns_share", float(share))
        print(f"🔢 drifted_columns_count = {count}")
        print(f"🔢 drifted_columns_share = {share}")
    else:
        print("⚠️ No DriftedColumnsCount entry found.")

    #  Row and column counts
    rowcount = next((m["value"] for m in metrics_list if m.get("metric_id") == "RowCount()"), None)
    colcount = next((m["value"] for m in metrics_list if m.get("metric_id") == "ColumnCount()"), None)
    if rowcount is not None:
        mlflow.log_metric("dataset_row_count", float(rowcount))
        print(f"🔢 dataset_row_count = {rowcount}")
    if colcount is not None:
        mlflow.log_metric("dataset_column_count", float(colcount))
        print(f"🔢 dataset_column_count = {colcount}")

    #  Per-feature value drift metrics
    for m in metrics_list:
        mid = m.get("metric_id", "")
        if mid.startswith("ValueDrift(column="):
            # extract column name
            col = mid.split("=")[1].rstrip(")")
            val = m.get("value")
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"drift_{col}", float(val))
                print(f"🔢 drift_{col} = {val}")
    
    print("✅ All requested drift & dataset metrics logged to MLflow.")


def run_drift():

    EXPERIMENT_NAME = "Lead Conversion Prediction Evidently"
    client = MlflowClient()

    # ─── 1️⃣ Ensure the MLflow experiment exists and is active ───
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = client.create_experiment(EXPERIMENT_NAME)
        print(f"✅ Created new experiment '{EXPERIMENT_NAME}' (ID={exp_id})")
    elif exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
        print(f"🔄 Restored deleted experiment '{EXPERIMENT_NAME}' (ID={exp.experiment_id})")
    else:
        print(f"ℹ️ Using existing experiment '{EXPERIMENT_NAME}' (ID={exp.experiment_id})")

    mlflow.set_experiment(EXPERIMENT_NAME)

    # ─── 2️⃣ Start your MLflow run ───
    with mlflow.start_run(run_name="Lead Conversion Prediction Drift Detection"):
        # Load and split
        df = load_data_from_postgres(table_name='lead_scoring_data')
        Xtr, Xv, Xt, ytr, yv, yt = split_data(df)

        # Keep raw for Evidently
        df_train = Xtr.copy()
        df_test  = Xt.copy()

        df_train = df_train.dropna(axis=1, how='all')
        df_test = df_test.dropna(axis=1, how='all')

        # Log Evidently reports
        log_evidently_report(df_train, df_test, dataset_name="train_vs_test")

if __name__=="__main__":
    run_drift()