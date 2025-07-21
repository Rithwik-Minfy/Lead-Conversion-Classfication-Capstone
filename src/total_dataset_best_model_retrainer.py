# model_trainer.py

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, accuracy_score
from xgboost import XGBClassifier
from data_cleaner import clean_data
from feature_engineer import feature_engineering
from data_loader import load_data_from_postgres

def retrain_final_model(
    table_name: str = "lead_scoring_data",
    mlflow_uri: str = "http://localhost:5000",
    experiment_name: str = "LeadConversionFinalTraining",
    model_name: str = "LeadConversionFinalRetrainedModel",
    save_local_path: str = "pkl_joblib_files/best_model_retrained.pkl"
):
    """
    Retrains the best model on the full dataset and logs it to MLflow.

    Parameters:
    - table_name: PostgreSQL table to load data from.
    - mlflow_uri: URI of the MLflow tracking server.
    - experiment_name: MLflow experiment name.
    - model_name: Name to register the retrained model.
    - save_local_path: Path to locally save the retrained model.

    Returns:
    - None
    """
    # Load full raw data
    print("🔄 Loading data from table:", table_name)
    df_raw = load_data_from_postgres(table_name=table_name)

    # Clean and engineer features
    df_cleaned = clean_data(df_raw)
    df_features = feature_engineering(df_cleaned)

    # Define features and target
    X = df_features.drop(columns=["Converted"])
    y = df_features["Converted"]

    # Load saved preprocessor
    preprocessor = joblib.load("pkl_joblib_files/preprocessor.pkl")
    X_processed = preprocessor.transform(X)

    # Define model
    final_model = XGBClassifier(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100
    )

    # Start MLflow logging
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Final_Model_Retraining"):
        mlflow.sklearn.autolog()

        final_model.fit(X_processed, y)

        y_pred = final_model.predict(X_processed)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)

        mlflow.log_metric("final_model_train_accuracy", acc)
        mlflow.log_metric("final_model_train_precision", prec)

        mlflow.sklearn.log_model(final_model, artifact_path="model", registered_model_name=model_name)

        joblib.dump(final_model, save_local_path)

        print(f"✅ Final model retrained and logged to MLflow as '{model_name}'")

if __name__ == "__main__":
    retrain_final_model()