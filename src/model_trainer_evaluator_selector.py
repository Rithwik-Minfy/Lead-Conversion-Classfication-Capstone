import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)


def train_evaluate_and_select_model_accuracy(X_train, y_train, X_test, y_test,
                                    model_name="Lead_Classfication_Best_Model_Accuracy",
                                    shap_dir="shap_outputs",
                                    model_dir="pkl_joblib_files"):
    # Ensure output folders exist
    shap_dir = os.path.abspath(shap_dir)
    model_dir = os.path.abspath(model_dir)
    os.makedirs(shap_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("lead_conversion_experiment_Accuracy")

    models = {
        'logistic_regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'xgboost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'params': {
                'n_estimators': [100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        },
        'lightgbm': {
            'model': LGBMClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [-1, 5]
            }
        }
    }

    best_model = None
    best_score = 0
    best_name = None
    best_run_id = None

    for name, config in models.items():
        with mlflow.start_run(run_name=f"{name}_run") as run:
            print(f"\n🔍 Training {name}...")

            grid = GridSearchCV(
                config['model'], config['params'],
                cv=5, scoring='f1',
                n_jobs=-1, verbose=0
            )
            grid.fit(X_train, y_train)

            best_estimator = grid.best_estimator_
            f1_cv = grid.best_score_

            y_pred = best_estimator.predict(X_test)
            y_proba = best_estimator.predict_proba(X_test)[:, 1]

            test_acc = accuracy_score(y_test, y_pred)
            test_prec = precision_score(y_test, y_pred)
            test_rec = recall_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            test_auc = roc_auc_score(y_test, y_proba)

            # Log to MLflow
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("f1_cv", f1_cv)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_precision", test_prec)
            mlflow.log_metric("test_recall", test_rec)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("test_auc_roc", test_auc)
            mlflow.sklearn.log_model(best_estimator, artifact_path="model")

            print(f"✅ Completed {name}: Accuracy = {test_acc:.4f}, AUC = {test_auc:.4f}")

            # SHAP explanation
            try:
                print("🔍 Calculating SHAP values...")
                if name in ['random_forest', 'xgboost', 'lightgbm']:
                    # Use TreeExplainer
                    explainer = shap.TreeExplainer(best_estimator, X_train, feature_names=X_train.columns)
                    shap_values = explainer(X_test)

                    # Handle single array or list
                    shap_summary = shap_values.values if hasattr(shap_values, "values") else shap_values

                    # SHAP expects NumPy arrays
                    shap.summary_plot(
                        shap_summary,
                        X_test.values,
                        feature_names=X_test.columns.tolist(),
                        show=False
                    )
                    shap_path = os.path.join(shap_dir, f"{name}_shap_summary.png")
                    plt.tight_layout()
                    plt.savefig(shap_path)
                    plt.close()
                    mlflow.log_artifact(shap_path)
                    print(f"📊 SHAP plot saved for {name}: {shap_path}")
                elif name == 'logistic_regression':
                    explainer = shap.LinearExplainer(best_estimator, X_train, feature_perturbation="interventional")
                    shap_values = explainer.shap_values(X_test)
                else:
                    explainer = shap.Explainer(best_estimator, X_train)
                    shap_values = explainer(X_test)

                shap.summary_plot(
                    shap_values if isinstance(shap_values, np.ndarray) else shap_values.values,
                    X_test.values if hasattr(X_test, "values") else X_test,
                    feature_names=X_test.columns.tolist(),
                    show=False
                )
                shap_path = os.path.join(shap_dir, f"{name}_shap_summary.png")
                plt.tight_layout()
                plt.savefig(shap_path)
                plt.close()
                mlflow.log_artifact(shap_path)
                print(f"📊 SHAP plot saved: {shap_path}")
            except Exception as e:
                print(f"⚠️ SHAP failed for {name}: {e}")

            if test_acc > best_score:
                best_score = test_acc
                best_model = best_estimator
                best_name = name
                best_run_id = run.info.run_id

    print(f"\n🎯 Best Model: {best_name} (Accuracy = {best_score:.4f})")

    # Save best model to pkl_joblib_files
    best_model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"📦 Best model saved to: {best_model_path}")

    # Register best model in MLflow
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=result.version, stage="Staging", archive_existing_versions=True
    )
    print(f"🟡 Model version {result.version} moved to Staging")

    client.transition_model_version_stage(
        name=model_name, version=result.version, stage="Production", archive_existing_versions=True
    )
    print(f"🟢 Model version {result.version} moved to Production")

    # Load MLflow model from registry
    mlflow_best_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")

    # Classification Report
    y_best_pred = mlflow_best_model.predict(X_test)
    print("\n🧾 Classification Report for MLflow best model on Accuracy:\n", classification_report(y_test, y_best_pred))

    return best_model, y_best_pred

def train_evaluate_and_select_model_precision(X_train, y_train, X_test, y_test, save_dir="pkl_joblib_files", model_name="Lead_Classfication_Best_Model_Precision"):
    os.makedirs(save_dir, exist_ok=True)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Lead_Conversion_Classification_Precision")

    models = {
        'logistic_regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'xgboost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        },
        'lightgbm': {
            'model': LGBMClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [-1, 5, 10],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        }
    }

    best_model = None
    best_precision = 0
    best_name = None
    best_run_id = None

    for name, config in models.items():
        with mlflow.start_run(run_name=f"{name}_run") as run:
            print(f"🔍 Tuning {name}...")

            grid = GridSearchCV(
                config['model'], config['params'],
                cv=5, scoring='precision', n_jobs=-1
            )
            grid.fit(X_train, y_train)

            # Predictions on test set
            y_pred = grid.best_estimator_.predict(X_test)
            class1_precision = precision_score(y_test, y_pred, pos_label=1)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("test_precision_class1", class1_precision)
            mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")

            if class1_precision > best_precision:
                best_model = grid.best_estimator_
                best_precision = class1_precision
                best_name = name
                best_run_id = run.info.run_id

    # Final evaluation
    y_final_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_final_pred)
    f1 = f1_score(y_test, y_final_pred)
    precision = precision_score(y_test, y_final_pred, pos_label=1)

    print(f"✅ Best Model: {best_name}")
    print(f"🔢 Accuracy: {acc:.4f}")
    print(f"🎯 Precision (class 1): {precision:.4f}")
    print(f"📊 F1 Score: {f1:.4f}")
    print("\n📝 Classification Report:\n", classification_report(y_test, y_final_pred))

    # Save model locally
    model_path = os.path.join(save_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"📁 Model saved to: {model_path}")

    # Register best model to MLflow Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = mlflow.tracking.MlflowClient()

    # Transition to Staging and then Production
    client.transition_model_version_stage(
        name=model_name, version=result.version, stage="Staging", archive_existing_versions=True
    )
    print(f"🚀 Model version {result.version} moved to Staging")

    client.transition_model_version_stage(
        name=model_name, version=result.version, stage="Production", archive_existing_versions=True
    )
    print(f"🏁 Model version {result.version} moved to Production")

    # Load MLflow model from registry
    mlflow_best_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")

    # Classification Report
    y_best_pred = mlflow_best_model.predict(X_test)
    print("\n🧾 Classification Report for MLflow best model on Precision:\n", classification_report(y_test, y_best_pred))

    return best_model, y_final_pred

# Example usage
if __name__ == "__main__":
    best_model_accuracy, y_pred_test_accuracy = train_evaluate_and_select_model_accuracy(X_train, y_train, X_test, y_test)
    best_model_precison, y_pred_test_precision = train_evaluate_and_select_model_precision(X_train, y_train, X_test, y_test)
