import boto3
import pandas as pd
import numpy as np
import time

def load_from_redshift(
    region='ap-south-1',
    workgroup_name='lead-workgroup',
    database_name='dev',
    secret_arn='arn:aws:secretsmanager:ap-south-1:258399971007:secret:redshift-serverless-secret-sz4Cm0',
    sql='SELECT * FROM lead_scoring LIMIT 100'
) -> pd.DataFrame:
    """
    Executes a SQL query on Redshift Serverless and returns the result as a pandas DataFrame.
    
    Parameters:
    - region (str): AWS region
    - workgroup_name (str): Redshift Serverless workgroup name
    - database_name (str): Redshift database name
    - secret_arn (str): ARN of AWS Secrets Manager secret with Redshift credentials
    - sql (str): SQL query to execute

    Returns:
    - pd.DataFrame: Query result
    """
    try:
        client = boto3.client('redshift-data', region_name=region)

        # Execute query
        response = client.execute_statement(
            WorkgroupName=workgroup_name,
            Database=database_name,
            SecretArn=secret_arn,
            Sql=sql
        )

        statement_id = response['Id']

        # Wait for query to complete
        while True:
            desc = client.describe_statement(Id=statement_id)
            status = desc['Status']
            if status in ['FINISHED', 'FAILED', 'ABORTED']:
                break
            time.sleep(1)

        if status != 'FINISHED':
            raise Exception(f"Query failed with status: {status}")

        # Retrieve results
        result = client.get_statement_result(Id=statement_id)
        columns = [col['name'] for col in result['ColumnMetadata']]
        rows = result['Records']

        data = [[list(col.values())[0] if col else None for col in row] for row in rows]
        df = pd.DataFrame(data, columns=columns)

        print("✅ Data loaded successfully from Redshift with shape:", df.shape)

        # column mapping
        column_name_map = {
            "prospect_id": "Prospect ID",
            "lead_number": "Lead Number",
            "lead_origin": "Lead Origin",
            "lead_source": "Lead Source",
            "do_not_email": "Do Not Email",
            "do_not_call": "Do Not Call",
            "converted": "Converted",
            "totalvisits": "TotalVisits",
            "total_time_spent_on_website": "Total Time Spent on Website",
            "page_views_per_visit": "Page Views Per Visit",
            "last_activity": "Last Activity",
            "country": "Country",
            "specialization": "Specialization",
            "how_did_you_hear_about_x_education": "How did you hear about X Education",
            "what_is_your_current_occupation": "What is your current occupation",
            "what_matters_most_to_you_in_choosing_a_course": "What matters most to you in choosing a course",
            "search": "Search",
            "magazine": "Magazine",
            "newspaper_articlearticle": "Newspaper Article",
            "x_education_forums": "X Education Forums",
            "newspaper": "Newspaper",
            "digital_advertisement": "Digital Advertisement",
            "through_recommendations": "Through Recommendations",
            "receive_more_updates_about_our_courses": "Receive More Updates About Our Courses",
            "tags": "Tags",
            "lead_quality": "Lead Quality",
            "update_me_on_supply_chain_content": "Update me on Supply Chain Content",
            "get_updates_on_dm_content": "Get updates on DM Content",
            "lead_profile": "Lead Profile",
            "city": "City",
            "asymmetrique_activity_index": "Asymmetrique Activity Index",
            "asymmetrique_profile_index": "Asymmetrique Profile Index",
            "asymmetrique_activity_score": "Asymmetrique Activity Score",
            "asymmetrique_profile_score": "Asymmetrique Profile Score",
            "i_agree_to_pay_the_amount_through_cheque": "I agree to pay the amount through cheque",
            "a_free_copy_of_mastering_the_interview": "A free copy of Mastering The Interview",
            "last_notable_activity": "Last Notable Activity"
        }
    
        df.rename(columns=column_name_map, inplace=True)
    
        # Optional: Log missing expected columns
        expected_columns = list(column_name_map.values())
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            print("⚠️ Missing expected columns after renaming:", missing_cols)
    
        df.replace('', np.nan, inplace=True)
            
        return df

    except Exception as e:
        print(f"❌ Failed to load data from Redshift: {e}")
        return pd.DataFrame()  # return empty DataFrame on error
    
# Applying the techniques observed in EDA
import pandas as pd
import numpy as np

def clean_data(df):

    # Prospect ID and Lead Number adds no additional information so drop these columns
    df = df.drop(columns=['Prospect ID', 'Lead Number'], errors='ignore')

    # Removing as these contains constant value "False"
    df = df.drop(columns=["Magazine", "Receive More Updates About Our Courses", "Update me on Supply Chain Content", "Get updates on DM Content","I agree to pay the amount through cheque"], errors='ignore')
    
    # Check for duplicate records
    print("\nDuplicate Records:", df.duplicated().sum())

    # if duplicate rows found then drop
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Filling the missing null values and label encoding "Asymmetrique Profile Index" and "Asymmetrique Activity Index"
    # and dropping unneccesary columns
    drop_cols = [
        'Newspaper', 'Newspaper Article',
        'X Education Forums', 'Search', 'Through Recommendations'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Standardize labels
    df['Lead Source'] = df['Lead Source'].str.strip().str.lower()
    df['Lead Source'] = df['Lead Source'].replace({'google': 'Google'})
    df['Lead Source'] = df['Lead Source'].fillna('Missing')

    # Replace 'Select' with 'Missing'
    select_to_missing_cols = [
        'Specialization', 'Lead Profile', 'City', 'How did you hear about X Education'
    ]
    for col in select_to_missing_cols:
        df[col] = df[col].replace('Select', 'Missing')
        df[col] = df[col].fillna('Missing')

    # Fill remaining high-null categorical columns
    df['Country'] = df['Country'].fillna('India')
    df['What is your current occupation'] = df['What is your current occupation'].fillna('Unemployed')
    df['What matters most to you in choosing a course'] = df['What matters most to you in choosing a course'].fillna('Better Career Prospects')
    df['Tags'] = df['Tags'].fillna('Missing')
    df['Lead Quality'] = df['Lead Quality'].fillna('Missing')

    # Ordinal encoding for Asymmetrique Index columns
    ordinal_map = {"01.High": 3, "02.Medium": 2, "03.Low": 1}
    df["Asymmetrique Profile Index"] = df["Asymmetrique Profile Index"].map(ordinal_map)
    df["Asymmetrique Activity Index"] = df["Asymmetrique Activity Index"].map(ordinal_map)

    # Fill with median
    for col in [
        'Asymmetrique Profile Index', 'Asymmetrique Activity Index',
        'Asymmetrique Profile Score', 'Asymmetrique Activity Score',
        'TotalVisits', 'Page Views Per Visit'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Done
    print("Data cleaned and ready. Remaining nulls:")
    print(df.isnull().sum()[df.isnull().sum() > 0])


    # Mapping the labels of the columns with similar meaning and if more labels to convert to less labels
    lead_source_map = {
        'google': 'Google',
        'google ads': 'Google',
        'organic search': 'Organic Search',
        'olark chat': 'Olark Chat',
        'direct traffic': 'Direct Traffic',
        'reference': 'Reference',
        'welingak website': 'Welingak',
        'facebook': 'Social Media',
        'bing': 'Other',
        'click2call': 'Other',
        'press_release': 'Other',
        'social media': 'Social Media',
        'live chat': 'Olark Chat',
        'youtubechannel': 'Other',
        'testone': 'Other',
        'pay per click ads': 'Other',
        'welearnblog_home': 'Other',
        'welearn': 'Other',
        'blog': 'Other',
        'nc_edm': 'Other'
    }

    specialization_map = {
        'finance management': 'Finance',
        'banking, investment and insurance': 'Finance',
        'human resource management': 'HR',
        'marketing management': 'Marketing',
        'operations management': 'Operations',
        'it projects management': 'IT',
        'business administration': 'Business',
        'supply chain management': 'Operations',
        'e-commerce': 'Business',
        'retail management': 'Marketing',
        'media and advertising': 'Marketing',
        'travel and tourism': 'Other',
        'international business': 'Business',
        'healthcare management': 'Other',
        'hospitality management': 'Other',
        'rural and agribusiness': 'Other',
        'e-business': 'Business',
        'services excellence': 'Other',
        'missing': 'Missing',
        'select': 'Missing'
    }

    tags_map = {
        'will revert after reading the email': 'Reverting',
        'interested in other courses': 'Interested Other',
        'interested  in full time mba': 'Interested Other',
        'graduation in progress': 'Interested Other',
        'not doing further education': 'Not Interested',
        'wrong number given': 'Not Reachable',
        'opp hangup': 'Not Reachable',
        'number not provided': 'Not Reachable',
        'invalid number': 'Not Reachable',
        'still thinking': 'Still Thinking',
        'shall take in the next coming month': 'Still Thinking',
        'want to take admission but has financial problems': 'Still Thinking',
        'lost to eins': 'Lost',
        'lost to others': 'Lost',
        'in touch with eins': 'Lost',
        'diploma holder (not eligible)': 'Not Eligible',
        'university not recognized': 'Not Eligible',
        'recognition issue (dec approval)': 'Not Eligible',
        'already a student': 'Already Student',
        'switched off': 'Not Reachable',
        'busy': 'Not Reachable',
        'ringing': 'Not Reachable',
        'missing': 'Missing',
        '': 'Missing',
    }

    lead_quality_map = {
        'high in relevance': 'High',
        'might be': 'Medium',
        'not sure': 'Medium',
        'low in relevance': 'Low',
        'worst': 'Low',
        'missing': 'Missing'
    }

    lead_profile_map = {
        'potential lead': 'Potential',
        'other leads': 'Other',
        'student of someschool': 'Student',
        'lateral student': 'Student',
        'dual specialization student': 'Student',
        'select': 'Missing',
        'missing': 'Missing'
    }

    heard_map = {
        'online search': 'Online',
        'word of mouth': 'Referral',
        'student of someschool': 'Referral',
        'multiple sources': 'Multiple',
        'advertisements': 'Ads',
        'social media': 'Social',
        'email': 'Direct',
        'sms': 'Direct',
        'other': 'Other',
        'select': 'Missing',
        'missing': 'Missing'
    }

    # Clean cell values only (not column names)
    df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

    # Apply mappings with correct column names
    df['Lead Source'] = df['Lead Source'].replace(lead_source_map)
    df['Specialization'] = df['Specialization'].replace(specialization_map)
    df['Tags'] = df['Tags'].replace(tags_map)
    df['Lead Quality'] = df['Lead Quality'].replace(lead_quality_map)
    df['Lead Profile'] = df['Lead Profile'].replace(lead_profile_map)
    df['How did you hear about X Education'] = df['How did you hear about X Education'].replace(heard_map)
    print("✅ Data cleaning completed")
    print("Shape of the data after cleaning ",df.shape)
    return df

    
import pandas as pd
import numpy as np

def feature_engineering(df):
    """
    Perform feature engineering on the cleaned DataFrame.
    """

    # ----------- Feature 1: Engagement Score -----------
    # Proxy for user engagement based on time and activity on site
    if set(['Total Time Spent on Website', 'Page Views Per Visit', 'TotalVisits']).issubset(df.columns):
        df['Engagement Score'] = (
            df['Total Time Spent on Website'] * 0.4 +
            df['Page Views Per Visit'] * 0.3 +
            df['TotalVisits'] * 0.3
        )

    # ----------- Feature 2: Combined Asymmetrique Score -----------
    if set(['Asymmetrique Activity Score', 'Asymmetrique Profile Score']).issubset(df.columns):
        df['Combined Asymmetrique Score'] = (
            df['Asymmetrique Activity Score'] + df['Asymmetrique Profile Score']
        )
        # Removing columns after taking the combined value as new feature
        df.drop(columns=['Asymmetrique Activity Score', 'Asymmetrique Profile Score'])

    # ----------- Feature 3: Is New Tag -----------
    if 'Tags' in df.columns:
        df['Is New Tag'] = df['Tags'].apply(lambda x: 1 if 'student' in str(x).lower() else 0)

    # ----------- Feature 4: Interaction Level based on Activity -----------
    if 'Last Activity' in df.columns:
        high_activity = ['SMS Sent', 'Email Opened', 'Email Link Clicked']
        df['High Interaction'] = df['Last Activity'].apply(lambda x: 1 if x in high_activity else 0)

    # ----------- Feature 5: Was Previously Interested -----------
    if 'Lead Profile' in df.columns:
        df['Potential Lead'] = df['Lead Profile'].apply(lambda x: 1 if 'potential' in str(x).lower() else 0)

    print("✅ Feature engineering complete. Shape after: ", df.shape)
    return df

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize

# Function to apply Winsorization
def winsorize_columns(df, columns, limits=(0.01, 0.01)):
    for col in columns:
        try:
            df[col] = winsorize(df[col], limits=limits)
        except Exception as e:
            print(f"[!] Could not winsorize column '{col}': {e}")
    return df

# Main preprocessing pipeline function
def preprocess_data(df, save_dir="pkl_joblib_files"):
    df.columns = df.columns.astype(str)
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------
    # 1. Define features
    # ----------------------
    target_col = 'Converted'
    
    numeric_cols = [
        'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit',
        'Asymmetrique Activity Score', 'Asymmetrique Profile Score'
    ]

    ordinal_cols = [
        'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Lead Quality'
    ]
    ordinal_map = [
        [1, 2, 3],                        # Activity Index (Low=1, Medium=2, High=3)
        [1, 2, 3],                        # Profile Index
        ['Low', 'Medium', 'High', 'Missing']  # Lead Quality
    ]

    categorical_cols = [
        'Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call', 'Last Activity',
        'Country', 'Specialization', 'How did you hear about X Education',
        'What is your current occupation', 'What matters most to you in choosing a course',
        'Tags', 'Lead Profile', 'City',
        'A free copy of Mastering The Interview', 'Last Notable Activity'
    ]

    # ----------------------
    # 2. Train-Test Split
    # ----------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ----------------------
    # 3. Winsorize outliers
    # ----------------------
    X_train = winsorize_columns(X_train.copy(), numeric_cols)
    X_test = winsorize_columns(X_test.copy(), numeric_cols)

    # ----------------------
    # 4. Build Transformers
    # ----------------------

    # Numeric Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('yeojohnson', PowerTransformer(method='yeo-johnson')),
        ('scaler', MinMaxScaler())
    ])

    # Ordinal Pipeline
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=ordinal_map))
    ])

    # Categorical Pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine all
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)


    # ----------------------
    # 5. Fit + Transform
    # ----------------------
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # ----------------------
    # 6. Save Preprocessor
    # ----------------------
    joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.pkl"))

    # ----------------------
    # 7. Get Feature Names
    # ----------------------
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + ordinal_cols + list(cat_feature_names)

    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    print("✅ Preprocessing complete. Artifacts saved to:", save_dir)
    print("🧾 X_train shape:", X_train_df.shape)
    print("🧾 X_test shape :", X_test_df.shape)
    print("🎯 y_train dist:\n", y_train.value_counts(normalize=True))

    print(X_train_df,X_test_df,y_train,y_test)
    return X_train_df, X_test_df, y_train, y_test

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
                                    model_name="Lead-Classfication-Best-Model-Accuracy",
                                    shap_dir="shap_outputs",
                                    model_dir="pkl_joblib_files"):
    # Ensure output folders exist
    shap_dir = os.path.abspath(shap_dir)
    model_dir = os.path.abspath(model_dir)
    os.makedirs(shap_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    mlflow.set_tracking_uri("arn:aws:sagemaker:ap-south-1:258399971007:mlflow-tracking-server/mlflow-tracking-server")
    mlflow.set_experiment("Lead-Conversion-Experiment-Accuracy")

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

def train_evaluate_and_select_model_precision(X_train, y_train, X_test, y_test, save_dir="pkl_joblib_files", model_name="Lead-Classfication-Best-Model-Precision"):
    os.makedirs(save_dir, exist_ok=True)

    mlflow.set_tracking_uri("arn:aws:sagemaker:ap-south-1:258399971007:mlflow-tracking-server/mlflow-tracking-server")
    mlflow.set_experiment("Lead-Conversion-Classification-Precision")

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
from data_loader import load_from_redshift
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
    with mlflow.start_run(run_name="Lead-Conversion-Prediction-Drift-Detection"):
        # Load and split
        df = load_from_redshift()
        Xtr, Xv, Xt, ytr, yv, yt = split_data(df)

        # Keep raw for Evidently
        df_train = Xtr.copy()
        df_test  = Xt.copy()

        df_train = df_train.dropna(axis=1, how='all')
        df_test = df_test.dropna(axis=1, how='all')

        # Log Evidently reports
        log_evidently_report(df_train, df_test, dataset_name="train_vs_test")

def run_pipeline():
    df = load_from_redshift()
    df_cleaned = clean_data(df)
    df_featured = feature_engineering(df_cleaned)
    X_train, X_test, y_train, y_test = preprocess_data(df_featured)
    best_model_accuracy, y_pred_test_accuracy = train_evaluate_and_select_model_accuracy(X_train, y_train, X_test, y_test)
    best_model_precison, y_pred_test_precision = train_evaluate_and_select_model_precision(X_train, y_train, X_test, y_test)
    run_drift()
    

if __name__=="__main__":
    run_pipeline()