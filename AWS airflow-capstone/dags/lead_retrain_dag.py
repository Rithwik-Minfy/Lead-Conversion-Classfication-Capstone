from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
import boto3
import pandas as pd
import tempfile
import os
import psycopg2
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# S3 and Redshift configs
S3_BUCKET = 'airflow-lead23'
S3_FILE_KEY = 'incoming/new_data.csv'
MAIN_SCRIPT_KEY = 'scripts/main.py'

DB_CONFIG = {
    'table': 'lead_scoring'
}


# DAG definition
default_args = {'owner': 'Rithwik'}

dag = DAG(
    'lead_conversion_retrain',
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval='@daily',
    catchup=False
)

# Step 1: Install Dependencies
install_deps = BashOperator(
    task_id='install_dependencies',
    bash_command='pip install -r /usr/local/airflow/requirements/requirements.txt',
    dag=dag
)

# Step 2: Check Data Drift
def check_data_drift(**kwargs):
    import boto3
    import pandas as pd
    import numpy as np
    import tempfile
    import psycopg2
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    import time

    s3 = boto3.client('s3')
    tmp_file = tempfile.NamedTemporaryFile(delete=False)

    # --- Step 1: Download new data from S3 ---
    s3.download_file(S3_BUCKET, S3_FILE_KEY, tmp_file.name)
    new_data = pd.read_csv(tmp_file.name)

    # --- Step 2: Load reference data from Redshift ---
    region='ap-south-1'
    workgroup_name='lead-workgroup'
    database_name='dev'
    secret_arn='arn:aws:secretsmanager:ap-south-1:258399971007:secret:redshift-serverless-secret-sz4Cm0'
    sql='SELECT * FROM lead_scoring'

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

    columns_to_drop = [
        'asymmetrique_profile_indexprofile index',
        'receivreceive_more_updates_about_our_coursese more updates about our courses'
    ]

    # Safely drop the columns only if they exist
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)


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
    ref_data = df

    print("📄 Before normalize Reference Columns:", ref_data.columns.tolist())
    print("📄 Before normalize New Data Columns:", new_data.columns.tolist())

    # --- Step 3: Normalize column names to avoid KeyError ---
    ref_data.columns = ref_data.columns.str.strip().str.lower().str.replace(' ', '_')
    new_data.columns = new_data.columns.str.strip().str.lower().str.replace(' ', '_')

    print("📄 After normalize Reference Columns:", ref_data.columns.tolist())
    print("📄 After normalize New Data Columns:", new_data.columns.tolist())

    # --- Step 4: Run Data Drift Detection ---
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=new_data)
    
    drift_score = report.as_dict()["metrics"][0]["result"]["dataset_drift"]

    # Save drift to XCom
    kwargs['ti'].xcom_push(key='drift_score', value=drift_score)

    print(f"📊 Drift Score: {drift_score}")
    if drift_score > 0.3:
        return 'retrain_model'
    else:
        return 'skip_retrain'


check_drift = PythonOperator(
    task_id='check_drift',
    python_callable=check_data_drift,
    provide_context=True,
    dag=dag
)

# Step 3: Retrain Model (Run main.py from S3)
retrain_model = BashOperator(
    task_id='retrain_model',
    bash_command=f'aws s3 cp s3://{S3_BUCKET}/{MAIN_SCRIPT_KEY} ./ && python main.py',
    dag=dag
)

# Step 4: Skip Model
skip_retrain = DummyOperator(task_id='skip_retrain', dag=dag)

# Branch
from airflow.operators.python import BranchPythonOperator

def branch_on_drift(**kwargs):
    drift = kwargs['ti'].xcom_pull(key='drift_score')
    return 'retrain_model' if drift > 0.3 else 'skip_retrain'

branch = BranchPythonOperator(
    task_id='branch_drift',
    python_callable=branch_on_drift,
    provide_context=True,
    dag=dag
)

# DAG flow
install_deps >> check_drift >> branch
branch >> retrain_model
branch >> skip_retrain
