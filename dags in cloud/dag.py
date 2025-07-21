from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import boto3
import pandas as pd
import time
import json
import tempfile
import os

# Remove sagemaker import from global level to prevent DAG import error
# in environments where SageMaker is not installed.
# We'll import it inside the run_sagemaker_job function.

default_args = {
    'owner': 'Rithwik',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'Lead-Conversion-Classification-Pipeline',
    default_args=default_args,
    description='Lead Conversion Classification Pipeline',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
) as dag:

    s3_bucket = 'airflow-lead23'
    mlflow_tracking_uri = 'arn:aws:sagemaker:ap-south-1:258399971007:mlflow-tracking-server/mlflow-tracking-server'
    model_name = 'LeadConversionModel'
    sagemaker_role = 'arn:aws:iam::258399971007:role/sagemaker-role'

    def fetch_data_from_redshift(**kwargs):
        region = 'ap-south-1'
        workgroup_name = 'lead-workgroup'
        database_name = 'dev'
        secret_arn = 'arn:aws:secretsmanager:ap-south-1:258399971007:secret:redshift-serverless-secret-sz4Cm0'
        sql = 'SELECT * FROM lead_scoring LIMIT 100'

        client = boto3.client('redshift-data', region_name=region)
        response = client.execute_statement(
            WorkgroupName=workgroup_name,
            Database=database_name,
            SecretArn=secret_arn,
            Sql=sql
        )
        statement_id = response['Id']

        while True:
            desc = client.describe_statement(Id=statement_id)
            if desc['Status'] in ['FINISHED', 'FAILED', 'ABORTED']:
                break
            time.sleep(1)

        if desc['Status'] != 'FINISHED':
            raise Exception(f"Query failed: {desc}")

        result = client.get_statement_result(Id=statement_id)
        columns = [col['name'] for col in result['ColumnMetadata']]
        rows = result['Records']
        data = [[list(col.values())[0] if col else None for col in row] for row in rows]
        df = pd.DataFrame(data, columns=columns)

        ds = kwargs['ds']
        s3_path = f'raw/lead_scoring_{ds}.csv'
        s3 = boto3.client('s3')
        with tempfile.NamedTemporaryFile() as tmpfile:
            df.to_csv(tmpfile.name, index=False)
            s3.upload_file(tmpfile.name, s3_bucket, s3_path)

    def run_sagemaker_job(entry_point, task_name, s3_input_key, **kwargs):
        import sagemaker
        from sagemaker.estimator import Estimator

        session = sagemaker.Session()
        estimator = Estimator(
            entry_point=entry_point,
            source_dir=f's3://{s3_bucket}/code/',
            role=sagemaker_role,
            instance_count=1,
            instance_type='ml.m5.large',
            output_path=f's3://{s3_bucket}/models/',
            base_job_name=task_name,
            image_uri='683313688378.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
            sagemaker_session=session,
            hyperparameters={
                's3_bucket': s3_bucket,
                's3_data_key': s3_input_key,
                'mlflow_tracking_uri': mlflow_tracking_uri,
                'model_name': model_name
            }
        )
        estimator.fit(wait=True)

    def run_prepare(**kwargs):
        ds = kwargs['ds']
        run_sagemaker_job('data_cleaner.py', 'prepare-job', f'raw/lead_scoring_{ds}.csv', **kwargs)

    def run_preprocessing(**kwargs):
        ds = kwargs['ds']
        run_sagemaker_job('ml_preprocessing.py', 'preprocess-job', f'prepared/lead_scoring_{ds}.csv', **kwargs)

    def run_training(**kwargs):
        ds = kwargs['ds']
        run_sagemaker_job('modeling.py', 'train-job', f'preprocessed/lead_scoring_{ds}.parquet', **kwargs)

    def run_monitoring(**kwargs):
        ds = kwargs['ds']
        run_sagemaker_job('monitoring.py', 'monitoring-job', f'preprocessed/lead_scoring_{ds}.parquet', **kwargs)

    def check_drift(**kwargs):
        s3 = boto3.client('s3')
        report_key = f'reports/evidently_{kwargs["ds"]}.json'

        with tempfile.NamedTemporaryFile() as tmp:
            s3.download_file(s3_bucket, report_key, tmp.name)
            with open(tmp.name, 'r') as f:
                report = json.load(f)

        drift_score = report.get("data_drift", {}).get("metrics", {}).get("dataset_drift", {}).get("drift_share", 0)
        threshold = 0.3
        return 'retrain_model' if drift_score >= threshold else 'skip_retraining'

    # Define Airflow tasks
    fetch_data = PythonOperator(task_id='fetch_data_from_redshift', python_callable=fetch_data_from_redshift, provide_context=True)
    prepare = PythonOperator(task_id='prepare_data', python_callable=run_prepare, provide_context=True)
    preprocess = PythonOperator(task_id='preprocess_data', python_callable=run_preprocessing, provide_context=True)
    train = PythonOperator(task_id='train_model', python_callable=run_training, provide_context=True)
    monitor = PythonOperator(task_id='monitoring', python_callable=run_monitoring, provide_context=True)
    drift_check = BranchPythonOperator(task_id='check_drift', python_callable=check_drift, provide_context=True)
    retrain_model = PythonOperator(task_id='retrain_model', python_callable=run_training, provide_context=True)
    skip_retraining = DummyOperator(task_id='skip_retraining')
    end = DummyOperator(task_id='end')

    # DAG flow
    fetch_data >> prepare >> preprocess >> train >> monitor >> drift_check
    drift_check >> retrain_model >> end
    drift_check >> skip_retraining >> end
