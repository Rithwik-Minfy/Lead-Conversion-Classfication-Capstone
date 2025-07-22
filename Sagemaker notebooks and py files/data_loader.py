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
