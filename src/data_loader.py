import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import os
from dotenv import load_dotenv

# Load environment variables from .env in root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def load_data_from_postgres(
    table_name,
    db_user=os.getenv("DB_USER"),
    db_password=os.getenv("DB_PASSWORD"),
    db_host=os.getenv("DB_HOST"),
    db_port=os.getenv("DB_PORT"),
    db_name=os.getenv("DB_NAME")
):
    """
    Load the dataset from a PostgreSQL table and validate its structure.

    Args:
        table_name (str): Name of the table in PostgreSQL.

    Returns:
        pandas.DataFrame: Loaded dataset.

    Raises:
        ValueError: If required columns are missing or the dataset is empty.
    """
    # Encode password to be URL-safe
    encoded_password = urllib.parse.quote_plus(db_password)

    # Create connection string
    connection_str = f'postgresql+psycopg2://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(connection_str)

    # Load data
    df = pd.read_sql_table(table_name, con=engine)

    # Check if dataset is empty
    if df.empty:
        raise ValueError("Dataset is empty")

    # Required columns for Lead Scoring project
    required_columns = [
        'Prospect ID', 'Lead Number', 'Lead Origin', 'Lead Source', 'Do Not Email',
        'Do Not Call', 'Converted', 'TotalVisits', 'Total Time Spent on Website',
        'Page Views Per Visit', 'Last Activity', 'Country', 'Specialization',
        'How did you hear about X Education', 'What is your current occupation',
        'What matters most to you in choosing a course', 'Search', 'Magazine',
        'Newspaper Article', 'X Education Forums', 'Newspaper',
        'Digital Advertisement', 'Through Recommendations',
        'Receive More Updates About Our Courses', 'Tags', 'Lead Quality',
        'Update me on Supply Chain Content', 'Get updates on DM Content',
        'Lead Profile', 'City', 'Asymmetrique Activity Index',
        'Asymmetrique Profile Index', 'Asymmetrique Activity Score',
        'Asymmetrique Profile Score', 'I agree to pay the amount through cheque',
        'A free copy of Mastering The Interview', 'Last Notable Activity'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df.columns = df.columns.astype(str)
    df.columns = [str(col) for col in df.columns]

    print("✅ Dataset loaded successfully from PostgreSQL with shape:", df.shape)
    print(df.head())
    return df


if __name__ == "__main__":

    df = load_data_from_postgres(table_name='lead_scoring_data')