import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import os
from dotenv import load_dotenv

# Load environment variables from .env in the root directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Fetch sensitive info
db_user = os.getenv("DB_USER")
db_password = urllib.parse.quote_plus(os.getenv("DB_PASSWORD"))  # Encode special chars
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

# Create SQLAlchemy engine
engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Load CSV
df = pd.read_csv("data/Lead_Scoring_Raw.csv")

# Write to PostgreSQL (replace if exists)
df.to_sql('lead_scoring_data', engine, if_exists='replace', index=False)

print("✅ Data loaded into PostgreSQL successfully!")
