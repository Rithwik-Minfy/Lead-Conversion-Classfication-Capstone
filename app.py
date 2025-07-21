import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from src.data_cleaner import clean_data
from src.feature_engineer import feature_engineering
from dotenv import load_dotenv
from sqlalchemy import create_engine
import urllib.parse
import mlflow.pyfunc

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['RESULT_FOLDER'] = "results"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri('http://localhost:5000')

# Load MLflow model from registry
MODEL_NAME = "LeadConversionModel"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/Production")

# Preprocessor
import joblib
PREPROCESSOR_PATH = "pkl_joblib_files/preprocessor.pkl"
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Database config from .env
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = urllib.parse.quote_plus(os.getenv("DB_PASSWORD"))
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_TABLE = "user_uploaded_predicted_lead_data"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Input fields
input_fields = [
    'Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call', 'Last Activity',
    'Country', 'Specialization', 'How did you hear about X Education',
    'What is your current occupation', 'What matters most to you in choosing a course',
    'Tags', 'Lead Profile', 'City', 'A free copy of Mastering The Interview', 'Last Notable Activity',
    'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit',
    'Asymmetrique Activity Score', 'Asymmetrique Profile Score',
    'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Lead Quality'
]

def classify_conversion(prob):
    if prob >= 0.7:
        return 'High'
    elif prob >= 0.4:
        return 'Medium'
    else:
        return 'Low'

@app.route('/')
def home():
    return render_template('form.html', input_fields=input_fields)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = {col: request.form.get(col) for col in input_fields}
        for col in ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit',
                    'Asymmetrique Activity Score', 'Asymmetrique Profile Score']:
            form_data[col] = float(form_data[col]) if form_data[col] else 0.0

        df = pd.DataFrame([form_data])
        df_clean = clean_data(df)
        df_feat = feature_engineering(df_clean)
        X_processed = preprocessor.transform(df_feat)

        prob = model.predict(X_processed)[0]
        y_pred = int(prob >= 0.5)
        label = classify_conversion(prob)

        return render_template("result.html",
                               prediction=y_pred,
                               probability=round(prob * 100, 2),
                               label=label)

    except Exception as e:
        return f"❌ Error: {e}"

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            df_clean = clean_data(df)
            df_feat = feature_engineering(df_clean)
            X_processed = preprocessor.transform(df_feat)
            y_prob = model.predict(X_processed)
            y_pred = (y_prob >= 0.5).astype(int)

            df['Converted_Prediction'] = y_pred
            df['Conversion_Probability'] = y_prob
            df['Conversion_Label'] = [classify_conversion(p) for p in y_prob]

            # Save locally
            result_path = os.path.join(app.config['RESULT_FOLDER'], "conversion_predictions.csv")
            df.to_csv(result_path, index=False)

            # Save to PostgreSQL
            engine = create_engine(DB_URL)
            df.to_sql(DB_TABLE, engine, if_exists='append', index=False)

            return send_file(result_path, as_attachment=True)

        except Exception as e:
            return f"❌ Error processing file: {e}"

@app.route('/download-template')
def download_template():
    template = pd.DataFrame(columns=input_fields)
    template_path = os.path.join(app.config['UPLOAD_FOLDER'], "sample_lead_template.csv")
    template.to_csv(template_path, index=False)
    return send_file(template_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
