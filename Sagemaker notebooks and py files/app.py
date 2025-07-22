import os
import pandas as pd
import numpy as np
import joblib
import logging
import mlflow
import mlflow.pyfunc
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pyngrok import ngrok
from data_cleaner import clean_data
from feature_engineer import feature_engineering

# Load environment variables
load_dotenv()

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['RESULT_FOLDER'] = "results"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load preprocessor
PREPROCESSOR_PATH = "pkl_joblib_files/preprocessor.pkl"
preprocessor = joblib.load(PREPROCESSOR_PATH)

# MLflow config (SageMaker-hosted)
mlflow.set_tracking_uri("arn:aws:sagemaker:ap-south-1:258399971007:mlflow-tracking-server/mlflow-tracking-server")

def get_latest_model(stage="Production", alias=None):
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    registered = client.search_registered_models()
    if not registered:
        raise RuntimeError("No models registered")
    
    candidates = []
    for m in registered:
        for v in m.latest_versions:
            if alias and alias in getattr(v, 'aliases', []):
                candidates.append((m.name, v.version, v.creation_timestamp))
            elif not alias and v.current_stage == stage:
                candidates.append((m.name, v.version, v.creation_timestamp))
    
    if not candidates:
        raise ValueError(f"No models found in stage={stage}, alias={alias}")
    
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0]  # model name

def load_model_from_registry():
    try:
        model_name = get_latest_model(stage="Production")
        model_uri = f"models:/{model_name}/Production"
        logger.info(f"📦 Loading model from: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise

model = load_model_from_registry()

# Input fields for form
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
    return render_template("form.html", input_fields=input_fields)

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
        prediction = int(prob >= 0.5)
        label = classify_conversion(prob)

        return render_template("result.html",
                               prediction=prediction,
                               probability=round(prob * 100, 2),
                               label=label)
    except Exception as e:
        return f"❌ Error during prediction: {e}"

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        df = pd.read_csv(filepath)

        df_clean = clean_data(df)
        df_feat = feature_engineering(df_clean)
        X_processed = preprocessor.transform(df_feat)
        y_prob = model.predict(X_processed)
        y_pred = (y_prob >= 0.5).astype(int)

        df['Converted_Prediction'] = y_pred
        df['Conversion_Probability'] = y_prob
        df['Conversion_Label'] = [classify_conversion(p) for p in y_prob]

        result_path = os.path.join(app.config['RESULT_FOLDER'], "conversion_predictions.csv")
        df.to_csv(result_path, index=False)

        return send_file(result_path, as_attachment=True)

    except Exception as e:
        logger.exception("❌ Upload prediction failed")
        return f"❌ Error processing file: {e}"

@app.route('/download-template')
def download_template():
    template = pd.DataFrame(columns=input_fields)
    path = os.path.join(app.config['UPLOAD_FOLDER'], "sample_template.csv")
    template.to_csv(path, index=False)
    return send_file(path, as_attachment=True)

# ngrok tunnel
if __name__ == '__main__':
    port = 8000
    try:
        NGROK_AUTH_TOKEN = "305pHqsrXWp7k6i4OJ4AZ2bUmif_2Nwd7D3AhPJixpxL2RTNV"
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(port).public_url
        logger.info(f"🌐 Public URL: {public_url}")
    except Exception as e:
        logger.warning(f"⚠️ ngrok tunnel not started: {e}")

    app.run(host='0.0.0.0', port=port)
