from data_loader import load_from_redshift
from data_cleaner import clean_data
from feature_engineer import feature_engineering
from data_preprocessor import preprocess_data
from model_trainer_evaluator_selector import train_evaluate_and_select_model_accuracy,train_evaluate_and_select_model_precision
from drift_detector import run_drift

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