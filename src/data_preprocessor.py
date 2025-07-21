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
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
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

    return X_train_df, X_test_df, y_train, y_test


# To test directly
if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = preprocess_data(df_cleaned)
