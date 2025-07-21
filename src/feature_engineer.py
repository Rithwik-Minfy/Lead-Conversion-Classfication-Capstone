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


if __name__ == "__main__":
    
    df_featured = feature_engineering(df_cleaned)
    df.to_csv("featured_data.csv", index=False)
