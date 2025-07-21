# Applying the techniques observed in EDA

def clean_data(df):

    # Prospect ID and Lead Number adds no additional information so drop these columns
    df = df.drop(columns=['Prospect ID', 'Lead Number'], errors='ignore')

    # Removing as these contains constant value "False"
    df = df.drop(columns=["Magazine", "Receive More Updates About Our Courses", "Update me on Supply Chain Content", "Get updates on DM Content","I agree to pay the amount through cheque"], errors='ignore')
    
    # Check for duplicate records
    print("\nDuplicate Records:", df.duplicated().sum())

    # if duplicate rows found then drop
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Filling the missing null values and label encoding "Asymmetrique Profile Index" and "Asymmetrique Activity Index"
    # and dropping unneccesary columns
    drop_cols = [
        'Newspaper', 'Newspaper Article',
        'X Education Forums', 'Search', 'Through Recommendations'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Standardize labels
    df['Lead Source'] = df['Lead Source'].str.strip().str.lower()
    df['Lead Source'] = df['Lead Source'].replace({'google': 'Google'})
    df['Lead Source'] = df['Lead Source'].fillna('Missing')

    # Replace 'Select' with 'Missing'
    select_to_missing_cols = [
        'Specialization', 'Lead Profile', 'City', 'How did you hear about X Education'
    ]
    for col in select_to_missing_cols:
        df[col] = df[col].replace('Select', 'Missing')
        df[col] = df[col].fillna('Missing')

    # Fill remaining high-null categorical columns
    df['Country'] = df['Country'].fillna('India')
    df['What is your current occupation'] = df['What is your current occupation'].fillna('Unemployed')
    df['What matters most to you in choosing a course'] = df['What matters most to you in choosing a course'].fillna('Better Career Prospects')
    df['Tags'] = df['Tags'].fillna('Missing')
    df['Lead Quality'] = df['Lead Quality'].fillna('Missing')

    # Ordinal encoding for Asymmetrique Index columns
    ordinal_map = {"01.High": 3, "02.Medium": 2, "03.Low": 1}
    df["Asymmetrique Profile Index"] = df["Asymmetrique Profile Index"].map(ordinal_map)
    df["Asymmetrique Activity Index"] = df["Asymmetrique Activity Index"].map(ordinal_map)

    # Fill with median
    for col in [
        'Asymmetrique Profile Index', 'Asymmetrique Activity Index',
        'Asymmetrique Profile Score', 'Asymmetrique Activity Score',
        'TotalVisits', 'Page Views Per Visit'
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Done
    print("Data cleaned and ready. Remaining nulls:")
    print(df.isnull().sum()[df.isnull().sum() > 0])


    # Mapping the labels of the columns with similar meaning and if more labels to convert to less labels
    lead_source_map = {
        'google': 'Google',
        'google ads': 'Google',
        'organic search': 'Organic Search',
        'olark chat': 'Olark Chat',
        'direct traffic': 'Direct Traffic',
        'reference': 'Reference',
        'welingak website': 'Welingak',
        'facebook': 'Social Media',
        'bing': 'Other',
        'click2call': 'Other',
        'press_release': 'Other',
        'social media': 'Social Media',
        'live chat': 'Olark Chat',
        'youtubechannel': 'Other',
        'testone': 'Other',
        'pay per click ads': 'Other',
        'welearnblog_home': 'Other',
        'welearn': 'Other',
        'blog': 'Other',
        'nc_edm': 'Other'
    }

    specialization_map = {
        'finance management': 'Finance',
        'banking, investment and insurance': 'Finance',
        'human resource management': 'HR',
        'marketing management': 'Marketing',
        'operations management': 'Operations',
        'it projects management': 'IT',
        'business administration': 'Business',
        'supply chain management': 'Operations',
        'e-commerce': 'Business',
        'retail management': 'Marketing',
        'media and advertising': 'Marketing',
        'travel and tourism': 'Other',
        'international business': 'Business',
        'healthcare management': 'Other',
        'hospitality management': 'Other',
        'rural and agribusiness': 'Other',
        'e-business': 'Business',
        'services excellence': 'Other',
        'missing': 'Missing',
        'select': 'Missing'
    }

    tags_map = {
        'will revert after reading the email': 'Reverting',
        'interested in other courses': 'Interested Other',
        'interested  in full time mba': 'Interested Other',
        'graduation in progress': 'Interested Other',
        'not doing further education': 'Not Interested',
        'wrong number given': 'Not Reachable',
        'opp hangup': 'Not Reachable',
        'number not provided': 'Not Reachable',
        'invalid number': 'Not Reachable',
        'still thinking': 'Still Thinking',
        'shall take in the next coming month': 'Still Thinking',
        'want to take admission but has financial problems': 'Still Thinking',
        'lost to eins': 'Lost',
        'lost to others': 'Lost',
        'in touch with eins': 'Lost',
        'diploma holder (not eligible)': 'Not Eligible',
        'university not recognized': 'Not Eligible',
        'recognition issue (dec approval)': 'Not Eligible',
        'already a student': 'Already Student',
        'switched off': 'Not Reachable',
        'busy': 'Not Reachable',
        'ringing': 'Not Reachable',
        'missing': 'Missing',
        '': 'Missing',
    }

    lead_quality_map = {
        'high in relevance': 'High',
        'might be': 'Medium',
        'not sure': 'Medium',
        'low in relevance': 'Low',
        'worst': 'Low',
        'missing': 'Missing'
    }

    lead_profile_map = {
        'potential lead': 'Potential',
        'other leads': 'Other',
        'student of someschool': 'Student',
        'lateral student': 'Student',
        'dual specialization student': 'Student',
        'select': 'Missing',
        'missing': 'Missing'
    }

    heard_map = {
        'online search': 'Online',
        'word of mouth': 'Referral',
        'student of someschool': 'Referral',
        'multiple sources': 'Multiple',
        'advertisements': 'Ads',
        'social media': 'Social',
        'email': 'Direct',
        'sms': 'Direct',
        'other': 'Other',
        'select': 'Missing',
        'missing': 'Missing'
    }

    # Clean cell values only (not column names)
    df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

    # Apply mappings with correct column names
    df['Lead Source'] = df['Lead Source'].replace(lead_source_map)
    df['Specialization'] = df['Specialization'].replace(specialization_map)
    df['Tags'] = df['Tags'].replace(tags_map)
    df['Lead Quality'] = df['Lead Quality'].replace(lead_quality_map)
    df['Lead Profile'] = df['Lead Profile'].replace(lead_profile_map)
    df['How did you hear about X Education'] = df['How did you hear about X Education'].replace(heard_map)
    print("✅ Data cleaning completed")
    print("Shape of the data after cleaning ",df.shape)
    return df


if __name__=="__main__":
    df_cleaned = clean_data(df)
