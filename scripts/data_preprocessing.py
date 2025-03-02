# scripts/data_preprocessing.py
import pandas as pd

def handle_missing_values(df):
    # Handle missing values
    df.dropna(inplace=True)
    return df

def encode_categorical_variables(df):
    """
    Perform one-hot encoding for categorical variables.
    """
    df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 
                                     'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                     'StreamingTV', 'StreamingMovies', 'Contract', 
                                     'PaperlessBilling', 'PaymentMethod'], drop_first=True)
    return df