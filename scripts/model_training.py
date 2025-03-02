# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scripts.model_training import train_logistic_regression

# Load the dataset using a relative path
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing values
df.dropna(inplace=True)

# Convert the target variable 'Churn' to numeric
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Perform one-hot encoding
df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 
                                 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                 'StreamingTV', 'StreamingMovies', 'Contract', 
                                 'PaperlessBilling', 'PaymentMethod'], drop_first=True)

# Split the data into features (X) and target (y)
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the Logistic Regression model
model = train_logistic_regression(X_train, y_train, X_test, y_test)