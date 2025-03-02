# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys
import os

# Add the 'scripts' folder to the Python path
sys.path.append(os.path.abspath('../scripts'))

# Import the evaluate_model function
from model_evaluation import evaluate_model # type: ignore

# Load the dataset using a relative path
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Make sure the 'TotalCharges' column is of numeric type
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing values in the 'TotalCharges' column
df.dropna(subset=['TotalCharges'], inplace=True)

# Drop the 'customerID' column
df.drop(columns=['customerID'], inplace=True)

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

# Initialize the model
model = LogisticRegression(random_state=42, max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
evaluate_model(model, X_test, y_test, y_pred)