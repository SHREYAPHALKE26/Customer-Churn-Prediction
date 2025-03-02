from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json

# Load the model
model = joblib.load('models/churn_model.pkl')

# Load feature names
with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data
        data = request.get_json()

        # Create a DataFrame with all feature names
        input_data = pd.DataFrame(columns=feature_names)

        # Fill in the input data
        for key, value in data.items():
            if key in input_data.columns:
                input_data[key] = [value]

        # Fill missing columns with 0
        input_data = input_data.fillna(0)

        # Ensure the columns are in the correct order
        input_data = input_data[feature_names]

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)