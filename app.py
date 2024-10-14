import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# Debugging print statement
print("Flask app is starting...")

app = Flask(__name__)

# Load the model and scaler with error handling
try:
    regmodel = pickle.load(open('regmodel.pkl', 'rb'))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    scalar = pickle.load(open('scaling.pkl', 'rb'))  # Corrected name of scaler file
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {e}")

@app.route('/')
def home():
    print("Home route accessed")
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print("Received data:", data)

        # Convert data to NumPy array
        input_data = np.array(list(data.values())).reshape(1, -1)
        print("Input array for prediction:", input_data)

        # Scale the input data using the loaded scaler
        new_data = scalar.transform(input_data)

        # Make the prediction using the loaded model
        output = regmodel.predict(new_data)
        print("Prediction:", output[0])

        return jsonify(output[0])
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("Starting Flask app...")
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
