import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

# YEAR = os.getenv('YEAR')
# MONTH = os.getenv('MONTH')

### Create wrapper for flask ###
app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.get_json()
    print("Transforming Features")
    X_val = dv.transform(features)
    print("Running predict")
    y_pred = model.predict(X_val)
    results = {
        'mean_duration': np.mean(y_pred)
    }
    print(np.mean(y_pred))
    print("Returning Results")
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)