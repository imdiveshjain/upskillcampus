from flask import Flask, request, jsonify
import joblib
from tensorflow.keras.models import load_model
import os
import redis

app = Flask(__name__)

RF_MODEL_PATH = os.path.join('models', 'rf_model.pkl')
XGB_MODEL_PATH = os.path.join('models', 'xgb_model.pkl')
LSTM_MODEL_PATH = os.path.join('models', 'lstm_model.h5')

if os.path.exists(RF_MODEL_PATH):
    rf_model = joblib.load(RF_MODEL_PATH)
else:
    rf_model = None

if os.path.exists(XGB_MODEL_PATH):
    xgb_model = joblib.load(XGB_MODEL_PATH)
else:
    xgb_model = None

if os.path.exists(LSTM_MODEL_PATH):
    lstm_model = load_model(LSTM_MODEL_PATH)
else:
    lstm_model = None

@app.route('/')
def home():
    return "Welcome to the Smart City Traffic Forecasting API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  
    if rf_model:
        rf_prediction = rf_model.predict([data['features']])
    if xgb_model:
        xgb_prediction = xgb_model.predict([data['features']])
    if lstm_model:
        lstm_prediction = lstm_model.predict([data['features']])
    
    result = {
        'rf_prediction': rf_prediction.tolist() if rf_model else None,
        'xgb_prediction': xgb_prediction.tolist() if xgb_model else None,
        'lstm_prediction': lstm_prediction.tolist() if lstm_model else None
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
