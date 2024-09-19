import os
from flask import Blueprint, request, jsonify
import joblib
from tensorflow.keras.models import load_model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

bp = Blueprint('routes', __name__)

# Define paths for models
RF_MODEL_PATH = os.path.join('models', 'rf_model.pkl')
XGB_MODEL_PATH = os.path.join('models', 'xgb_model.pkl')
LSTM_MODEL_PATH = os.path.join('models', 'lstm_model.h5')

# Load models if they exist
rf_model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
xgb_model = joblib.load(XGB_MODEL_PATH) if os.path.exists(XGB_MODEL_PATH) else None
lstm_model = load_model(LSTM_MODEL_PATH) if os.path.exists(LSTM_MODEL_PATH) else None

@bp.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get('features', [])

    # Initialize predictions
    rf_prediction = None
    xgb_prediction = None
    lstm_prediction = None

    # Predict using available models
    if features:
        if rf_model:
            rf_prediction = rf_model.predict([features])
        if xgb_model:
            xgb_prediction = xgb_model.predict([features])
        if lstm_model:
            lstm_prediction = lstm_model.predict([features])

    # Prepare result
    result = {
        'rf_prediction': rf_prediction.tolist() if rf_prediction is not None else None,
        'xgb_prediction': xgb_prediction.tolist() if xgb_prediction is not None else None,
        'lstm_prediction': lstm_prediction.tolist() if lstm_prediction is not None else None
    }

    return jsonify(result)
