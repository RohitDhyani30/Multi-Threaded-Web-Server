import time
import joblib
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, jsonify
from collections import deque
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- Model Definition ---
class LSTMResidualModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, dropout=0.2):
        super(LSTMResidualModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size//2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- Model & Config Loading ---
print("=" * 60)
print("Loading Hybrid LSTM-ARIMA Model...")
try:
    arima_model = joblib.load('hybrid_arima_component.pkl')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = LSTMResidualModel().to(device)
    lstm_model.load_state_dict(torch.load('hybrid_lstm_pytorch.pth', map_location=device, weights_only=True))
    lstm_model.eval()
    scaler = joblib.load('residual_scaler.pkl')
    print(f"✅ Models loaded successfully. Device: {device}")
except Exception as e:
    print(f"⚠️  Fallback mode (models failed to load): {e}")
    arima_model = lstm_model = scaler = None
print("=" * 60)

PHASE_CONFIG = {
    "Low": {"min_req": 0, "max_req": 120, "min_threads": 10, "max_threads": 20},
    "Normal": {"min_req": 121, "max_req": 500, "min_threads": 25, "max_threads": 40},
    "High": {"min_req": 501, "max_req": 1500, "min_threads": 60, "max_threads": 100},
    "Extreme": {"min_req": 1501, "max_req": 5000, "min_threads": 120, "max_threads": 160}
}

LOOKBACK_WINDOW = 20
request_history = deque(maxlen=1000)
last_5s_requests = deque(maxlen=LOOKBACK_WINDOW)

# --- Core Logic ---
def predict_next_load():
    """Hybrid LSTM-ARIMA prediction with fallback"""
    if len(last_5s_requests) < LOOKBACK_WINDOW or not arima_model:
        if len(last_5s_requests) > 0:
            return int(np.mean(list(last_5s_requests)[-5:]))
        return 50 # Default startup value
    
    try:
        history_array = np.array(list(last_5s_requests), dtype=float)
        
        # 1. ARIMA forecast
        temp_arima = arima_model.apply(history_array)
        arima_forecast = temp_arima.forecast(steps=1)[0]
        fitted_values = temp_arima.fittedvalues
        
        if len(fitted_values) < LOOKBACK_WINDOW:
            return max(0, int(arima_forecast))
        
        # 2. Calculate residuals
        residuals = history_array[-LOOKBACK_WINDOW:] - fitted_values[-LOOKBACK_WINDOW:]
        
        # 3. LSTM residual prediction
        residuals_scaled = scaler.transform(residuals.reshape(-1, 1)).flatten()
        X_input = torch.FloatTensor(residuals_scaled).reshape(1, LOOKBACK_WINDOW, 1).to(device)
        
        with torch.no_grad():
            lstm_pred_scaled = lstm_model(X_input).cpu().numpy()[0][0]
        
        lstm_residual_pred = scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
        
        # 4. Hybrid prediction
        final_prediction = arima_forecast + lstm_residual_pred
        return max(0, int(final_prediction))
    
    except Exception as e:
        return int(np.mean(list(last_5s_requests)[-5:])) if len(last_5s_requests) >= 5 else 50

def classify_load_phase(request_count):
    """Classify request load into phases"""
    for phase, config in PHASE_CONFIG.items():
        if config["min_req"] <= request_count <= config["max_req"]:
            return phase
    return "Extreme" if request_count > PHASE_CONFIG["Extreme"]["max_req"] else "Low"

def map_requests_to_threads(predicted_requests):
    """Map predicted request count to thread pool size via linear interpolation"""
    phase = classify_load_phase(predicted_requests)
    config = PHASE_CONFIG[phase]
    
    req_range = config["max_req"] - config["min_req"]
    thread_range = config["max_threads"] - config["min_threads"]
    
    if req_range > 0:
        position = max(0, min(1, (predicted_requests - config["min_req"]) / req_range))
        base_threads = config["min_threads"] + (position * thread_range)
    else:
        base_threads = config["min_threads"]
    
    # Trend-based adjustment for smoother transitions
    if len(last_5s_requests) >= 4:
        recent = list(last_5s_requests)[-4:]
        trend = (recent[-1] - recent[0]) / 3.0
        
        if trend > 150: adjustment = 10   # Sharp increase
        elif trend > 50: adjustment = 5    # Moderate increase
        elif trend < -150: adjustment = -10 # Sharp decrease
        elif trend < -50: adjustment = -5   # Moderate decrease
        else: adjustment = 0
        base_threads += adjustment
    
    suggested_threads = int(round(base_threads))
    suggested_threads = max(config["min_threads"], min(config["max_threads"], suggested_threads))
    return max(10, min(160, suggested_threads)) # Absolute safety bounds

# --- API Endpoints ---
@app.route('/ml/update_load/<int:request_count>', methods=['POST'])
def update_load(request_count):
    """Receive actual request count from server every 5 seconds"""
    last_5s_requests.append(request_count)
    request_history.append({'timestamp': time.time(), 'requests': request_count})
    return jsonify({'status': 'updated', 'history_size': len(last_5s_requests)})

@app.route('/ml/suggest_threads', methods=['GET'])
def suggest_threads():
    """Generate thread pool suggestion based on hybrid prediction"""
    current_load = last_5s_requests[-1] if len(last_5s_requests) > 0 else 0
    predicted_requests = predict_next_load()
    
    decision_load = max(current_load, predicted_requests)
    suggested_threads = map_requests_to_threads(decision_load)
    phase = classify_load_phase(decision_load)
    
    # Compact logging
    print(f"[{time.strftime('%H:%M:%S')}] Curr:{current_load:4d} | Pred:{predicted_requests:4d} | "
          f"Phase:{phase:8s} | Threads:{suggested_threads:3d}")
    
    return jsonify({
        'suggested_threads': suggested_threads,
        'predicted_requests': predicted_requests,
        'current_requests': current_load,
        'decision_load': decision_load,
        'phase': phase,
        'model': 'Hybrid-LSTM-ARIMA',
        'confidence': 'high' if len(last_5s_requests) >= LOOKBACK_WINDOW else 'warming_up'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'model_loaded': arima_model is not None,
        'history_size': len(last_5s_requests)
    })

if __name__ == '__main__':
    print("--- ML PREDICTION SERVER (Hybrid LSTM-ARIMA) ---")
    print("Endpoints:")
    print("  POST /ml/update_load/<count>")
    print("  GET  /ml/suggest_threads")
    print("  GET  /health")
    print(f"\n⚡ Running on http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)