# ml_controller_simplified.py
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

# --- Model definition (same architecture) ---
class LSTMResidualModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size // 2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        out = out[:, -1, :]              # take last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- Load models & scaler (with robust fallback) ---
print("=" * 60)
print("Loading Hybrid LSTM-ARIMA Model...")
arima_model = lstm_model = scaler = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    arima_model = joblib.load("hybrid_arima_component.pkl")
    scaler = joblib.load("residual_scaler.pkl")
    lstm_model = LSTMResidualModel().to(device)
    # map_location ensures CPU/GPU compatibility; corrected torch.load usage
    lstm_state = torch.load("hybrid_lstm_pytorch.pth", map_location=device)
    lstm_model.load_state_dict(lstm_state)
    lstm_model.eval()
    print(f"✅ Models loaded. Device: {device}")
except Exception as e:
    print(f"⚠️  Model load failed (falling back): {e}")
print("=" * 60)

# --- Configuration & state ---
PHASE_CONFIG = {
    "Low": {"min_req": 0, "max_req": 120, "min_threads": 10, "max_threads": 20},
    "Normal": {"min_req": 121, "max_req": 500, "min_threads": 25, "max_threads": 40},
    "High": {"min_req": 501, "max_req": 1500, "min_threads": 60, "max_threads": 100},
    "Extreme": {"min_req": 1501, "max_req": 5000, "min_threads": 120, "max_threads": 160},
}
LOOKBACK_WINDOW = 20
request_history = deque(maxlen=1000)
last_5s_requests = deque(maxlen=LOOKBACK_WINDOW)

# --- Helper functions ---
def predict_next_load():
    """Return integer prediction of next 5s request count using ARIMA + LSTM residuals,
    or fallbacks if models/data are unavailable."""
    if len(last_5s_requests) < LOOKBACK_WINDOW or arima_model is None:
        # warming-up fallback: simple moving average of last up to 5 windows
        if len(last_5s_requests) >= 1:
            recent = list(last_5s_requests)[-5:]
            return int(max(0, np.mean(recent)))
        return 50  # default startup value

    try:
        history = np.array(last_5s_requests, dtype=float)
        arima_fit = arima_model.apply(history)
        arima_forecast = float(arima_fit.forecast(steps=1)[0])
        fitted = np.asarray(arima_fit.fittedvalues, dtype=float)

        if len(fitted) < LOOKBACK_WINDOW:
            return int(max(0, arima_forecast))

        residuals = history[-LOOKBACK_WINDOW:] - fitted[-LOOKBACK_WINDOW:]

        # scale, predict residual with LSTM, unscale
        scaled = scaler.transform(residuals.reshape(-1, 1)).reshape(1, LOOKBACK_WINDOW, 1)
        X = torch.from_numpy(scaled.astype(np.float32)).to(device)
        with torch.no_grad():
            out = lstm_model(X).cpu().numpy().squeeze()
        lstm_resid = float(scaler.inverse_transform(out.reshape(-1, 1)).squeeze())

        hybrid = arima_forecast + lstm_resid
        return int(max(0, round(hybrid)))
    except Exception:
        # if anything fails, fallback to recent average
        recent = list(last_5s_requests)[-5:] if len(last_5s_requests) >= 1 else [50]
        return int(max(0, np.mean(recent)))

def classify_load_phase(count):
    for name, cfg in PHASE_CONFIG.items():
        if cfg["min_req"] <= count <= cfg["max_req"]:
            return name
    return "Extreme" if count > PHASE_CONFIG["Extreme"]["max_req"] else "Low"

def map_requests_to_threads(predicted_requests):
    cfg = PHASE_CONFIG[classify_load_phase(predicted_requests)]
    req_range = cfg["max_req"] - cfg["min_req"]
    thread_range = cfg["max_threads"] - cfg["min_threads"]

    if req_range > 0:
        position = (predicted_requests - cfg["min_req"]) / req_range
        position = max(0.0, min(1.0, position))
        base = cfg["min_threads"] + position * thread_range
    else:
        base = cfg["min_threads"]

    # trend-based smoothing
    adjustment = 0
    if len(last_5s_requests) >= 4:
        recent = list(last_5s_requests)[-4:]
        trend = (recent[-1] - recent[0]) / 3.0
        if trend > 150: adjustment = 10
        elif trend > 50: adjustment = 5
        elif trend < -150: adjustment = -10
        elif trend < -50: adjustment = -5
    base += adjustment

    suggested = int(round(base))
    suggested = max(cfg["min_threads"], min(cfg["max_threads"], suggested))
    return max(10, min(160, suggested))

# --- Flask endpoints ---
@app.route("/ml/update_load/<int:request_count>", methods=["POST"])
def update_load(request_count):
    last_5s_requests.append(request_count)
    request_history.append({"timestamp": time.time(), "requests": request_count})
    return jsonify({"status": "updated", "history_size": len(last_5s_requests)})

@app.route("/ml/suggest_threads", methods=["GET"])
def suggest_threads():
    current = last_5s_requests[-1] if last_5s_requests else 0
    predicted = predict_next_load()
    decision = max(current, predicted)
    threads = map_requests_to_threads(decision)
    phase = classify_load_phase(decision)

    print(f"[{time.strftime('%H:%M:%S')}] Curr:{current:4d} Pred:{predicted:4d} Phase:{phase:8s} Threads:{threads:3d}")

    return jsonify({
        "suggested_threads": threads,
        "predicted_requests": predicted,
        "current_requests": current,
        "decision_load": decision,
        "phase": phase,
        "model": "Hybrid-LSTM-ARIMA",
        "confidence": "high" if len(last_5s_requests) >= LOOKBACK_WINDOW else "warming_up"
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "running",
        "model_loaded": arima_model is not None and lstm_model is not None and scaler is not None,
        "history_size": len(last_5s_requests)
    })

if __name__ == "__main__":
    print("--- ML PREDICTION SERVER (Hybrid LSTM-ARIMA) ---")
    print("POST /ml/update_load/<count>")
    print("GET  /ml/suggest_threads")
    print("GET  /health")
    app.run(host="0.0.0.0", port=5000, debug=False)
