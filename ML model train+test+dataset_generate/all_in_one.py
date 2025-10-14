import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import joblib
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ===============================
# STEP 1: GENERATE BALANCED DATASET
# ===============================

def generate_balanced_server_dataset(num_rows=10000, interval=5):
    """
    Generate balanced dataset with equal representation of all phases
    and realistic transitions
    """
    print("=" * 80)
    print("📊 GENERATING BALANCED SERVER LOAD DATASET")
    print("=" * 80)
    
    timestamps = np.arange(0, num_rows * interval, interval)
    
    # Phase definitions
    load_levels = {
        "low": (10, 30),
        "normal": (80, 150),
        "high": (300, 500)
    }
    
    # Ensure balanced phases (33% each)
    phase_distribution = ['low', 'normal', 'high']
    rows_per_phase = num_rows // 3
    
    phases = []
    
    # Create balanced phase sequence
    for phase in phase_distribution:
        target_rows = rows_per_phase
        
        # Split this phase into 2-4 segments
        num_segments = random.randint(3, 5)
        for seg in range(num_segments):
            if seg == num_segments - 1:
                duration = target_rows - sum(d for p, d in phases if p == phase)
            else:
                duration = random.randint(target_rows // num_segments - 200,
                                        target_rows // num_segments + 200)
            
            if duration > 0:
                phases.append((phase, duration))
    
    # Shuffle phases for realism
    random.shuffle(phases)
    
    # Adjust to exact row count
    total = sum(d for _, d in phases)
    if total != num_rows:
        phases[-1] = (phases[-1][0], phases[-1][1] + (num_rows - total))
    
    # Generate requests with smooth transitions
    requests = []
    prev_value = np.random.randint(80, 150)
    
    for phase, duration in phases:
        low, high = load_levels[phase]
        
        target_start = np.random.randint(low + (high-low)//4, high - (high-low)//4)
        target_end = np.random.randint(low + (high-low)//4, high - (high-low)//4)
        
        transition_length = min(30, duration // 2)
        transition = np.linspace(prev_value, target_start, transition_length)
        
        main_length = duration - transition_length
        base_values = np.linspace(target_start, target_end, main_length)
        
        noise = np.random.normal(0, (high - low) * 0.08, main_length)
        main_segment = np.clip(base_values + noise, low * 0.8, high * 1.2)
        
        segment = np.concatenate([transition, main_segment])
        requests.extend(segment)
        prev_value = segment[-1]
    
    requests = requests[:num_rows]
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Requests_Last_5s": np.round(requests).astype(int)
    })
    
    def classify_phase(val):
        if 10 <= val <= 30: return "Low"
        elif 80 <= val <= 150: return "Normal"
        elif 300 <= val <= 500: return "High"
        else: return "Transition"

    phases_check = df['Requests_Last_5s'].apply(classify_phase)
    phase_counts = phases_check.value_counts()

    print("\n✅ Dataset generated with phase distribution:")
    for phase, count in phase_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {phase:12}: {count:5} samples ({percentage:5.2f}%)")
    
    return df


# Generate balanced dataset
balanced_df = generate_balanced_server_dataset(num_rows=10000, interval=5)
balanced_df.to_csv("balanced_server_dataset.csv", index=False)
print(f"\n✅ Balanced dataset saved as 'balanced_server_dataset.csv'\n")


# ===============================
# STEP 2: PYTORCH LSTM MODEL
# ===============================

class LSTMResidualModel(nn.Module):
    """PyTorch LSTM model for residual prediction"""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, dropout=0.2):
        super(LSTMResidualModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
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
        
        out = out[:, -1, :]  # Take last timestep
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


print("=" * 80)
print("🚀 TRAINING HYBRID LSTM-ARIMA MODEL (PyTorch Version)")
print("=" * 80)

# Load dataset
df = pd.read_csv('balanced_server_dataset.csv')
df = df.sort_values(by='Timestamp')
rps_series = df['Requests_Last_5s'].astype(float).values
timestamps = df['Timestamp'].values

# Train-test split
train_size = int(len(rps_series) * 0.8)
test_size = len(rps_series) - train_size
train = rps_series[:train_size]
test = rps_series[train_size:]

print(f"\n📊 Dataset Split:")
print(f"   Training samples: {train_size}")
print(f"   Testing samples:  {test_size}")


# ===============================
# STEP 2A: ARIMA for Linear Component
# ===============================
print("\n📈 Step 1: Training ARIMA for linear patterns...")
arima_order = (3, 1, 2)
arima_model = ARIMA(train, order=arima_order)
arima_fit = arima_model.fit()

# FIX: Handle ARIMA fitted values correctly
arima_train_pred = arima_fit.fittedvalues

# Calculate residuals - align arrays properly
# With differencing, fittedvalues starts from index 1
if len(arima_train_pred) < len(train):
    # Pad the beginning with zero residual
    aligned_train = train[len(train) - len(arima_train_pred):]
    arima_residuals = aligned_train - arima_train_pred
else:
    arima_residuals = train - arima_train_pred

print(f"✅ ARIMA({arima_order[0]},{arima_order[1]},{arima_order[2]}) trained")
print(f"   Training samples: {len(train)}")
print(f"   Fitted values: {len(arima_train_pred)}")
print(f"   Residuals: {len(arima_residuals)}")
print(f"   Residual mean: {np.mean(arima_residuals):.3f}")
print(f"   Residual std:  {np.std(arima_residuals):.3f}")


# ===============================
# STEP 2B: PyTorch LSTM for Residuals
# ===============================
print("\n🧠 Step 2: Training PyTorch LSTM for non-linear residuals...")

def create_sequences(data, lookback=20):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

lookback = 20
scaler = MinMaxScaler(feature_range=(-1, 1))
residuals_scaled = scaler.fit_transform(arima_residuals.reshape(-1, 1)).flatten()

X_train, y_train = create_sequences(residuals_scaled, lookback)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

print(f"   LSTM training samples: {X_train.shape[0]}")
print(f"   Lookback window: {lookback}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = LSTMResidualModel(input_size=1, hidden_size=64, num_layers=3, dropout=0.2).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
best_loss = float('inf')
patience = 10
patience_counter = 0

print(f"   Training on: {device}")
for epoch in range(num_epochs):
    lstm_model.train()
    epoch_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        outputs = lstm_model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(lstm_model.state_dict(), 'hybrid_lstm_pytorch.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

print(f"✅ PyTorch LSTM trained successfully")
print(f"   Best loss: {best_loss:.6f}")

# Load best model
lstm_model.load_state_dict(torch.load('hybrid_lstm_pytorch.pth', weights_only=True))
lstm_model.eval()

# Save components
joblib.dump(arima_fit, 'hybrid_arima_component.pkl')
joblib.dump(scaler, 'residual_scaler.pkl')
print("\n✅ Hybrid model components saved")


# ===============================
# STEP 3: HYBRID PREDICTION
# ===============================

def hybrid_predict_pytorch(history_data, arima_model, lstm_model, scaler, device, lookback=20):
    """Predict using hybrid approach with PyTorch"""
    try:
        # ARIMA prediction
        temp_arima = arima_model.apply(history_data)
        arima_forecast = temp_arima.forecast(steps=1)[0]
        
        # Get residuals
        fitted_values = temp_arima.fittedvalues
        
        # FIX: Handle array alignment for residuals
        if len(fitted_values) >= lookback:
            recent_fitted = fitted_values[-lookback:]
            # Align actual values with fitted values
            offset = len(history_data) - len(fitted_values)
            recent_actual = history_data[offset + len(fitted_values) - lookback : offset + len(fitted_values)]
            residuals = recent_actual - recent_fitted
        else:
            # Not enough history, use ARIMA only
            return max(0, arima_forecast)
        
        # LSTM prediction
        residuals_scaled = scaler.transform(residuals.reshape(-1, 1)).flatten()
        X_input = torch.FloatTensor(residuals_scaled).reshape(1, lookback, 1).to(device)
        
        with torch.no_grad():
            lstm_pred_scaled = lstm_model(X_input).cpu().numpy()[0][0]
        
        lstm_residual_pred = scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
        
        final_prediction = arima_forecast + lstm_residual_pred
        return max(0, final_prediction)
    
    except Exception as e:
        # Fallback to simple prediction if error occurs
        print(f"Warning: Prediction error - {e}, using fallback")
        return max(0, np.mean(history_data[-10:]))


# ===============================
# STEP 4: TESTING
# ===============================

print("\n" + "=" * 80)
print("🔄 PERFORMING HYBRID ROLLING FORECAST")
print("=" * 80)

predictions = []
history = list(train)

for i in range(len(test)):
    pred = hybrid_predict_pytorch(np.array(history), arima_fit, lstm_model, scaler, device, lookback)
    predictions.append(pred)
    history.append(test[i])
    
    if (i + 1) % 500 == 0:
        print(f"   Processed {i + 1}/{test_size} samples")

predictions = np.array(predictions)
print("✅ Forecast completed")


# ===============================
# STEP 5: EVALUATION
# ===============================

print("\n" + "=" * 80)
print("📊 MODEL PERFORMANCE")
print("=" * 80)

mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))
mape = np.mean(np.abs((test - predictions) / test)) * 100
r2 = r2_score(test, predictions)

print(f"\n📈 Overall Metrics:")
print(f"   MAE:  {mae:.3f} requests")
print(f"   RMSE: {rmse:.3f} requests")
print(f"   MAPE: {mape:.2f}%")
print(f"   R²:   {r2:.4f}")


# ===============================
# STEP 6: PHASE-WISE ANALYSIS
# ===============================

def classify_phase(value):
    """Classify single value into phase"""
    if 10 <= value <= 30:
        return "Low"
    elif 80 <= value <= 150:
        return "Normal"
    elif 300 <= value <= 500:
        return "High"
    else:
        return "Transition"

# Classify all test samples
actual_phases = [classify_phase(val) for val in test]
predicted_phases = [classify_phase(val) for val in predictions]

# Create results dataframe
results_df = pd.DataFrame({
    'Timestamp': timestamps[train_size:],
    'Actual': test,
    'Predicted': predictions,
    'Actual_Phase': actual_phases,
    'Predicted_Phase': predicted_phases,
    'Error': test - predictions,
    'Abs_Error': np.abs(test - predictions),
    'Pct_Error': np.abs((test - predictions) / (test + 1e-6)) * 100  # Add small value to avoid division by zero
})

print("\n" + "=" * 80)
print("🎯 PHASE-WISE PERFORMANCE")
print("=" * 80)

phases_to_analyze = ["Low", "Normal", "High"]

for phase in phases_to_analyze:
    phase_data = results_df[results_df['Actual_Phase'] == phase]
    
    if len(phase_data) > 0:
        phase_mae = phase_data['Abs_Error'].mean()
        phase_rmse = np.sqrt((phase_data['Error'] ** 2).mean())
        phase_mape = phase_data['Pct_Error'].mean()
        phase_samples = len(phase_data)
        
        # Phase classification accuracy
        correct = (phase_data['Actual_Phase'] == phase_data['Predicted_Phase']).sum()
        phase_accuracy = (correct / phase_samples) * 100
        
        print(f"\n{phase.upper()} Phase:")
        print(f"   Samples:     {phase_samples}")
        print(f"   MAE:         {phase_mae:.3f} requests")
        print(f"   RMSE:        {phase_rmse:.3f} requests")
        print(f"   MAPE:        {phase_mape:.2f}%")
        print(f"   Accuracy:    {phase_accuracy:.2f}%")


# Save results
results_df.to_csv('hybrid_pytorch_results.csv', index=False)
print("\n✅ Detailed results saved to 'hybrid_pytorch_results.csv'")


# ===============================
# STEP 7: VISUALIZATION
# ===============================

print("\n" + "=" * 80)
print("📊 GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

test_timestamps = timestamps[train_size:]

# Plot 1: Predictions vs Actual
axes[0].plot(test_timestamps, test, label='Actual', alpha=0.8, linewidth=1.5, color='blue')
axes[0].plot(test_timestamps, predictions, label='Predicted', alpha=0.8, 
             linewidth=1.5, linestyle='--', color='red')
axes[0].set_title('Hybrid LSTM-ARIMA: Actual vs Predicted', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Requests per 5s')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Error
axes[1].plot(test_timestamps, results_df['Error'], color='red', alpha=0.6, linewidth=1)
axes[1].axhline(0, color='black', linestyle='-', linewidth=1.5)
axes[1].set_title('Prediction Error', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Error')
axes[1].grid(True, alpha=0.3)

# Plot 3: Percentage Error
axes[2].plot(test_timestamps, results_df['Pct_Error'], color='orange', alpha=0.7, linewidth=1)
axes[2].axhline(mape, color='red', linestyle='--', linewidth=2, label=f'Mean MAPE: {mape:.2f}%')
axes[2].set_title('Percentage Error', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('% Error')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hybrid_model_results.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: hybrid_model_results.png")
plt.show()

print("\n" + "=" * 80)
print("✅ TRAINING AND EVALUATION COMPLETE!")
print("=" * 80)

print("\n🎯 Model Quality Assessment:")
if mape < 10 and r2 > 0.85:
    print("   🎉 EXCELLENT - Model is production-ready!")
elif mape < 15 and r2 > 0.75:
    print("   ✅ GOOD - Model performance is acceptable")
else:
    print("   ⚠️  NEEDS IMPROVEMENT - Consider retraining or tuning")

print(f"\n💾 Saved Files:")
print("   - balanced_server_dataset.csv")
print("   - hybrid_arima_component.pkl")
print("   - hybrid_lstm_pytorch.pth")
print("   - residual_scaler.pkl")
print("   - hybrid_pytorch_results.csv")
print("   - hybrid_model_results.png")
