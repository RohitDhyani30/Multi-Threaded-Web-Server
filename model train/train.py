import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

data = pd.read_csv('enhanced_ml_dataset.csv')

# Feature engineering (create lagged features for Requests_Last_5s)
for lag in range(1, 6):
    data[f'Requests_Lag_{lag}'] = data['Requests_Last_5s'].shift(lag)

# Drop rows with NaN due to lagging
data.dropna(inplace=True)

# Define features and target
X = data[['Requests_Lag_1', 'Requests_Lag_2', 'Requests_Lag_3', 'Requests_Lag_4', 'Requests_Lag_5',
          'CPU_Usage(%)', 'Memory_Usage(%)', 'Avg_Response_Time(ms)', 'Thread_Utilization(%)', 'Requests_Per_Second']]
y = data['Requests_Last_5s']

# Train-test split (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'xgb_server_request_model.joblib')
print("Model saved as 'xgb_server_request_model.joblib'")

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"XGBoost RMSE: {rmse:.2f}")
print(f"XGBoost MAE: {mae:.2f}")


plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Requests')
plt.plot(y_pred, label='Predicted Requests')
plt.title('Actual vs Predicted Requests (Test Set)')
plt.xlabel('Test Data Points')
plt.ylabel('Requests Last 5s')
plt.legend()
plt.show()
