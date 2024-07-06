# Based on M. A. Hossain, et al. Hybrid Deep Learning Model for Stock Price
# Prediction,  2018 IEEE Symposium Series on Computational Intelligence (SSCI)
# DOI: 10.1109/SSCI.2018.8628641

import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    epsilon = 1e-10  # Add a small value to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mse, mae, mape

# Fetch historical data
#data = yf.download('^GSPC', start='2020-01-01', end='2023-01-01')
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Calculate moving averages and MACD
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA25'] = data['Close'].rolling(window=25).mean()
ema12 = data['Close'].ewm(span=12, adjust=False).mean()
ema26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema12 - ema26
data = data.dropna()

# Features and targets
features = data[['Close', 'MA5', 'MA10', 'MA25', 'MACD']].values
targets = data['Close'].values

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Prepare dataset for LSTM
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 15
X, y = create_dataset(features_scaled, window_size)
split_index = int(len(X) * 0.7)
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Function to plot results
def plot_results(y_train, y_test, train_predict, test_predict, title):
    plt.figure(figsize=(14, 5))
    plt.plot(data.index[window_size:split_index + window_size], y_train, color='red', label='Real Price (Train)')
    plt.plot(data.index[split_index + window_size:], y_test, color='orange', label='Real Price (Test)')
    plt.plot(data.index[window_size:split_index + window_size], train_predict, color='blue', label='Predicted Price (Train)')
    plt.plot(data.index[split_index + window_size:], test_predict, color='green', label='Predicted Price (Test)')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# Model 1: LSTM
model1 = Sequential([
    LSTM(100, return_sequences=True, input_shape=(window_size, 5)),
    LSTM(100, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(1)
])
model1.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
model1.fit(X_train, y_train, validation_split=0.1, shuffle=False, epochs=50, batch_size=16, verbose=1)
train_predict1 = model1.predict(X_train).reshape(-1)
test_predict1 = model1.predict(X_test).reshape(-1)
train_rmse1 = np.sqrt(mean_squared_error(y_train, train_predict1))
test_rmse1 = np.sqrt(mean_squared_error(y_test, test_predict1))
train_mse1, train_mae1, train_mape1 = calculate_metrics(y_train, train_predict1)
test_mse1, test_mae1, test_mape1 = calculate_metrics(y_test, test_predict1)
print(f'Model 1 - LSTM Train RMSE: {train_rmse1}, Test RMSE: {test_rmse1}')
plot_results(y_train, y_test, train_predict1, test_predict1, 'LSTM Model')

# Model 2: GRU
model2 = Sequential([
    GRU(100, return_sequences=True, input_shape=(window_size, 5)),
    GRU(50, return_sequences=False),
    Dense(1)
])
model2.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
model2.fit(X_train, y_train, validation_split=0.1, shuffle=False, epochs=50, batch_size=16, verbose=1)
train_predict2 = model2.predict(X_train).reshape(-1)
test_predict2 = model2.predict(X_test).reshape(-1)
train_rmse2 = np.sqrt(mean_squared_error(y_train, train_predict2))
test_rmse2 = np.sqrt(mean_squared_error(y_test, test_predict2))
train_mse2, train_mae2, train_mape2 = calculate_metrics(y_train, train_predict2)
test_mse2, test_mae2, test_mape2 = calculate_metrics(y_test, test_predict2)
print(f'Model 2 - GRU Train RMSE: {train_rmse2}, Test RMSE: {test_rmse2}')
plot_results(y_train, y_test, train_predict2, test_predict2, 'GRU Model')

# Model 3: Combined LSTM and GRU
model3 = Sequential([
    LSTM(250, return_sequences=True, input_shape=(window_size, 5)),
    Dropout(0.2),
    GRU(150, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model3.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
model3.fit(X_train, y_train, validation_split=0.1, shuffle=False, epochs=50, batch_size=16, verbose=1)
train_predict3 = model3.predict(X_train).reshape(-1)
test_predict3 = model3.predict(X_test).reshape(-1)
train_rmse3 = np.sqrt(mean_squared_error(y_train, train_predict3))
test_rmse3 = np.sqrt(mean_squared_error(y_test, test_predict3))
train_mse3, train_mae3, train_mape3 = calculate_metrics(y_train, train_predict3)
test_mse3, test_mae3, test_mape3 = calculate_metrics(y_test, test_predict3)
print(f'Model 3 - Combined LSTM and GRU Train RMSE: {train_rmse3}, Test RMSE: {test_rmse3}')
plot_results(y_train, y_test, train_predict3, test_predict3, 'Combined LSTM and GRU Model')

# Summary of results
results = {
    "Model": ["LSTM", "GRU", "Combined LSTM and GRU"],
    "Train RMSE": [train_rmse1, train_rmse2, train_rmse3],
    "Test RMSE": [test_rmse1, test_rmse2, test_rmse3],
    "Train MSE": [train_mse1, train_mse2, train_mse3],
    "Test MSE": [test_mse1, test_mse2, test_mse3],
    "Train MAE": [train_mae1, train_mae2, train_mae3],
    "Test MAE": [test_mae1, test_mae2, test_mae3],
    "Train MAPE": [train_mape1, train_mape2, train_mape3],
    "Test MAPE": [test_mape1, test_mape2, test_mape3]
}

results_df = pd.DataFrame(results)
print(results_df)

# Save the results to a CSV file
results_df.to_csv('model_comparison_results.csv', index=False)

# Plot the metrics for comparison
metrics = ['RMSE', 'MSE', 'MAE', 'MAPE']
for metric in metrics:
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['Model'], results_df[f'Train {metric}'], color='blue', alpha=0.7, label=f'Train {metric}')
    plt.bar(results_df['Model'], results_df[f'Test {metric}'], color='orange', alpha=0.7, label=f'Test {metric}')
    plt.title(f'Model Comparison - {metric}')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.legend()
    plt.show()

print("Comparison complete. Results saved to 'model_comparison_results.csv'.")

