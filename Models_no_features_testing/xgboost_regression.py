import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def directional_accuracy(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return np.mean(true_dir == pred_dir) * 100

def run_xgboost(stock):
    api_key = 'ymg2PfkAO6Wwe0bGClkVMinLs9WB4VYV'
   
    def fetch_data(stock, start_date, end_date):
        url = f'https://api.polygon.io/v2/aggs/ticker/{stock}/range/30/minute/{start_date}/{end_date}'
        params = {'apiKey': api_key}
        response = requests.get(url, params=params)
        data = response.json()
       
        if 'results' not in data:
            print("Error fetching data!")
            return None
       
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['c']]
        return df

    end_date = pd.to_datetime('today')
    end_date = end_date - pd.Timedelta(days=300)
    start_date = end_date - pd.Timedelta(days=100)
    df = fetch_data(stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if df is None:
        exit()

    df['c'].plot(title=f'{stock} Closing Prices', figsize=(12, 6))
    plt.show()

    # Normalize closing prices
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['c']])
    df['scaled_close'] = scaled

    # Create lag features
    look_back = 30
    for i in range(1, look_back + 1):
        df[f'lag_{i}'] = df['scaled_close'].shift(i)

    df.dropna(inplace=True)

    # Prepare input and output
    X = df[[f'lag_{i}' for i in range(1, look_back + 1)]].values
    y = df['scaled_close'].values

    # Train/test split
    split = int(len(X) * 0.66)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Inverse transform to original scale
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
    da = directional_accuracy(y_test_actual, y_pred_actual)

    print(f'RMSE: {rmse:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'Directional Accuracy: {da:.2f}%')

    # Plot predictions
    plt.plot(y_test_actual, color='blue', label='Actual Prices')
    plt.plot(y_pred_actual, color='red', label='Predicted Prices')
    plt.title(f'{stock} XGBoost Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    return [rmse, r2, mape, da]

# Example usage
run_xgboost("AAPL")