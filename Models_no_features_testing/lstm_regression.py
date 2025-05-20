import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def run_lstm(stock):
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
        
        # Filter to market hours (9:30 AM to 4:00 PM) and weekdays (Mon–Fri)
        # df = df.between_time('09:30', '16:00')
        # df = df[df.index.to_series().dt.dayofweek < 5]
        
        print(df)
        return df

    end_date = pd.to_datetime('today')
    end_date = end_date - pd.Timedelta(days=300)
    start_date = end_date - pd.Timedelta(days=100)
    df = fetch_data(stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if df is None:
        exit()

    df['c'].plot(title=f'{stock} Closing Prices', figsize=(12, 6))
    plt.show()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    def create_dataset(data, look_back=30):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i-look_back:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    train_data = scaled_data[:30*len(df)//90]
    test_data = scaled_data[60*len(df)//90:]

    X_train, y_train = create_dataset(train_data, look_back=30)
    X_test, y_test = create_dataset(test_data, look_back=30)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose = 0)
    y_pred = model.predict(X_test)

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_actual = scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)

    print(f'RMSE: {rmse:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'MAPE: {mape:.2f}%')

    plt.plot(y_test_actual, color='blue', label='Actual Prices')
    plt.plot(y_pred_actual, color='red', label='Predicted Prices')
    plt.title(f'{stock} LSTM Model Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    return [rmse, r2, mape]

run_lstm("AAPL")