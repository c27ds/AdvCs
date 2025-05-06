import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

def run_rf(stock):
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
    start_date = end_date - pd.Timedelta(days=30)
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
            X.append(data[i-look_back:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    train_data = scaled_data[:20*len(df)//30]
    test_data = scaled_data[20*len(df)//30:]

    X_train, y_train = create_dataset(train_data, look_back=30)
    X_test, y_test = create_dataset(test_data, look_back=30)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    print(f'RMSE: {rmse}')

    r2 = r2_score(y_test_actual, y_pred_actual)
    print(f'R²: {r2}')

    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
    print(f'MAPE: {mape}')

    def calculate_direction(data):
        direction = []
        for i in range(1, len(data)):
            if data[i] > data[i - 1]:
                direction.append(1)
            elif data[i] < data[i - 1]:
                direction.append(-1)
            else:
                direction.append(0)
        return direction

    actual_direction = calculate_direction(y_test_actual.flatten())
    predicted_direction = calculate_direction(y_pred_actual.flatten())

    correct_predictions = sum(1 for a, p in zip(actual_direction, predicted_direction) if a == p)
    directional_accuracy = correct_predictions / len(actual_direction)

    print(f'Directional Accuracy: {directional_accuracy * 100:.2f}%')

    plt.plot(y_test_actual, color='blue', label='Actual Prices')
    plt.plot(y_pred_actual, color='red', label='Predicted Prices')
    plt.title(f'{stock} Random Forest Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    return [rmse, r2, mape]

run_rf("AAPL")
