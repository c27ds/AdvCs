import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

def run_rf_classifier(stock):
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

    if df is None or df.shape[0] < 100:
        print("Not enough data.")
        return

    df['Return'] = df['c'].pct_change().shift(-1)
    df['Movement'] = df['Return'].apply(lambda x: 1 if x > 0.002 else -1 if x < -0.002 else 0)
    df.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['c']])

    look_back = 30
    X, y = [], []
    for i in range(look_back, len(scaled_data) - 1):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(df['Movement'].iloc[i])

    X = np.array(X)
    y = np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy for {stock}: {accuracy:.3f}")

run_rf_classifier("GOOG")
