import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import requests

def run_xgb_classifier(stock):
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

    end_date = pd.to_datetime('today') - pd.Timedelta(days=300)
    start_date = end_date - pd.Timedelta(days=100)
    df = fetch_data(stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if df is None or df.shape[0] < 100:
        print("Not enough data.")
        return

    df['Return'] = df['c'].pct_change().shift(-1)
    df['Movement'] = df['Return'].apply(lambda x: 2 if x > 0.002 else 0 if x < -0.002 else 1)  # -1 → 0, 0 → 1, 1 → 2
    df.dropna(inplace=True)

    df['c1'] = df['c'].shift(1)
    df['c2'] = df['c'].shift(2)
    df['c3'] = df['c'].shift(3)
    df.dropna(inplace=True)

    X = df[['c1', 'c2', 'c3']]
    y = df['Movement'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"3-Class Test Accuracy for {stock}: {acc:.3f}")

run_xgb_classifier("XOM")


