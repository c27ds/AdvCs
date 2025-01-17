import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import datetime
from keras.utils import to_categorical
from keras.layers import LSTM, Dropout, BatchNormalization
import joblib

end_date = datetime.datetime.now() - datetime.timedelta(days=10)
print(end_date)
start_date = end_date - datetime.timedelta(days=20)
print(start_date)
price_data = yf.Ticker("XLK").history(start=start_date, end=end_date, interval="30m")
price_data.index = pd.to_datetime(price_data.index)
price_data = price_data.between_time('09:30', '16:00')
price_data = price_data[price_data.index.to_series().dt.dayofweek < 5]
data = price_data[['Open', 'High', 'Low', 'Close', 'Volume']]

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
sequence_length = 302
x = []
y = []

def generate_label(data, idx, lookforward=5, threshold=0.01):
    if idx + lookforward >= len(data):
        return 0
    current_price = data[idx, 3]
    future_price = data[idx + lookforward, 3]
    future_return = (future_price - current_price) / current_price
    if future_return > threshold:
        print("Up")
        print(future_return)
        return 1
    elif future_return < -threshold:
        print("Down")
        print(future_return)
        return -1
    print("Flat")
    print(future_return)
    return 0

for i in range(sequence_length, len(data_scaled)):
    x.append(data_scaled[i-sequence_length:i])
    label = generate_label(data_scaled, i, lookforward=5, threshold=0.01)
    y.append(label)

x = np.array(x)
y = np.array(y)
print(y)
y_categorical = to_categorical(y, num_classes=3)
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y_categorical, epochs=10, batch_size=32)
model.save("lstmtradingmodel.h5")