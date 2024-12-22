import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import datetime
from keras.utils import to_categorical
# Train on 20 out of the last 30 days (test on last 10)
end_date = datetime.datetime.now() - datetime.timedelta(days=10)
print(end_date)
start_date = end_date - datetime.timedelta(days=20)
print(start_date)
# Start with XLK - can do others later
price_data = yf.Ticker("XLK").history(start=start_date, end=end_date, interval="30m")
data = price_data[['Open', 'High', 'Low', 'Close', 'Volume']]

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
sequence_length = 60
x = []
y = []

def generate_label(data, idx, lookforward=16, threshold=0.01):
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

y_categorical = to_categorical(y, num_classes=3)
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(x.shape[1], x.shape[2])))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x, y_categorical, epochs=10, batch_size=32)
model.save("rnn_training.h5")