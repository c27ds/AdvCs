import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib

# Fetch stock data
def fetch_stock_data(stock):
    end_date = datetime.datetime.now() - datetime.timedelta(days=10)
    print(end_date)
    start_date = end_date - datetime.timedelta(days=20)
    print(start_date)
    price_data = yf.Ticker(stock).history(start=start_date, end=end_date, interval="30m")
    print(price_data)
    price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data.between_time('09:30', '16:00')
    price_data = price_data[price_data.index.to_series().dt.dayofweek < 5]
    data = price_data[['Close', 'Volume']]
    print(data)

# Generate labels
def generate_labels(data, look_forward=10, threshold=0.05):
    labels = []
    close_prices = data['Close'].values
    for i in range(len(close_prices)):
        future_prices = close_prices[i + 1:i + 1 + look_forward]
        if len(future_prices) < look_forward:
            labels.append(None)  # Not enough data
            continue
        
        max_future = max(future_prices)
        min_future = min(future_prices)
        
        if min_future < close_prices[i] * (1 - threshold):
            labels.append(0)  # Down
        elif max_future > close_prices[i] * (1 + threshold):
            labels.append(2)  # Up
        else:
            labels.append(1)  # Flat
    
    return labels

# Prepare sequences
def create_sequences(data, labels, seq_length=30):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length].values
        label = labels[i + seq_length]
        if label is not None:
            sequences.append(seq)
            targets.append(label)
    return np.array(sequences), np.array(targets)

# DRNN Model
class DRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_to_hidden = nn.Linear(input_size, hidden_size * 2)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size * 2, hidden_size * 2) for _ in range(num_layers - 1)])
        self.hidden_to_output = nn.Linear(hidden_size * 2, output_size)
        self.activation = nn.LeakyReLU(0.01)
        self.self_recurrent = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.register_buffer("initial_hidden", torch.zeros(1, hidden_size * 2))
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, prev_hidden):
        hidden = self.activation(self.input_to_hidden(x) + self.self_recurrent(prev_hidden))
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))
        output = self.hidden_to_output(hidden)
        return output, hidden

# Train model
def train_drnn(model, train_x, train_y, epochs=250, lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    train_x, train_y = torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long)
    prev_hidden = model.initial_hidden.clone()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        
        for i in range(len(train_x)):
            output, prev_hidden = model(train_x[i], prev_hidden.detach())
            loss = criterion(output.unsqueeze(0), train_y[i].unsqueeze(0))
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_x):.6f}")
    
    return model

# Main function
def main(stock):
    data = fetch_stock_data(stock)
    print(data)
    labels = generate_labels(data)
    labels = [l for l in labels if l is not None]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    sequences, targets = create_sequences(pd.DataFrame(scaled_data), labels)
    
    model = DRNN(input_size=2, hidden_size=16, output_size=3)
    trained_model = train_drnn(model, sequences, targets)
    
    torch.save(trained_model.state_dict(), "drnn_stock_model.pth")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved!")

if __name__ == "__main__":
    main("TSLA")