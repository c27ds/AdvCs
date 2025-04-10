import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import yfinance as yf
import pandas as pd

# Diagonal RNN model definition with both forward and backward passes
class DRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Mapping input to a hidden representation (doubled dimension for internal computations)
        self.input_to_hidden = nn.Linear(input_size, hidden_size * 2)
        
        # Additional hidden layers (using ModuleList)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size * 2) 
            for _ in range(num_layers - 1)
        ])
        
        # Output mapping
        self.hidden_to_output = nn.Linear(hidden_size * 2, output_size)
        
        # Self-recurrent diagonal connection (applied in both directions)
        self.self_recurrent = nn.Linear(hidden_size * 2, hidden_size * 2)
        
        # Extra transformation for the backward pass (transforming future hidden state)
        self.hidden_to_hidden = nn.Linear(hidden_size * 2, hidden_size * 2)
        
        self.activation = nn.LeakyReLU(0.01)
        
        # Register initial hidden state (will be cloned at runtime)
        self.register_buffer("initial_hidden", torch.zeros(1, hidden_size * 2))
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, prev_hidden):
        """
        Forward pass: uses the previous hidden state.
        x: current input tensor (1 x input_size)
        prev_hidden: previous hidden state (1 x hidden_size*2)
        """
        # Combine transformed input with transformed previous hidden state
        hidden = self.activation(self.input_to_hidden(x) + self.self_recurrent(prev_hidden))
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))
        output = self.hidden_to_output(hidden)
        return output, hidden

    def backward_pass(self, x, future_hidden):
        """
        Backward pass: incorporates future hidden state.
        x: current input tensor (1 x input_size)
        future_hidden: hidden state from a future time step (1 x hidden_size*2)
        """
        # Combine transformed input with two different transformations of the future hidden state.
        hidden = self.activation(
            self.input_to_hidden(x) + 
            self.hidden_to_hidden(future_hidden) + 
            self.self_recurrent(future_hidden)
        )
        # Process through the hidden layers in reverse order
        for layer in reversed(self.hidden_layers):
            hidden = self.activation(layer(hidden))
        output = self.hidden_to_output(hidden)
        return output, hidden

# Function to plot expected vs. actual outputs
def plot_results(expected, actual):
    plt.figure(figsize=(10,5))
    plt.plot(expected, label='Expected Output', linestyle='dashed')
    plt.plot(actual, label='Actual Output', marker='o', markersize=3)
    plt.xlabel("Time Steps")
    plt.ylabel("Output Value")
    plt.title("Expected vs. Actual Output")
    plt.legend()
    plt.show()

# Function to visualize the DRNN structure using NetworkX
def visualize_drnn(input_size, hidden_size, output_size):
    G = nx.DiGraph()
    
    input_nodes = [f"Input_{i}" for i in range(input_size)]
    # For visualization, we show "hidden_size" nodes even though internal dimension is hidden_size*2.
    hidden_nodes = [f"Hidden_{i}" for i in range(hidden_size)]
    output_nodes = [f"Output_{i}" for i in range(output_size)]
    
    all_nodes = input_nodes + hidden_nodes + output_nodes
    G.add_nodes_from(all_nodes)
    
    # Connect input nodes to hidden nodes
    for i in input_nodes:
        for h in hidden_nodes:
            G.add_edge(i, h)
    
    # Add self-recurrent (diagonal) connections for hidden nodes
    for h in hidden_nodes:
        G.add_edge(h, h, label='Self')
        
    # Connect hidden nodes to output nodes
    for h in hidden_nodes:
        for o in output_nodes:
            G.add_edge(h, o)
    
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', 
            node_size=2000, font_size=8)
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True) if 'label' in d}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Diagonal RNN Structure Visualization")
    plt.show()

# Function to load stock data using yfinance
def load_stock_data(ticker="AAPL", period="59d", interval="30m"):
    # First attempt
    data = yf.download(ticker, period=period, interval=interval)
    
    # If no data returned, try a fallback period
    if data.empty:
        print(f"No data returned for period '{period}' with interval '{interval}'. Trying a fallback period '60d'.")
        data = yf.download(ticker, period="60d", interval=interval)
    
    if data.empty:
        raise ValueError(f"No data fetched for ticker '{ticker}'. Check ticker, period, and interval. Response: {data}")
    
    # If the columns are a MultiIndex, reduce to single level
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Compute volatility as (High - Low) / Close (a relative measure)
    data['Volatility'] = (data['High'] - data['Low']) / data['Close']
    
    # Use 'Close' and 'Volatility' as features
    features = data[['Close', 'Volatility']].values.astype(np.float32)
    # Target: next time step's Close value
    targets = data['Close'].shift(-1).values.astype(np.float32)
    
    # Remove the last row where target is NaN
    features = features[:-1]
    targets = targets[:-1]
    
    # Normalize features (z-score normalization)
    features_mean = features.mean(axis=0)
    features_std = features.std(axis=0)
    features = (features - features_mean) / features_std
    
    # Normalize targets as well
    target_mean = targets.mean()
    target_std = targets.std()
    targets_norm = (targets - target_mean) / target_std
    
    # Convert to torch tensors
    x = torch.tensor(features)
    y = torch.tensor(targets_norm).unsqueeze(1)  # shape: (N, 1)
    
    return x, y, target_mean, target_std

# Training function using sequence-based learning (BPTT in both forward and backward directions)
def train_drnn(model, x, y, epochs=2500, lr=0.0001, seq_length=20):
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate decay parameters
    start_decay_epoch = 500
    end_decay_epoch = 2500
    initial_lr = lr
    final_lr = 0.00001

    for epoch in range(epochs):
        total_loss = 0
        # Initialize separate hidden states for forward and backward passes
        forward_hidden = model.initial_hidden.clone()
        backward_hidden = model.initial_hidden.clone()
        
        # Adjust learning rate if applicable
        if epoch >= start_decay_epoch:
            decay_factor = (epoch - start_decay_epoch) / (end_decay_epoch - start_decay_epoch)
            new_lr = initial_lr - decay_factor * (initial_lr - final_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # Process the entire time series in chunks (sequences)
        for i in range(0, len(x) - seq_length):
            optimizer.zero_grad()
            
            seq_input = x[i:i+seq_length]
            # For forward pass, targets are next time step values (from t=0 to seq_length-1)
            forward_target = y[i+1:i+seq_length+1]
            # For backward pass, we use the same sequence but shift the target one step backward.
            # We predict the previous time step’s target using future context.
            backward_target = y[i: i+seq_length-1]  # valid for t=1...seq_length-1
            
            forward_loss = 0
            # Forward pass: from t=0 to t=seq_length-1
            for t in range(seq_length):
                output, forward_hidden = model(seq_input[t].unsqueeze(0), forward_hidden.detach())
                if torch.isnan(output).any():
                    print(f"NaN detected in forward output at sequence {i}, time step {t}")
                    continue
                forward_loss += criterion(output, forward_target[t].unsqueeze(0))
            forward_loss = forward_loss / seq_length

            backward_loss = 0
            # Backward pass: from t=seq_length-1 down to 1
            for t in reversed(range(1, seq_length)):
                output, backward_hidden = model.backward_pass(seq_input[t].unsqueeze(0), backward_hidden.detach())
                if torch.isnan(output).any():
                    print(f"NaN detected in backward output at sequence {i}, time step {t}")
                    continue
                backward_loss += criterion(output, backward_target[t-1].unsqueeze(0))
            backward_loss = backward_loss / (seq_length - 1)

            total_loss_batch = forward_loss + backward_loss
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        avg_loss = total_loss / (len(x) - seq_length)
        if torch.isnan(torch.tensor(avg_loss)):
            print("Training diverged. Try reducing learning rate or adjusting model architecture.")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# For prediction we use the forward pass; backward pass is mainly used during training.
def predict(model, x, seq_length=20):
    model.eval()
    outputs = []
    hidden = model.initial_hidden.clone()
    with torch.no_grad():
        for i in range(len(x) - seq_length):
            output, hidden = model(x[i].unsqueeze(0), hidden.detach())
            outputs.append(output.item())
    return outputs

# Main execution code
def main():
    # Load stock data (ticker, period, and interval can be adjusted)
    x, y, target_mean, target_std = load_stock_data(ticker="AAPL", period="59d", interval="30m")
    
    # Set input and output sizes based on the data
    input_size = x.shape[1]  # Using 2 features (Close, Volatility)
    hidden_size = 16         # Internal dimension will be hidden_size*2
    output_size = 1          # Predicting the next Close value
    
    # Initialize the DRNN model
    model = DRNN(input_size, hidden_size, output_size)
    
    # Train the model (adjust epochs and learning rate as needed)
    train_drnn(model, x, y, epochs=250, lr=0.0001, seq_length=20)
    
    # Generate predictions using forward pass
    predictions = predict(model, x, seq_length=20)
    
    # De-normalize predictions and true target values for plotting
    predictions_denorm = [p * target_std + target_mean for p in predictions]
    true_values_denorm = [v * target_std + target_mean for v in y.numpy().flatten()[:-20]]
    
    # Plot the results
    plot_results(true_values_denorm, predictions_denorm)
    
    # Visualize the DRNN structure
    visualize_drnn(input_size, hidden_size, output_size)

if __name__ == "__main__":
    main()
