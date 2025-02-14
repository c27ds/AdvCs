import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

class DRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_to_hidden = nn.Linear(input_size, hidden_size * 2)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size * 2, hidden_size * 2) for _ in range(num_layers - 1)])
        self.hidden_to_output = nn.Linear(hidden_size * 2, output_size)
        self.activation = nn.LeakyReLU(0.01)  # Reverted to LeakyReLU for better gradient flow
        
        # Self-recurrent diagonal connections
        self.self_recurrent = nn.Linear(hidden_size * 2, hidden_size * 2)
        
        self.register_buffer("initial_hidden", torch.zeros(1, hidden_size * 2))
        self.apply(self.init_weights)  # Apply custom weight initialization

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Use Xavier initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, prev_hidden):
        # Forward pass with self-recurrent connections
        hidden = self.activation(self.input_to_hidden(x) + self.self_recurrent(prev_hidden))
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))
        output = self.hidden_to_output(hidden)
        return output, hidden

    def backward_pass(self, x, future_hidden):
        # Backward pass through time
        hidden = self.activation(self.input_to_hidden(x) + self.hidden_to_hidden(future_hidden) + self.self_recurrent(future_hidden))
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

# Function to visualize the Diagonal RNN structure
def visualize_drnn(input_size, hidden_size, output_size):
    G = nx.DiGraph()
    
    input_nodes = [f"Input_{i}" for i in range(input_size)]
    hidden_nodes = [f"Hidden_{i}" for i in range(hidden_size)]
    output_nodes = [f"Output_{i}" for i in range(output_size)]
    
    all_nodes = input_nodes + hidden_nodes + output_nodes
    G.add_nodes_from(all_nodes)
    
    for i in input_nodes:
        for h in hidden_nodes:
            G.add_edge(i, h)
    for h in hidden_nodes:
        G.add_edge(h, h, label='Self')  # Self-recurrent diagonal connections
    for h in hidden_nodes:
        for o in output_nodes:
            G.add_edge(h, o)
    
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=8)
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True) if 'label' in d}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Diagonal RNN Structure Visualization")
    plt.show()

# Improved DRNN training with sequence-based learning and lookahead prediction
def train_drnn(model, x, y, epochs=2500, lr=0.0001, seq_length=20):  # Reduced learning rate
    criterion = nn.SmoothL1Loss()  # Use SmoothL1Loss for more stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  # Use AdamW with weight decay
    
    # Define decay parameters
    start_decay_epoch = 500
    end_decay_epoch = 2500
    initial_lr = 0.0001
    final_lr = 0.00001  # Adjust this to set the final learning rate

    for epoch in range(epochs):
        total_loss = 0
        prev_hidden = model.initial_hidden.clone()
        
        # Manually adjust learning rate
        if epoch >= start_decay_epoch:
            decay_factor = (epoch - start_decay_epoch) / (end_decay_epoch - start_decay_epoch)
            new_lr = initial_lr - decay_factor * (initial_lr - final_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
        for i in range(0, len(x) - seq_length):
            optimizer.zero_grad()
            
            seq_input = x[i:i+seq_length]
            seq_target = y[i+1:i+seq_length+1]
            
            sequence_loss = 0
            for t in range(seq_length):
                output, prev_hidden = model(seq_input[t].unsqueeze(0), prev_hidden.detach())
                
                if torch.isnan(output).any():
                    print(f"NaN detected in output at sequence {i}, time step {t}")
                    continue
                
                sequence_loss += criterion(output, seq_target[t].unsqueeze(0))
            
            sequence_loss = sequence_loss / seq_length
            
            # Convert sequence_loss to a tensor before checking for NaN
            if torch.isnan(torch.tensor(sequence_loss)):
                print(f"NaN loss detected at sequence {i}. Skipping backward pass.")
                continue
                
            sequence_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Increased gradient clipping
            optimizer.step()
            
            total_loss += sequence_loss.item()
        
        avg_loss = total_loss / (len(x) - seq_length)
        if torch.isnan(torch.tensor(avg_loss)):
            print("Training diverged. Try reducing learning rate or adjusting model architecture.")
            break
            
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# Example usage
input_size = 3
hidden_size = 16
output_size = 1
model = DRNN(input_size, hidden_size, output_size)

time_steps = 1500
x = torch.randn(time_steps, input_size)
x = (x - x.mean()) / x.std()  # Move normalization here
y = torch.sin(torch.linspace(0, 10*np.pi, time_steps)).unsqueeze(1)

train_drnn(model, x, y, epochs=250)  # Set epochs to 250

actual_outputs = []
prev_hidden = model.initial_hidden.clone()

for i in range(time_steps - 20):
    output, prev_hidden = model(x[i].unsqueeze(0), prev_hidden.detach())
    actual_outputs.append(output.item())

plot_results(y.numpy().flatten()[:-20], actual_outputs)
visualize_drnn(input_size, hidden_size, output_size)