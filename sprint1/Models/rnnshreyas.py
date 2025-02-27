import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

class SOM:
    def __init__(self, map_size, input_dim):
        self.map_size = map_size
        self.weights = np.random.rand(map_size[0], map_size[1], input_dim)

    def update_weights(self, input_data, learning_rate, sigma):
        input_data = input_data.squeeze()  # Ensure it's shape (hidden_size * 2,)
        bmu = self.find_bmu(input_data)
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                influence = np.exp(-((i - bmu[0])**2 + (j - bmu[1])**2) / (2 * sigma**2))
                self.weights[i, j] += learning_rate * influence * (input_data - self.weights[i, j])

    def find_bmu(self, input_data):
        distances = np.linalg.norm(self.weights - input_data, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

class RBM:
    def __init__(self, visible_dim, hidden_dim):
        self.weights = torch.randn(visible_dim, hidden_dim) * 0.01
        self.visible_bias = torch.zeros(visible_dim)
        self.hidden_bias = torch.zeros(hidden_dim)

    def update_weights(self, visible_data, learning_rate):
        hidden_prob = torch.sigmoid(visible_data @ self.weights + self.hidden_bias)
        hidden_state = torch.bernoulli(hidden_prob)
        visible_prob = torch.sigmoid(hidden_state @ self.weights.t() + self.visible_bias)
        hidden_prob_neg = torch.sigmoid(visible_prob @ self.weights + self.hidden_bias)
        self.weights += learning_rate * (visible_data.t() @ hidden_prob - visible_prob.t() @ hidden_prob_neg)
        self.visible_bias += learning_rate * (visible_data - visible_prob).mean(0)
        self.hidden_bias += learning_rate * (hidden_prob - hidden_prob_neg).mean(0)
        
class DRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
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

def lyapunov_stability_update(model, learning_rate):
    for param in model.parameters():
        param.data -= learning_rate * param.grad

# Improved DRNN training with sequence-based learning and lookahead prediction
def train_drnn(model, x, y, epochs=2500, lr=0.0001, seq_length=20):
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    som = SOM(map_size=(10, 10), input_dim=model.hidden_size * 2)
    rbm = RBM(visible_dim=model.hidden_size * 2, hidden_dim=model.output_size)
    
    for epoch in range(epochs):
        total_loss = 0
        prev_hidden = model.initial_hidden.clone()
        
        for i in range(0, len(x) - seq_length):
            optimizer.zero_grad()
            
            seq_input = x[i:i+seq_length]
            seq_target = y[i+1:i+seq_length+1]
            
            sequence_loss = 0
            for t in range(seq_length):
                output, prev_hidden = model(seq_input[t].unsqueeze(0), prev_hidden.detach())
                
                # Update weights using SOM
                som.update_weights(prev_hidden.detach().numpy(), learning_rate=0.01, sigma=1.0)
                
                # Update weights using RBM
                rbm.update_weights(prev_hidden, learning_rate=0.01)
                
                sequence_loss += criterion(output, seq_target[t].unsqueeze(0))
            
            sequence_loss = sequence_loss / seq_length
            sequence_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update weights using Lyapunov stability
            lyapunov_stability_update(model, learning_rate=0.01)
            
            total_loss += sequence_loss.item()
        
        avg_loss = total_loss / (len(x) - seq_length)
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

# Example usage
input_size = 3
hidden_size = 16
output_size = 1
model = DRNN(input_size, hidden_size, output_size)

time_steps = 1500
x = torch.randn(time_steps, input_size)
x = (x - x.mean()) / x.std()  # Move normalization here
y = torch.sin(torch.linspace(0, 10*np.pi, time_steps)).unsqueeze(1)

train_drnn(model, x, y, epochs=45)

actual_outputs = []
prev_hidden = model.initial_hidden.clone()

for i in range(time_steps - 20):
    output, prev_hidden = model(x[i].unsqueeze(0), prev_hidden.detach())
    actual_outputs.append(output.item())

plot_results(y.numpy().flatten()[:-20], actual_outputs)
visualize_drnn(input_size, hidden_size, output_size)

torch.save(model.state_dict(), "drnn_model.pth")

import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump((x.mean().item(), x.std().item()), f)