import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Step 1: Fetch XLK data in 30-minute timeframe
def fetch_xlk_data():
    # Download 60 days of XLK data in 30-minute intervals
    ticker = "XLK"
    data = yf.download(ticker, period="1mo", interval="30m")
    return data

# Step 2: Preprocess the data
def preprocess_data(data, sequence_length=60):
    # Select relevant features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[features]
    
    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled.iloc[i:i+sequence_length].values)
        y.append(data_scaled.iloc[i+sequence_length]['Close'])  # Predict the 'Close' price
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

# Step 3: Define the DRNN structure using TensorFlow
class DRNN(Model):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(DRNN, self).__init__()
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        
        # Hidden Layer 1 (Diagonal Recurrent)
        self.hidden1 = SimpleRNN(hidden1_size, return_sequences=True, return_state=True)
        
        # Hidden Layer 2 (Diagonal Recurrent)
        self.hidden2 = SimpleRNN(hidden2_size, return_sequences=False, return_state=True)
        
        # Output Layer
        self.output_layer = Dense(output_size, activation='linear')
        
    def call(self, inputs, states=None):
        # Get batch size dynamically
        batch_size = tf.shape(inputs)[0]
        
        # Initialize states if not provided
        if states is None:
            states = [tf.zeros((batch_size, self.hidden1_size)), 
                     tf.zeros((batch_size, self.hidden2_size))]
        
        # Forward pass through hidden layers
        hidden1_output, hidden1_state = self.hidden1(inputs, initial_state=states[0])
        hidden2_output, hidden2_state = self.hidden2(hidden1_output, initial_state=states[1])
        
        # Output layer
        output = self.output_layer(hidden2_output)
        
        return output, [hidden1_state, hidden2_state]

# Step 4: Define the SOM (Self-Organizing Map) for weight initialization
class SOM(Layer):
    def __init__(self, map_size, input_dim):
        super(SOM, self).__init__()
        self.map_size = map_size
        self.input_dim = input_dim
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(self.map_size, self.input_dim),
            initializer='random_normal',
            trainable=True,
            name='som_weights'
        )
        
    def call(self, inputs):
        # Reshape 3D input to 2D if needed
        if len(inputs.shape) == 3:
            inputs = inputs[:, -1, :]
            
        # Find the winning neuron (Best Matching Unit)
        distances = tf.norm(self.W - inputs[:, None, :], axis=2)
        winner = tf.argmin(distances, axis=1)
        
        # Return the weights for DRNN initialization
        return self.W

# Step 5: Define the RBM (Restricted Boltzmann Machine) for output layer training
class RBM(Layer):
    def __init__(self, visible_dim, hidden_dim):
        super(RBM, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(self.visible_dim, self.hidden_dim),
            initializer='random_normal',
            trainable=True
        )
        self.visible_bias = self.add_weight(
            shape=(self.visible_dim,),
            initializer='zeros',
            trainable=True
        )
        self.hidden_bias = self.add_weight(
            shape=(self.hidden_dim,),
            initializer='zeros',
            trainable=True
        )
        
    def call(self, inputs):
        # Positive phase: Compute hidden probabilities
        hidden_prob = tf.sigmoid(tf.matmul(inputs, self.W) + self.hidden_bias)
        hidden_state = tf.nn.relu(tf.sign(hidden_prob - tf.random.uniform(tf.shape(hidden_prob))))
        
        # Negative phase: Reconstruct visible layer
        visible_prob = tf.sigmoid(tf.matmul(hidden_state, tf.transpose(self.W)) + self.visible_bias)
        visible_state = tf.nn.relu(tf.sign(visible_prob - tf.random.uniform(tf.shape(visible_prob))))
        
        # Contrastive Divergence: Update weights and biases
        learning_rate = 0.01
        positive_association = tf.matmul(tf.transpose(inputs), hidden_prob)
        negative_association = tf.matmul(tf.transpose(visible_state), hidden_state)
        self.W.assign_add(learning_rate * (positive_association - negative_association))
        self.visible_bias.assign_add(learning_rate * tf.reduce_mean(inputs - visible_state, axis=0))
        self.hidden_bias.assign_add(learning_rate * tf.reduce_mean(hidden_prob - hidden_state, axis=0))
        
        return visible_state

# Step 6: Define the HDL-DRNNC
class HDL_DRNNC(Model):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(HDL_DRNNC, self).__init__()
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        
        # Initialize layers
        self.drnn = DRNN(input_size, hidden1_size, hidden2_size, output_size)
        self.som = SOM(map_size=hidden1_size, input_dim=input_size)
        self.rbm = RBM(visible_dim=output_size, hidden_dim=output_size)
        
    def call(self, inputs):
        # Initialize DRNN weights using SOM
        som_weights = self.som(inputs)
        # The correct way to access and set SimpleRNN weights
        kernel_weights = tf.reshape(som_weights, (self.input_size, self.hidden1_size))
        self.drnn.hidden1.cell.kernel = kernel_weights
        
        # Forward pass through DRNN
        output, states = self.drnn(inputs)
        
        # Train output layer using RBM
        rbm_output = self.rbm(output)
        
        return rbm_output

# Step 7: Load and preprocess the data
data = fetch_xlk_data()
sequence_length = 60  # Use past 60 time steps to predict the next time step
X, y, scaler = preprocess_data(data, sequence_length)

# Split the data into training and testing sets
train_ratio = 0.8  # 80% for training, 20% for testing
split_index = int(len(X) * train_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 8: Define model parameters
input_size = X_train.shape[2]  # Number of features (e.g., Open, High, Low, Close, Volume)
hidden1_size = 100
hidden2_size = 100
output_size = 1  # Predict the 'Close' price

# Step 9: Create and compile the HDL-DRNNC model
hdl_drnc = HDL_DRNNC(input_size, hidden1_size, hidden2_size, output_size)
hdl_drnc.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Step 10: Train the model
hdl_drnc.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))

# Step 11: Test the model
predictions = hdl_drnc(X_test)
predictions = predictions.numpy()

# Inverse transform the predictions to original scale
predictions = scaler.inverse_transform(np.concatenate([X_test[:, -1, :-1], predictions.reshape(-1, 1)], axis=1))[:, -1]
actual = scaler.inverse_transform(np.concatenate([X_test[:, -1, :-1], y_test.reshape(-1, 1)], axis=1))[:, -1]

# Step 12: Compare predictions to actual prices
comparison = pd.DataFrame({'Actual': actual, 'Predicted': predictions})
print(comparison)