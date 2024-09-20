# filename: simple_neural_net.py
import numpy as np

# Generate some random training data
np.random.seed(42)
X = np.random.rand(10, 2)  # 10 samples with 2 features
y = np.random.rand(10, 1)   # 10 target values

# Initialize weights and biases
weights_input_to_hidden = np.random.rand(2, 2)  # 2 features to 2 neurons
weights_hidden_to_output = np.random.rand(2, 1)  # 2 neurons to 1 output
bias_hidden = np.random.rand(2)
bias_output = np.random.rand(1)

# Define parameters
learning_rate = 0.01
epochs = 1000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    predicted_output = output_layer_input  # No activation for output layer in regression

    # Calculate the loss
    loss = np.mean((y - predicted_output) ** 2)  # Mean squared error

    # Backpropagation
    output_error = predicted_output - y
    output_delta = output_error  # Since we are not using activation on output

    hidden_layer_error = output_delta.dot(weights_hidden_to_output.T)
    hidden_layer_delta = hidden_layer_error * hidden_layer_output * (1 - hidden_layer_output)  # Derivative of sigmoid

    # Update weights and biases
    weights_hidden_to_output -= hidden_layer_output.T.dot(output_delta) * learning_rate
    bias_output -= np.sum(output_delta, axis=0) * learning_rate
    weights_input_to_hidden -= X.T.dot(hidden_layer_delta) * learning_rate
    bias_hidden -= np.sum(hidden_layer_delta, axis=0) * learning_rate

# Print final weights and biases
print("Final Weights from Input to Hidden Layer:\n", weights_input_to_hidden)
print("Final Weights from Hidden to Output Layer:\n", weights_hidden_to_output)
print("Final Biases for Hidden Layer:\n", bias_hidden)
print("Final Bias for Output Layer:\n", bias_output)
print("Final Predicted Output:\n", predicted_output)
print("Final Loss:\n", loss)