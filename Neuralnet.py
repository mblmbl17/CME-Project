import numpy as np



def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative (x):
    return x*(1-x)



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size = 2

hidden_size = 2
output_size = 1

weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

biases_hidden = np.random.uniform(size=(1, hidden_size))
bias_output = np.random.uniform(size=(1, output_size))


learning_rate = 0.12
epochs = 5000

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)
    
    # Backpropagation
    error = y - output
    d_output = error * sigmoid_derivative(output)
    
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    biases_hidden += np.sum(d_hidden, axis=0) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_output += np.sum(d_output, axis=0) * learning_rate


print("Final Predictions:")
print(output)