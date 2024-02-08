import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)

def train_linear_regression_model(X, y):
    """Train linear regression model"""
    model = LinearRegression().fit(X, y)
    return model

def predict_price_linear_regression(model, date):
    """Predict price using linear regression model"""
    return model.predict(date)

def prepare_data_for_neural_network(X, y, predict_func):
    """Prepare data for neural network"""
    X_normalized = X / np.amax(X, axis=0)
    date_features = np.array([predict_func(date) for date in X_normalized[:, 0]]).reshape(-1, 1)
    X_with_regression = np.concatenate((X_normalized, date_features), axis=1)
    return X_with_regression, y

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    return x * (1 - x)

def initialize_parameters(input_size, hidden_size, output_size):
    """Initialize weights and biases for neural network"""
    weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
    weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
    biases_hidden = np.random.uniform(size=(1, hidden_size))
    bias_output = np.random.uniform(size=(1, output_size))
    return weights_input_hidden, weights_hidden_output, biases_hidden, bias_output

def forward_propagation(X, weights_input_hidden, weights_hidden_output, biases_hidden, bias_output):
    """Perform forward propagation in neural network"""
    hidden_layer_input = np.dot(X, weights_input_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)
    return hidden_layer_output, output

def backward_propagation(X, y, hidden_layer_output, output, weights_hidden_output):
    """Perform backward propagation in neural network"""
    error = y - output
    d_output = error * sigmoid_derivative(output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)
    return d_output, d_hidden

def update_weights_and_biases(X, hidden_layer_output, d_output, d_hidden, weights_input_hidden, weights_hidden_output, biases_hidden, bias_output, learning_rate):
    """Update weights and biases using gradient descent"""
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    biases_hidden += np.sum(d_hidden, axis=0) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_output += np.sum(d_output, axis=0) * learning_rate
    return weights_input_hidden, weights_hidden_output, biases_hidden, bias_output

def train_neural_network(X, y, input_size, hidden_size, output_size, learning_rate, epochs, predict_func):
    """Train the neural network"""
    weights_input_hidden, weights_hidden_output, biases_hidden, bias_output = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        hidden_layer_output, output = forward_propagation(X, weights_input_hidden, weights_hidden_output, biases_hidden, bias_output)
        d_output, d_hidden = backward_propagation(X, y, hidden_layer_output, output, weights_hidden_output)
        weights_input_hidden, weights_hidden_output, biases_hidden, bias_output = update_weights_and_biases(X, hidden_layer_output, d_output, d_hidden, weights_input_hidden, weights_hidden_output, biases_hidden, bias_output, learning_rate)
    
    return weights_input_hidden, weights_hidden_output, biases_hidden, bias_output

def predict_price_neural_network(X, weights_input_hidden, weights_hidden_output, biases_hidden, bias_output):
    """Predict price using trained neural network"""
    _, output = forward_propagation(X, weights_input_hidden, weights_hidden_output, biases_hidden, bias_output)
    return output

# Main function
def main():
    # Load data
    data = load_data("FINAL DATA.csv")
    y_nn = data["Price"].values
    x_nn = data["Date"].values.reshape(-1, 1)

    # Train linear regression model
    model_lr = train_linear_regression_model(x_nn, y_nn)

    # Function to predict price using linear regression model
    predict_func = lambda date: predict_price_linear_regression(model_lr, date)

    # Prepare data for neural network
    X_nn, y_nn = prepare_data_for_neural_network(x_nn, y_nn, predict_func)

    # Neural network parameters
    input_size = 3
    hidden_size = 4
    output_size = 1
    learning_rate = 0.1
    epochs = 100000

    # Train the neural network
    weights_input_hidden, weights_hidden_output, biases_hidden, bias_output = train_neural_network(X_nn, y_nn, input_size, hidden_size, output_size, learning_rate, epochs, predict_func)

    # Predict price using trained neural network
    predictions = predict_price_neural_network(X_nn, weights_input_hidden, weights_hidden_output, biases_hidden, bias_output)

    # Print neural network predictions
    print("\nNeural Network Predictions:")
    for i in range(len(x_nn)):
        print("Date:", x_nn[i][0], "Predicted Price (Neural Network):", predictions[i, 0])

    # Plot the results
    plt.figure(figsize=(6,4))
    plt.scatter(x_nn, y_nn, c="purple", label="Data")
    plt.plot(x_nn, predict_price_linear_regression(model_lr, x_nn), c="r", label="Linear Regression")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Price by Date")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
