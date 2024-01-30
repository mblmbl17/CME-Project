import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("FINAL DATA.csv")
y_nn = data["Price"].values
x_nn = data["Date"].values.reshape(-1, 1)

# Train regression model
regression_model = LinearRegression().fit(x_nn, y_nn)

# Extract coefficients
coef_regression = regression_model.coef_[0]
intercept_regression = regression_model.intercept_

# Function to predict price using regression model
def predict_price(date):
    return coef_regression * date + intercept_regression

# Prepare data for neural network
X_nn = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_nn = np.array([[0], [1], [1], [0]])

# Add regression predictions as an additional feature to X_nn
date_features = np.array([predict_price(date) for date in X_nn[:, 0]]).reshape(-1, 1)
X_nn_with_regression = np.concatenate((X_nn, date_features), axis=1)

# Print the new data
print("New Data with Regression Predictions:")
for i in range(len(X_nn_with_regression)):
    print("Date:", X_nn_with_regression[i, 0], "Predicted Price:", X_nn_with_regression[i, 2], "Original Features:", X_nn_with_regression[i, :2])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input_size = 2
hidden_size = 2
output_size = 1

weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

biases_hidden = np.random.uniform(size=(1, hidden_size))
bias_output = np.random.uniform(size=(1, output_size))

learning_rate = .1
epochs = 100000000

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X_nn_with_regression, weights_input_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)

    # Backpropagation
    error = y_nn - output
    d_output = error * sigmoid_derivative(output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    biases_hidden += np.sum(d_hidden, axis=0) * learning_rate
    weights_input_hidden += X_nn_with_regression.T.dot(d_hidden) * learning_rate
    bias_output += np.sum(d_output, axis=0) * learning_rate
print("\nNew Data with Predictions:")
for i in range(len(X_nn_with_regression)):
    
    
    
    print("Date:", X_nn_with_regression[i, 0], "Predicted Price (Regression):", X_nn_with_regression[i, 2], "Original Features:", X_nn_with_regression[i, :2], "Predicted Price (Neural Network):", output[i, 0])
print("\nFinal Predictions:")
print(output)
