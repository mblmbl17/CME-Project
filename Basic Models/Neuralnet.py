import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load data
def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)

# Prepare data
def prepare_data(data):
    X = data["Date"].values.reshape(-1, 1)
    y = data["Price"].values.reshape(-1, 1)
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)
    return X_scaled, y_scaled, scaler

# Build and train neural network model
def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train neural network
def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return history

# Make predictions
def make_predictions(model, X, scaler):
    predictions_scaled = model.predict(X)
    predictions = scaler.inverse_transform(predictions_scaled)
    return predictions

# Plot results
def plot_results(X, y, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, predictions, color='red', label='Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Prediction with Neural Network')
    plt.legend()
    plt.show()

def main():
    # Load data
    data = load_data("FINAL DATA.csv")
    
    # Prepare data
    X, y, scaler = prepare_data(data)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build neural network model
    model = build_model(input_shape=X_train.shape[1])
    
    # Train model
    train_model(model, X_train, y_train, epochs=100, batch_size=32)
    
    # Make predictions
    predictions = make_predictions(model, X, scaler)
    
    # Plot results
    plot_results(X, data["Price"].values, predictions)

if __name__ == "__main__":
    main()
