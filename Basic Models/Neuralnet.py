import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
Predictor_1 = joblib.load('Price_V_Date.py')
Predictor_2 = joblib.load('Fertilizer_V_Price.py')


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
    Brain = build_Brain(input_shape=X_train.shape[1])
    
    # Train model
    train_model(Brain, X_train, y_train, epochs=100, batch_size=32)
    
    # Make predictions
    predictions = make_predictions(Brain, X, scaler)
    
    # Plot results
    plot_results(X, data["Price"].values, predictions)

if __name__ == "__main__":
    main()
