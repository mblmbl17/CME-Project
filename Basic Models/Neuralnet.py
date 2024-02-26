import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler

def train_neural_network(X, y, future_data=None):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the neural network model
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
    model.fit(X_train, y_train)

    # Make p--redictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    test_score = model.score(X_test, y_test)
    print(f"Test score: {test_score}")

    # Print predictions
    print("Predictions on testing data:")
    for i in range(len(X_test)):
        print(f"Feature: {X_test[i]}, Actual Price: {y_test[i]}, Predicted Price: {y_pred[i]}")

    # Make predictions on future data if provided
    if future_data is not None:
        future_predictions = model.predict(future_data)
        print("\nPredictions on future data:")
        for i, prediction in enumerate(future_predictions):
            print(f"Feature: {future_data[i]}, Predicted Price: {prediction}")

    # Save the trained model
    joblib.dump(model, "neural_network_model.pkl")

def main():
    # Load data from CSV file
    data = pd.read_csv('Basic Models/FINAL DATA.csv')
    selected_rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    updatedData = data[data.index.isin(selected_rows)]
    y = updatedData["Price2"].values
    X = updatedData["Date2"].values.reshape(-1, 1)  # Reshape to 2D array

    # Train the neural network
    train_neural_network(X, y)

    # Example of future data for prediction
    future_data = np.array([[0.5], [0.6]])  # Replace with your actual future data

    # Make predictions on future data
    train_neural_network(X, y, future_data=future_data)

if __name__ == "__main__":
    main()

