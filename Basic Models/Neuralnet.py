import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler

def train_neural_network(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the neural network model
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
    model.fit(X_train, y_train)

    
   # loaded_model = get_trained_model()
    # Evaluate the model
    loaded_model = joblib.load("model")
    test_score = model.score(X_test, y_test)
    print(f"Test score: {test_score}")

def predict_future_prices(future_data):
    # Load the saved model from the file
    loaded_model = joblib.load('Price_V_Date(YEAR).py')

    # Make predictions for future prices
    future_predictions = loaded_model.predict(future_data)

    # Print the predicted prices for the future
    print("Predicted prices for the future:")
    print(future_predictions)

def main():
    # Assuming you have historical price data stored in X and y arrays
    # X contains features (e.g., date, volume, etc.), and y contains target prices
    X = np.load('')  # Replace with your actual data
    y = np.array([1, 2])  # Replace with your actual data
    data = pd.read_csv()
    # Train the neural network
    train_neural_network(X, y)


  

if __name__ == "__main__":
    main()
