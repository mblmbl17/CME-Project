import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("US Corn Futures Historical Data (1).csv")
y = data["Price"].values
x = data["Change %"].values

x = x.reshape(-1,1)

model = LinearRegression().fit(x, y)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

x_predict = 43

# Print out the linear equation and r squared value
prediction = model.predict([[x_predict]])
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")
print(f"Prediction when x is {x_predict}: {prediction}")

# Predict the the blood pressure of someone who is 43 years old.
# Print out the prediction
# Create the model in matplotlib and include the line of best fit
plt.figure(figsize=(6,4))

plt.scatter(x, y, c="purple")
plt.scatter(x_predict, prediction, c="blue")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Price by Date")

plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

plt.legend()
plt.show()