import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("FINAL DATA.csv")
y = data["Price"].values
x = data["Date"].values

x = x.reshape(-1,1)

model = LinearRegression().fit(x, y)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

# Print out the linear equation and r squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

# Predict the the blood pressure of someone who is 43 years old.
# Print out the prediction
# Create the model in matplotlib and include the line of best fit
plt.figure(figsize=(6,4))

plt.scatter(x, y, c="purple")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Price by Date")

plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

plt.legend()
plt.show()