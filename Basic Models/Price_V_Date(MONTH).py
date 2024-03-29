import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
data = pd.read_csv("Basic Models/FINAL DATA.csv")
selected_rows = list(range(238))
updatedData = data[data.index.isin(selected_rows)]
y = updatedData["Price(Month)"].values
x = updatedData["Month"].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)
 
xtrain = xtrain.reshape(-1, 1)

model = LinearRegression().fit(xtrain, ytrain)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)

# Print out the linear equation and r squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

# Predict the the blood pressure of someone who is 43 years old.
# Print out the prediction
# Create the model in matplotlib and include the line of best fit

xtest = xtest.reshape(-1, 1)
predict = model.predict(xtest)
predict = np.around(predict, 2)

plt.figure(figsize=(6,4))

plt.scatter(xtrain, ytrain, c="purple", label="Training Data")
plt.scatter(xtest, ytest, c="blue", label="Testing Data")
plt.xlabel("Date (Month)")
plt.ylabel("Price")
plt.title("Price by Date (Month)")

plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

plt.legend()
plt.show()
joblib.dump(model, 'Neuralnet.py')