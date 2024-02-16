import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("Basic Models/FINAL DATA.csv")

data2 = pd.read_csv("Basic Models/ModifiedImportData.csv")

#only take data from 2004 to 2012
x_rows = [0,1,2,3,4,5,6,7,8]
y2_rows = [11,12,13,14,15,16,17,18,19]
updatedData = data[data.index.isin(y2_rows)]
updatedData2 = data2[data2.index.isin(x_rows)]
x = updatedData2["Dollars"].values
y = updatedData["Price2"].values

print(x)
print(y)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)

print(xtrain)
print(xtest)
print(ytrain)
print(ytest)
xtrain = xtrain.reshape(-1, 1)
print(xtrain)
model = LinearRegression().fit(xtrain, ytrain)

coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)

print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

xtest = xtest.reshape(-1, 1)
predict = model.predict(xtest)
predict = np.around(predict, 2)

# print("\nTesting Linear Model with Testing Data:")
# for index in range(len(xtest)):
#    actual = ytest[index]
#    predicted_y = predict[index]
#    x_coord = xtest[index]
#    print("x value:", float(x_coord), "Predicted y value:", predicted_y, "Actual y value:", actual)

plt.figure(figsize=(6,4))

plt.scatter(xtrain, ytrain, c="purple", label="Training Data")
plt.scatter(xtest, ytest, c="blue", label="Testing Data")
plt.xlabel("Dollars")
plt.ylabel("Fertilizer Price")
plt.title("Dollars v Fertilizer Price")

plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

plt.legend()
plt.show()
