import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
data = pd.read_csv("Basic Models/FINAL DATA.csv")
selected_rows = [0,1,2,3,4,5,6,7,8,9,10,11]
selected_rows_2023 = [2,3,4,5,6,7,8,9,10,11]
updatedData = data[data.index.isin(selected_rows)]
updatedData_2023 = data[data.index.isin(selected_rows_2023)]
y1 = updatedData_2023["2023"].values
y2 = updatedData["2022"].values
y3 = updatedData["2021"].values
y4 = updatedData["2020"].values
y5 = updatedData["2019"].values
y6 = updatedData["2018"].values
y7 = updatedData["2017"].values
y8 = updatedData["2016"].values
y9 = updatedData["2015"].values
y10 = updatedData["2014"].values
y11 = updatedData["2013"].values
y12 = updatedData["2012"].values
y13 = updatedData["2011"].values
y14 = updatedData["2010"].values
y15 = updatedData["2009"].values
y16 = updatedData["2008"].values
y17 = updatedData["2007"].values
y18 = updatedData["2006"].values
y19 = updatedData["2005"].values
y20 = updatedData["2004"].values

x = updatedData["Months"].values

final_data = {
    'x' : x,
    'y2' : y2,
    'y3' : y3,
    'y4' : y4,
    'y5' : y5,
    'y6' : y6,
    'y7' : y7,
    'y8' : y8,
    'y9' : y9,
    'y10' : y10,
    'y11' : y11,
    'y12' : y12,
    'y13' : y13,
    'y14' : y14,
    'y15' : y15,
    'y16' : y16,
    'y17' : y17,
    'y18' : y18,
    'y19' : y19,
    'y20' : y20
}

df = pd.DataFrame(final_data)

print(df)

plt.plot(df['x'], df['y2'], label = "2022")
plt.plot(df['x'], df['y3'], label = "2021")
plt.plot(df['x'], df['y4'], label = "2020")
plt.plot(df['x'], df['y5'], label = "2019")
plt.plot(df['x'], df['y6'], label = "2018")
plt.plot(df['x'], df['y7'], label = "2017")
plt.plot(df['x'], df['y8'], label = "2016")
plt.plot(df['x'], df['y9'], label = "2015")
plt.plot(df['x'], df['y10'], label = "2014")
plt.plot(df['x'], df['y11'], label = "2013")
plt.plot(df['x'], df['y12'], label = "2012")
plt.plot(df['x'], df['y13'], label = "2011")
plt.plot(df['x'], df['y14'], label = "2010")
plt.plot(df['x'], df['y15'], label = "2009")
plt.plot(df['x'], df['y16'], label = "2008")
plt.plot(df['x'], df['y17'], label = "2007")
plt.plot(df['x'], df['y18'], label = "2006")
plt.plot(df['x'], df['y19'], label = "2005")
plt.plot(df['x'], df['y20'], label = "2004")

plt.show()

plt.xlabel('Month')
plt.ylabel('Price')
plt.title("Price V Date (Month)")

plt.legend()
plt.grid(True)

# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)
 
# xtrain = xtrain.reshape(-1, 1)

# model = LinearRegression().fit(xtrain, ytrain)

# # Find the coefficient, bias, and r squared values. 
# # Each should be a float and rounded to two decimal places. 
# coef = round(float(model.coef_), 2)
# intercept = round(float(model.intercept_), 2)
# r_squared = model.score(xtrain, ytrain)

# # Print out the linear equation and r squared value
# print(f"Model's Linear Equation: y = {coef}x + {intercept}")
# print(f"R Squared value: {r_squared}")

# # Predict the the blood pressure of someone who is 43 years old.
# # Print out the prediction
# # Create the model in matplotlib and include the line of best fit

# xtest = xtest.reshape(-1, 1)
# predict = model.predict(xtest)
# predict = np.around(predict, 2)



# print("\nTesting Linear Model with Testing Data:")
# for index in range(len(xtest)):
#    actual = ytest[index]
#    predicted_y = predict[index]
#    x_coord = xtest[index]
#    print("x value:", float(x_coord), "Predicted y value:", predicted_y, "Actual y value:", actual)

# plt.figure(figsize=(6,4))

# plt.scatter(xtrain, ytrain, c="purple", label="Training Data")
# plt.scatter(xtest, ytest, c="blue", label="Testing Data")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.title("Price by Date")

# plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

# plt.legend()
# plt.show()
# joblib.dump(model, 'Neuralnet.py')
