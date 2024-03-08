import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class PriceVDateDay:
    def __init__(self,x=[],y=[])-> None:
        self.day=x
        self.price=y
       
        xtrain, xtest, ytrain, ytest = train_test_split(self.day, self.price, test_size=.2)

        xtrain = xtrain.reshape(-1, 1)

        self.model = LinearRegression().fit(xtrain, ytrain)

        # Find the coefficient, bias, and r squared values. 
        # Each should be a float and rounded to two decimal places. 
        #coef = round(float(model.coef_), 2)
        #intercept = round(float(model.intercept_), 2)
        #r_squared = model.score(xtrain, ytrain)

        # Print out the linear equation and r squared value
        #print(f"Model's Linear Equation: y = {coef}x + {intercept}")
        #print(f"R Squared value: {r_squared}")

        # Predict the the blood pressure of someone who is 43 years old.
        # Print out the prediction
        # Create the model in matplotlib and include the line of best fit

        xtest = xtest.reshape(-1, 1)
        predict1 = self.model.predict(xtest)
        predict1 = np.around(predict1, 2)

        # print("\nTesting Linear Model with Testing Data:")
        # for index in range(len(xtest)):
        #     actual = ytest[index]
        #     predicted_y = predict[index]
        #     x_coord = xtest[index]
        #     print("x value:", float(x_coord), "Predicted y value:", predicted_y, "Actual y value:", actual)
    def prediction_day(self,day)->float:
        prediction = self.model.predict(day)
        return prediction

        # plt.figure(figsize=(6,4))

        # plt.scatter(xtrain, ytrain, c="purple", label="Training Data")
        # plt.scatter(xtest, ytest, c="blue", label="Testing Data")
        # plt.xlabel("Date")
        # plt.ylabel("Price")
        # plt.title("Price by Date (Daily)")

        # plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

        # plt.legend()
        # plt.show()
        # print("The predicted day:" + prediction_day(40000))