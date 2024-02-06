import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("FINAL DATA.csv")

data2 = pd.read_csv("ModifiedImportData.csv")
x = data2["Year"].values
y1 = data2["Dollars"].values
y2 = data["Price"].values

print(x)
print(y1)
print(y2)

plt.plot(x, y1)
plt.xlabel("Year")
plt.ylabel("Dollars")
plt.show()
