import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

fig, ax1 = plt.subplots()

data = pd.read_csv("Basic Models/FINAL DATA.csv")

data2 = pd.read_csv("Basic Models/ModifiedImportData.csv")
x = data2["Year"].values
y1 = data2["Dollars"].values
y2 = data["Price"].values

ax1.plot(x, y1)
ax2 = ax1.twinx()

print(x)
print(y1)
print(y2)

plt.show()

