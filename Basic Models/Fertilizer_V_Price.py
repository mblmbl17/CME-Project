import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

fig, ax1 = plt.subplots()

data = pd.read_csv("Basic Models/FINAL DATA.csv")

data2 = pd.read_csv("Basic Models/ModifiedImportData.csv")

#only take data from 2004 to 2012
x_rows = [0,1,2,3,4,5,6,7,8]
y2_rows = [11,12,13,14,15,16,17,18,19]
updatedData = data[data.index.isin(y2_rows)]
updatedData2 = data2[data2.index.isin(x_rows)]
x = updatedData2["Year"].values
y1 = updatedData2["Dollars"].values
y2 = updatedData["Price2"].values

ax1.plot(x, y1, color = "blue")
ax2 = ax1.twinx()
ax2.plot(x, y2, color = "red")

ax1.spines['right'].set_position(('axes',1.15))

ax1.set_xlabel("Year")
ax1.set_ylabel("Dollars", color = "blue")
ax2.set_ylabel("Fertilizer Price", color = "red")

ax1.tick_params(axis='y', colors = "blue")
ax2.tick_params(axis='y', colors = "red")

ax2.spines['left'].set_color("blue")
ax2.spines['right'].set_color("red")

plt.show()

