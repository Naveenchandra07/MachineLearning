# -*- coding: utf-8 -*-
"""PolynomialLR_PositionSalaries.py"""

#Import required libraries
import pandas as pd
import numpy as np

#Import Dataset
dataset = pd.read_csv('/content/drive/My Drive/Colab Notebooks/PolynomialLinearRegression/PositionSalaries/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Training Linear Regression Model
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

#Training Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizing Linear Regression
import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'red')
plt.scatter(X, linear_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing Polynomial Linear Regression
import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'red')
plt.scatter(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result using Linear Regression
X_test = [[6.5]]
linear_regressor.predict(X_test)

#Predicting new result using Polynomial Linear Regression
X_poly_test = poly_reg.fit_transform([[6.5]])
lin_reg_2.predict(X_poly_test)