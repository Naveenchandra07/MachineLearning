# -*- coding: utf-8 -*-
"""MultipleLinearRegression.py"""

#Importing required modules
import pandas as pd
import numpy as np

#Importing Dataset
dataset = pd.read_csv('50_Startups.csv')

#Extracting features and labels
y = dataset.iloc[:,-1]
X = dataset.iloc[:, :-1]

#Encoding Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Training Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting test results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
y_test = y_test.to_numpy()
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))