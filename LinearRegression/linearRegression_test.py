#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 12:52:24 2022

@author: nvakili
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linearRegression import LinearRegression

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    return corr ** 2
    
    
X, y = datasets.make_regression(n_samples=100, n_features=1,
                                noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                test_size=.2,random_state=1234)

regressor = LinearRegression(lr=.03, n_iter=100)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)
error = mse(y_test, predicted)
print('MSE=',error)

accu = r2_score(y_test, predicted)
print('Accuracy=', accu)


y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()




