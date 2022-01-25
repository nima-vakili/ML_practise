#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:55:28 2022

@author: nvakili
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def accuracy(y_true, y_predict):
    accuracy = np.sum(y_true == y_predict)/len(y_true)
    return accuracy


iris = datasets.load_iris()
X, y = iris.data, iris.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap)
plt.show()

nbrs = NearestCentroid()
nbrs.fit(X_train, y_train)
predictions = nbrs.predict(X_test)

accuracy_ = accuracy(y_test, predictions) 
print("KNN classification accuracy", accuracy_)
