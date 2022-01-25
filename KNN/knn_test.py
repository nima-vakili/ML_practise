#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:36:24 2022

@author: nvakili
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_predict):
    accuracy = np.sum(y_true == y_predict)/len(y_true)
    return accuracy

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from knn import KNN
k = 3
clf = KNN(3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy_ = accuracy(y_test, predictions) 
print("KNN classification accuracy", accuracy_)