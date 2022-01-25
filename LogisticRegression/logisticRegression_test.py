#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:28:44 2022

@author: nvakili
"""

import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from logisticRegression import LogisticRegression

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred)/len(y_true)

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

regressor = LogisticRegression(lr=0.0001, n_iter=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("LR classification accuracy:", accuracy(y_test, predictions))
