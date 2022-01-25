#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:36:48 2022

@author: nvakili
"""

import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred)/len(y_true)

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


print("LR classification accuracy:", accuracy(y_test, predictions))