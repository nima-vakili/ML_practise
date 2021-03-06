#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 12:42:49 2022

@author: nvakili
"""

import numpy as np

class LinearRegression:
    def __init__(self, lr=.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            y_predict = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predict-y))
            db = (1/n_samples) * np.sum(y_predict-y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        y_predict = np.dot(X, self.weights) + self.bias
        return y_predict
        