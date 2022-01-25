#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:17:29 2022

@author: nvakili
"""

import numpy as np

class LogisticRegression:
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
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
          
            dw = (1/ n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/ n_samples) * np.sum(y_pred-y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db 
            
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred_calc = [1 if i>.5 else 0 for i in y_pred]
        return y_pred_calc