#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:13:20 2022

@author: nvakili
"""

import numpy as np

class Perceptron:
    def __init__(self, lr=.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
      
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        y_ = np.array([1 if i>0 else 0 for i in y])
        
        for _ in range(self.n_iter):
            for ind, x in enumerate(X):
        
                y_model = np.dot(x, self.weights) + self.bias
                y_pred = self._sign(y_model)
                
                update = self.lr * (y_[ind]-y_pred)
                self.weights += update * x
                self.bias += update
                
    def predict(self, X):
         y_model = np.dot(X, self.weights) + self.bias
         y_pred = self._sign(y_model)      
         return y_pred
     
    def _sign(self, x):
        return np.where(x>=0, 1, 0)