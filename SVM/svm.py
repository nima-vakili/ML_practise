#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:39:40 2022

@author: nvakili
"""

import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None 
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y<=0, -1, 1)
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            for ind, x in enumerate(X):
                condition = y_[ind] * (np.dot(x, self.weights) - self.bias) >=1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x, y[ind]))
                    self.bias -= self.lr * y[ind]
        def predict(self, X):
            approx = np.dot(X, self.weights) - self.bias
            return np.sign(approx)
        
                
                
                
                
                
                
                
                
                
                
                
                