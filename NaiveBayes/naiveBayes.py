#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:07:59 2022

@author: nvakili
"""

import numpy as np

class NaiveBayes:
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._prior = np.zeros(n_classes, dtype=np.float64)
        
        for ind, c in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[ind, :] = X_c.mean(0)
            self._var[ind, :] = X_c.var(0)
            self._prior[ind] = X_c.shape[0]/float(n_samples)
            
    def predict(self, X):
        y_predict = [self._predict(x) for x in X]
        return np.array(y_predict)
    
    def _predict(self, x):
        posteriors = []
        for ind, c in enumerate(self._classes):
            prior = np.log(self._prior[ind])
            posterior = np.sum(np.log(self._pdf(ind, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, ind, x):
        mean = self._mean[ind]
        var = self._var[ind]
        numerator = np.exp(-((x-mean)**2)/(2*var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator/denominator
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            