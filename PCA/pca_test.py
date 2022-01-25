#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:47:21 2022

@author: nvakili
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from pca import PCA


data = datasets.load_iris()
X = data.data 
y = data.target

print('X_orig:', X.shape)

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print('X_transformed:', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2, c=y, cmap=plt.cm.get_cmap("viridis", 3))
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()