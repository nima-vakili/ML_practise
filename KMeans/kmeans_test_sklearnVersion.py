#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:20:43 2022

@author: nvakili
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeans(n_clusters=clusters).fit(X)
y_pred = k.predict(X)

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(X[:, 0], X[:, 1], c=y_pred)

for point in k.cluster_centers_:
    ax.scatter(*point, marker="x", color="black", linewidth=2)
