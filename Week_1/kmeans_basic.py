# Project: Basic K-Means with scikit-learn
# Cluster simple 2D data and understand how sklearn K-Means works.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data
X = np.array([
    [15, 39],
    [16, 81],
    [17, 6],
    [18, 77],
    [19, 40],
    [20, 76]
])

# Model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Results
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster labels:", labels)
print("Centroids:\n", centroids)
