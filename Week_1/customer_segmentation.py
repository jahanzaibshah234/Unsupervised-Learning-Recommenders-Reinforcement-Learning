# Project: Customer Segmentation
# Segment customers like a real business problem.

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate customer-like data
X, _ = make_blobs(
    n_samples=300, 
    centers=4, 
    cluster_std=1.2,  
    random_state=42
)

# Train K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="x",
    c="red",
    s=200
)
plt.title("Customer Segmentation")
plt.show()