# Project: Basic K-Means from Scratch

"""Goal

Cluster customers based on:

Annual income

Spending score"""

# Step 1: Sample data
import numpy as np

X = np.array([
    [15, 39],
    [16, 81],
    [17, 6],
    [18, 77],
    [19, 40],
    [20, 76]
])

# Step 2: Choose K and initialize centroids
K = 2
centroids = X[np.random.choice(len(X), K, replace=False)]


# Step 3: Assign points
def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [np.linalg.norm(x - c) for c in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)

# Step 4: Update centroids
def update_centroids(X, clusters, K):
    new_centroids = []
    for k in range(K):
        points = X[clusters == k]
        new_centroids.append(points.mean(axis=0))
    return np.array(new_centroids)

# Step 5: Iterate
for _ in range(10):
    clusters = assign_clusters(X, centroids)
    centroids = update_centroids(X, clusters, K)

print("Final centroids:", centroids)
