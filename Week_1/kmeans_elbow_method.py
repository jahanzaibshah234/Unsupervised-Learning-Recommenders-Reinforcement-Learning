# Project: Visual Clustering + Elbow Method
# Choose K properly using the elbow method

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    [15, 39],
    [16, 81],
    [17, 6],
    [18, 77],
    [19, 40],
    [20, 76]
])

inertia = []

for k in range(1, 7):
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(X)
  inertia.append(kmeans.inertia_)

plt.plot(range(1, 7), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Cost (Inertia)")
plt.title("Elbow Method")
plt.show()