# Project: Customer Segmentation with Real CSV Dataset
# Use a CSV dataset to cluster customers by Annual Income and Spending Score.

# Import required libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Mall_Customers.csv"
data = pd.read_csv(url)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Elbow Method to find K
inertia = []
for k in range(1, 11):
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(X)
  inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Cost (Inertia)")
plt.title("Elbow Method")
plt.show()

# Apply K-Means
K = 5 # from elbow
kmeans = KMeans(n_clusters=K, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.show()