# Project: Image Compression using K-Means
# Compress an image by reducing the number of colors using K-Means clustering.

# Import required libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import data

# Load image
img = data.coffee()

# Normalize
img_data = img / 255.0 
w, h, d = img_data.shape
pixels = img_data.reshape(-1, 3) # flatten

# K-Means
K = 16 # number of colors
kmeans = KMeans(n_clusters=K, random_state=42)
labels = kmeans.fit_predict(pixels)
centroids = kmeans.cluster_centers_

# Recreate compressed image
compressed_img = centroids[labels].reshape(w, h, d)

# Display
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Compressed Image")
plt.imshow(compressed_img)
plt.axis("off")
plt.show()
