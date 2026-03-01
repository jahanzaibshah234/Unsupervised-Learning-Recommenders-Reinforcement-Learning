# Project: Handwritten Digit Compression via PCA

"""
Description: This project demonstrates computer vision and dimensionality reduction 
by using Principal Component Analysis (PCA) to compress 64-pixel images of handwritten digits. 
By dynamically commanding the AI to retain 95% of the statistical variance, the pipeline drastically 
reduces file size while successfully reconstructing the images with minimal Mean Squared Error (MSE).
"""

# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Load Data ---
digits = load_digits()
X = digits.data

# ---- Standardize ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Apply PCA ---
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_reduced)

# Print number of components
print("Optimal Components Chosen by AI:", pca.n_components_)

# Print variance ratio
print("Explained Variance:", pca.explained_variance_ratio_)

# --- Visualize original vs reconstructed --- 
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
  axes[0, i].imshow(X_scaled[i].reshape(8, 8), cmap='gray')
  axes[0, i].set_title("Original")
  axes[0, i].axis('off')

  axes[1, i].imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
  axes[1, i].set_title("Reconstructed")
  axes[1, i].axis("off")

plt.show()

# --- Reconstruction error ---
error = np.mean((X_scaled - X_reconstructed) ** 2)
print("Reconstruction MSE:", error)