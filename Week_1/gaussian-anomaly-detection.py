# Project: Gaussian Anomaly Detection
# Detect anomalies by modeling normal data distribution.

"""Description: A Python script implementing Gaussian Anomaly Detection from scratch using NumPy. 
It learns a statistical pattern from normal data and flags outliers in a test set."""

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Normal behavior
X_train = np.random.normal(loc=50, scale=5, size=(1000, 1))

# Some anomalies
X_test = np.concatenate([
    np.random.normal(loc=50, scale=5, size=(200, 1)),
    np.random.normal(loc=80, scale=2, size=(10, 1))  # anomalies 
])

# Fit Gaussian model
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)

# Compute probability & detect anomalies
def gaussian_model(x, mu, sigma):
  return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x - mu)**2 / (2*sigma**2))

probs = gaussian_model(X_test, mu, sigma)

epsilon = 1e-4
anomalies = X_test[probs < epsilon]

print(anomalies)

# Plot data and anomaly
plt.hist(X_train, bins=50, density=True, alpha=0.5, label="Normal Data"); # <--- Semicolon hides the output

plt.scatter(anomalies, np.zeros_like(anomalies), color='red', s=100, zorder=5, label="Anomalies")

plt.title("Gaussian Anomaly Detection")
plt.xlabel("Data Values")
plt.xlim(30, 90) 
plt.legend()
plt.show()