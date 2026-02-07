# Project: Isolation Forest
# Detect anomalies without assuming Gaussian distribution.

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Create dataset
X = np.random.normal(0, 1, size=(1000, 2))
X_outliers = np.random.uniform(low=-6, high=6, size=(50, 2))

X_all = np.vstack([X, X_outliers])

# Train Isolation Forest
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

preds = model.fit_predict(X_all)

# Visualize anomalies
plt.scatter(X_all[:, 0], X_all[:, 1], c=preds)
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()