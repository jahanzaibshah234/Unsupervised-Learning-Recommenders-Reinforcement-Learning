# Project: One-Class SVM (Classic anomaly detector)
# Learn boundary-based anomaly detection.

"""Implements One-Class SVM for Novelty Detection. The model is trained purely on 
'normal' data to learn a decision boundary (using an RBF kernel) 
and subsequently identifies outliers in a separate test set."""

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

# Generate data
X_train = np.random.normal(0, 1, size=(500, 2))
X_test = np.vstack([
    np.random.normal(0, 1, size=(200, 2)),
    np.random.normal(5, 1, size=(20, 2))
])

# Train model
model = OneClassSVM(
    kernel="rbf",
    gamma=0.1,
    nu=0.05
)

model.fit(X_train)
preds = model.predict(X_test)

# Visualize
plt.scatter(X_test[:, 0], X_test[:, 1], c=preds)
plt.title("One-Class SVM Anomaly Detection")
plt.show()
