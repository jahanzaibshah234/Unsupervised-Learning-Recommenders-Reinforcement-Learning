# Project: Iris PCA - Dimensionality Reduction & Classification

"""
Description: This project builds a robust, production-ready machine learning pipeline that uses 
Principal Component Analysis (PCA) to safely compress 4-dimensional data into a 2D space without data leakage. 
It trains a Logistic Regression model on the optimized data to classify flower species, proving 
its accuracy through a custom confusion matrix and before-and-after scatter plot visualizations.
"""

# --- Imports ---
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- Load Data ---
X, y = load_iris(return_X_y=True)

# Load feature names for better labeling
iris = load_iris()
feature_names = iris.feature_names

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Apply PCA ---
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Print the percentage of total information retained by the 2 new components
print(pca.explained_variance_ratio_)

# --- Model Training & Prediction ---
model = LogisticRegression()
model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)

# --- Model Evaluation ---
cm = confusion_matrix(y_test, y_pred)

flower_names = ['setosa', 'versicolor', 'virginica']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=flower_names)
disp.plot(cmap="Blues")
plt.show()

# --- Data Visualization ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', s=80)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Original Data')
plt.colorbar(label='Target Classes')

plt.subplot(1, 2, 2)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=80)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result')
plt.colorbar(label='Target Classes')

plt.tight_layout()
plt.show()
