# Project: Breast Cancer Diagnosis: PCA & Logistic Regression Pipeline

"""
Description: Developed an end-to-end machine learning pipeline using Scikit-Learn 
to classify breast cancer tumors with over 99% accuracy. Implemented strict 
Pipeline architecture to prevent data leakage during scaling and used PCA to 
compress 30 complex medical features into a highly separable 2D space for 
visualization.
"""

# --- Imports ---
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# --- Load Data ---
X, y = load_breast_cancer(return_X_y=True)

# Load feature names for better labeling
breast = load_breast_cancer()
feature_names = breast.feature_names

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Pipeline with PCA ---
my_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("model", LogisticRegression())
])

# --- Train ---
my_pipeline.fit(X_train, y_train)

# --- Predict ---
y_pred = my_pipeline.predict(X_test)

# --- Accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# --- PCA info ---
pca = my_pipeline.named_steps['pca']
print("Original Features:", X.shape[1])
print("Reduced Features:", pca.n_components_)
print("Explained Variance:", pca.explained_variance_ratio_)

# --- Model Evaluation ---
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(cm, display_labels=breast.target_names)
disp.plot(cmap="Blues")
plt.show()

# Display the classification report
print(classification_report(y_test, y_pred, target_names=breast.target_names))

# --- Data Visualization ---
# Create a side-by-side comparison of the data before and after PCA
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=80)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Original Data (Training Set)')
plt.colorbar(ticks=[0, 1], label='Target Classes')

plt.subplot(1, 2, 2)

# Grab the scaler
scaler = my_pipeline.named_steps['scaler']

# Scale the data first, then apply PCA
X_train_scaled = scaler.transform(X_train)
X_train_pca = pca.transform(X_train_scaled)

plt.scatter(X_train_pca[:, 0], X_train_pca[:,1], c=y_train, cmap='coolwarm', edgecolors='k', s=80)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result (Training Set)')
plt.colorbar(ticks=[0, 1], label='Target Classes')

plt.tight_layout()
plt.show()