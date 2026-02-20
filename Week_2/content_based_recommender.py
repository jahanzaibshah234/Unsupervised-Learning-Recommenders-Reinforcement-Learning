# Project: Content-Based Recommender (TensorFlow)
# Recommend items using known item features and learned user preferences.

"""Description: A TensorFlow-based Content-Based Recommender System that learns user 
preference vectors to predict star ratings based on fixed item features. It utilizes a masked cost 
function to handle sparse data and Cosine Similarity to recommend items with similar attributes."""

# Import libraries
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# Data
# User interaction (ratings / likes)
Y = tf.constant([
    [5, 4, 0, 0, 3],
    [0, 0, 4, 3, 0],
    [5, 0, 0, 4, 4],
    [0, 3, 5, 0, 0]
], dtype=tf.float32)

# Create Mask (1 for rated, 0 for unrated)
R = tf.cast(Y > 0, tf.float32)

# Get dimensions from Y automatically
num_users, num_items = Y.shape
num_features = 6 # genre tags
lambda_ = 0.1

# Parameters
# X: Item Features (Fixed Facts)
X = tf.constant(np.random.rand(num_items, num_features), dtype=tf.float32)

# w: User Preferences (Learned Variables)
w = tf.Variable(tf.random.normal((num_users, num_features)))

# Cost function
def cost_func():
    preds = tf.matmul(w, X, transpose_b=True)
    error = (preds - Y) * R
    reg = (lambda_ / 2) * tf.reduce_sum(w**2)
    return 0.5 * tf.reduce_sum(error**2) + reg
  
# Training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

print("Training Content-Based Model...")
for i in range(500):
    with tf.GradientTape() as tape:
        cost_value = cost_func()

        # Calculate gradients
        grads = tape.gradient(cost_value, [w])

        # Apply gradients
        optimizer.apply_gradients(zip(grads, [w]))

        if i % 50 == 0:
            print(f"Iteration {i}, cost = {cost_value.numpy():.2f}")
            
print("Training Complete!")

# Recommendations
scores = tf.matmul(w, X, transpose_b=True)

print("\n--- Recommendations ---")
for i in range(num_users):
    ranked_items = tf.argsort(scores[i], direction="DESCENDING")
    print(f"User {i} → Recommended items:", ranked_items.numpy())

# Similarity Analysis
print("\n--- Item Similarity Analysis ---")
similarity = cosine_similarity(X.numpy())
similar_indices = np.argsort(similarity[0])[::-1]
print(f"Items most similar to Item 0: {similar_indices}")
print(f"(Scores: {similarity[0][similar_indices]})")
