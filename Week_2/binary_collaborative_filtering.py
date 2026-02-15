# Project: Collaborative Filtering with Binary Feedback (Likes / Clicks)

"""Description: A TensorFlow-based recommender system that uses Matrix Factorization and 
Sigmoid activation to predict the probability of user interactions (Likes/Clicks)."""

# Import libraries
import numpy as np
import tensorflow as tf

# Binary interaction matrix
# 1 = liked / clicked, 0 = no interaction
Y = tf.constant([
    [1, 1, 0, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 0]
], dtype=tf.float32)

num_users, num_items = Y.shape
num_features = 3
lambda_ = 0.1

# Parameters
X = tf.Variable(tf.random.normal((num_items, num_features)))
w = tf.Variable(tf.random.normal((num_users, num_features)))

# Sigmoid prediction
def predict():
    return tf.sigmoid(tf.matmul(w, X, transpose_b=True))

# Binary cross-entropy loss
def cost_function():
    preds = predict()
    bce = -Y * tf.math.log(preds + 1e-8) - (1 - Y) * tf.math.log(1 - preds + 1e-8)
    reg = (lambda_ / 2) * (tf.reduce_sum(w**2) + tf.reduce_sum(X**2))
    return tf.reduce_sum(bce) + reg

# Train
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

print("Training Started...")
for i in range(500):
    with tf.GradientTape() as tape:
        cost_value = cost_function()

    # Calculate gradients
    grad = tape.gradient(cost_value, [w, X])

    # Apply gradients
    optimizer.apply_gradients(zip(grad, [w, X]))

    if i % 50 == 0:
        print(f"Training loss at iteration {i}: Cost {cost_value.numpy():.2f}")

print("Training Complete!")

print("\n--- Detailed Recommendations ---")

items_names = ["Action Movie", "Comedy Movie", "Romance Movie", "Horror Movie"]

# Get final predictions
all_scores = predict()

for i in range(num_users):
    print(f"\nUser {i} Preferences:")

    # Sort items by highest score first
    ranked_ids = tf.argsort(all_scores[i], direction="DESCENDING")
  
    for rank, item_id in enumerate(ranked_ids):
        score = all_scores[i][item_id]
        name = items_names[item_id]

        # Check if user already liked it in the original data
        status = "ALREADY LIKED" if Y[i, item_id] == 1 else "New Recommendation"
        
        # Print formatted output
        print(f"   Rank {rank+1}: {name} (Score: {score:.1%}) -- {status}")
