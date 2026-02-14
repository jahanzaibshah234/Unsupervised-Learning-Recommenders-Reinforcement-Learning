# TensorRec: Collaborative Filtering 

"""Description: A custom-built recommendation system using TensorFlow and Matrix Factorization 
to predict missing user ratings with optimized gradient descent."""

# Import required libraries 
import numpy as np
import tensorflow as tf

# Initialize parameters
num_users = 5
num_items = 4
num_features = 3

# Learnable Parameters (Random initialization)
X = tf.Variable(tf.random.normal((num_items, num_features)))
w = tf.Variable(tf.random.normal((num_users, num_features)))

# Ratings matrix
# 0 indicates a missing rating
Y = tf.constant([
    [5, 4, 0, 0],
    [3, 0, 0, 2],
    [0, 4, 5, 0],
    [0, 0, 4, 3],
    [5, 0, 0, 0]
], dtype=tf.float32)
R = tf.cast(Y > 0, tf.float32)

# Cost function
lambda_ = 0.1  # Regularization parameter

def cost_function():
  pred = tf.matmul(w, X, transpose_b=True)
  J = (pred - Y) * R
  J = 0.5 * tf.reduce_sum(J**2) + (lambda_/2) * (tf.reduce_sum(w**2) + tf.reduce_sum(X**2))
  return J

# Train
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

print("Training Started...")

for i in range(500):
  with tf.GradientTape() as tape:
    cost_value = cost_function()

  # Calculate gradients
  grads = tape.gradient(cost_value, [w, X])

  # Apply gradients
  optimizer.apply_gradients(zip(grads, [w, X]))
  
  if i % 50 == 0:
    print(f"Training loss at iteration {i}: Cost {cost_value.numpy():.2f}")

print("Training Complete!")

# Make Predictions
# Calculate predictions
final_predictions = tf.matmul(w, X, transpose_b=True)

# FORCE values to be between 0 and 5
final_predictions = tf.clip_by_value(final_predictions, 0.0, 5.0)

# Display Results
print("\n--- Final Results ---")
original_Y = Y.numpy()
predicted_Y = final_predictions.numpy()

for i in range(num_users):
  for j in range(num_items):
    # Only print the prediction if the user hadn't rated it originally
    if original_Y[i, j] == 0:
      print(f"User {i} has not seen Item {j}. We predict rating: {predicted_Y[i, j]:.1f} ⭐")
