# Project: Collaborative Filtering with Mean Normalization

"""Description: A TensorFlow-based recommendation system using Matrix Factorization 
and Mean Normalization to predict missing user ratings with custom Gradient Descent.
"""

# Import libraries
import numpy as np
import tensorflow as tf

# Data
Y = np.array([
    [5, 4, 0, 0],
    [3, 0, 0, 2],
    [0, 4, 5, 0],
    [0, 0, 4, 3],
    [5, 0, 0, 0]
], dtype=np.float32)

R = (Y > 0).astype(np.float32)


num_users, num_items = Y.shape
num_features = 3
lambda_ = 0.1

# Mean Normalization
Y_mean = np.sum(Y, axis=0) / (np.sum(R, axis=0) + 1e-8)
Y_norm = (Y - Y_mean) * R

Y_norm = tf.constant(Y_norm)
R = tf.constant(R)
Y_mean = tf.constant(Y_mean)

# Parameters
X = tf.Variable(tf.random.normal((num_items, num_features)))
w = tf.Variable(tf.random.normal((num_users, num_features)))

# Cost function
def cost_function():
  pred = tf.matmul(w, X, transpose_b=True)
  J = (pred - Y_norm) * R
  J = 0.5 * tf.reduce_sum(J**2) + (lambda_ / 2) * (tf.reduce_sum(w**2) + tf.reduce_sum(X**2))

  return J

# Train
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(500):
  with tf.GradientTape() as tape:
    cost_value = cost_function()

  # Calculate gradients
  grads = tape.gradient(cost_value, [w, X])

  # Apply gradients
  optimizer.apply_gradients(zip(grads, [w, X]))

  if i % 50 == 0:
    print(f"Training loss at iteration {i}: Cost {cost_value.numpy():.2f}")

print("Training Completed!")

# Make Predictions
# Predictions (add mean back)
predictions = tf.matmul(w, X, transpose_b=True) + Y_mean

# FORCE values to be between 0 and 5
predictions = tf.clip_by_value(predictions, 0.0, 5.0)

# Display Results
print("\n--- Predictions for unseen items ---")
for i in range(num_users):
  for j in range(num_items):
    # Only print the prediction if the user hadn't rated it originally
    if Y[i, j] == 0:
      print(f"User {i} has not seen Item {j}. We predict rating: {predictions[i, j].numpy():.1f} ⭐")
