# Project: Deep Learning Content-Based Recommender
# Uses Neural Networks to learn complex user and item representations.

"""Description: A Deep Learning Recommender utilizing a Two-Tower architecture.
It passes User Features and Item Features through separate dense neural networks,
applies Unit Normalization the outputs, and computes their dot product to predict scaled ratings."""

# Import required libraries
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# DATA
Y = np.array([
    [5, 4, 0, 0, 3],
    [0, 0, 4, 3, 0],
    [5, 0, 0, 4, 4],
    [0, 3, 5, 0, 0]
], dtype=np.float32)

# Get dimensions from Y automatically
num_users, num_items = Y.shape
num_item_features = 6
num_user_features = 6

# Item Features & User Features (Fixed Facts)
item_features = np.random.rand(num_items, num_item_features).astype(np.float32)
user_features = np.random.rand(num_users, num_user_features).astype(np.float32)

# PREPARE DATA FOR NEURAL NETWORK
user_train = []
item_train = []
y_train = []

for u in range(num_users):
    for i in range(num_items):
        if Y[u, i] > 0:
            user_train.append(user_features[u])
            item_train.append(item_features[i])
            y_train.append(Y[u, i])


user_train = np.array(user_train)
item_train = np.array(item_train)
y_train = np.array(y_train)

# MINMAX SCALER
scalerTarget = MinMaxScaler(feature_range=(-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train_scaled = scalerTarget.transform(y_train.reshape(-1, 1)).flatten()

# BUILD THE TWO-TOWER NEURAL NETWORK
num_outputs = 32
tf.random.set_seed(42)

# Tower 1: User Brain
user_NN = tf.keras.models.Sequential([
   tf.keras.layers.Dense(units=256, activation='relu'),
   tf.keras.layers.Dense(units=128, activation='relu'),
   tf.keras.layers.Dense(units=num_outputs, activation='linear')
])

# Tower 2: Item Brain
item_NN = tf.keras.models.Sequential([
   tf.keras.layers.Dense(units=256, activation='relu'),
   tf.keras.layers.Dense(units=128, activation='relu'),
   tf.keras.layers.Dense(units=num_outputs, activation='linear')
])

# Create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features,))
vu = user_NN(input_user)
vu = tf.keras.layers.UnitNormalization(axis=1)(vu)

# Create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features,))
vm = item_NN(input_item)
vm = tf.keras.layers.UnitNormalization(axis=1)(vm)

# Compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# Specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

# COMPILE & TRAIN
cost_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)

model.summary()

print("Training Deep Two-Tower Model...")
model.fit([user_train, item_train], y_train_scaled, epochs=30, verbose=1)
print("Training Complete!")

# Recommendations
print("\n--- Recommendations ---")
for u in range(num_users):
    user_input = np.tile(user_features[u], (num_items, 1))

    scaled_scores = model.predict(
        [user_input, item_features],
        verbose=0
    )

    real_scores = scalerTarget.inverse_transform(scaled_scores).flatten()

    ranked_items = np.argsort(real_scores)[::-1]
    print(f"User {u} recommended items:", ranked_items)

# SIMILARITY ANALYSIS
print("\n--- Item Similarity Analysis ---")
learned_item_embeddings = item_NN.predict(item_features, verbose=0)
similarity = cosine_similarity(learned_item_embeddings)

similar_indices = np.argsort(similarity[0])[::-1]
print(f"Items most similar to Item 0 (Based on AI's learned features): {similar_indices}")
print(f"(Scores: {similarity[0][similar_indices]})")
