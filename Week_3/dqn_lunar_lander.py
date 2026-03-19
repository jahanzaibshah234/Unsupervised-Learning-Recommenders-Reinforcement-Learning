# Project: Deep Q-Network (DQN) for Lunar Lander

"""
Description: This script trains a Reinforcement Learning agent to solve the LunarLander-v3 environment 
using a Deep Q-Network. The goal of the agent is to safely land a spacecraft on a lunar landing pad 
by firing its main, left, and right engine thrusters. It uses an Experience Replay buffer to 
remember past flight data and the Bellman Equation to calculate future rewards, gradually 
training a TensorFlow Neural Network to predict the best sequence of engine burns.
"""

import gymnasium as gym
import numpy as np
import tensorflow as tf
import random
from collections import deque

# Init environment
env = gym.make("LunarLander-v3")

state_size = 8
action_size = 4

# Hyperparameters
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.975
learning_rate = 0.001
batch_size = 32

# Replay buffer (keeps latest 2000 steps)
memory = deque(maxlen=2000)

# Build Q-Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(state_size,)),
    tf.keras.layers.Dense(units=24, activation='relu'),
    tf.keras.layers.Dense(units=24, activation='relu'),
    tf.keras.layers.Dense(units=action_size, activation='linear')
])

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

# Training function (Experience Replay)
def replay():
  if len(memory) < batch_size:
    return

  batch = random.sample(memory, batch_size)

  for state, action, reward, next_state, done in batch:
    target = reward

    if not done:
      # Bellman Equation
      target = reward + gamma * np.max(model.predict(next_state, verbose=0))

    # Update the specific action's Q-value
    target_f = model.predict(state, verbose=0)
    target_f[0][action] = target

    # Train network
    model.fit(state, target_f, epochs=1, verbose=0)


# Training loop
for episode in range(200):

  state, _ = env.reset()
  state = np.reshape(state, [1, state_size])

  done = False

  while not done:
    # Epsilon-greedy (Explore vs Exploit)
    if np.random.rand() <= epsilon:
      action = env.action_space.sample()
    else:
      action = np.argmax(model.predict(state, verbose=0))

    # Take action in environment
    next_state, reward, done, truncated, _ = env.step(action)
    next_state = np.reshape(next_state, [1, state_size])

    # Save memory and move forward
    memory.append((state, action, reward, next_state, done))
    state = next_state

    replay()

  # Reduce exploration
  if epsilon > epsilon_min:
    epsilon *= epsilon_decay

  print(f"Episode {episode}, Epsilon {epsilon:.3f}")

print("Training complete")
