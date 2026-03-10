# Project: Bellman Equation Example

import numpy as np

gamma = 0.9

reward = 5

future_q_values = [10, 8, 7]

best_future = np.max(future_q_values)

Q_value = reward + gamma * best_future

print("Reward:", reward)
print("Best future Q:", best_future)
print("Updated Q-value:", Q_value)