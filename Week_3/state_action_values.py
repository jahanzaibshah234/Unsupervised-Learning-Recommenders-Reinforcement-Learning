# Project: State-Action Value Function Example

import numpy as np

states = ["A", "B"]
actions = ["Left", "Right"]

# Q-table
Q = np.array([
    [5, 8],   # state A
    [3, 6]    # state B
])

for i, state in enumerate(states):
  best_action_index = np.argmax(Q[i])

  print("State:", state)
  print("Q values:", Q[i])
  print("Best action:", actions[best_action_index])
  print()
  