# Project: Mars Rover Return Simulator
# Simulate the Mars Rover example and compute returns for different policies.

import numpy as np

gamma = 0.9

# Rewards sequence for a mission
rewards = [1, 1, 1, 10] # explore, explore, explore, find water

# Calculate return
G = 0
for t in range(len(rewards)):
  G += (gamma**t) * rewards[t]

print("Rewards:", rewards)
print("Return (G):", G)