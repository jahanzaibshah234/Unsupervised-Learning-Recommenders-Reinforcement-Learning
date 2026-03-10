# Project: Policy Decision Example

import numpy as np

actions = ["left", "right", "Drill"]

# Expected rewards for each action
expected_rewards = [5, 10, 20]

best_action_index = np.argmax(expected_rewards)

print("Actions:", actions)
print("Expected rewards:", expected_rewards)
print("Best action:", actions[best_action_index])
