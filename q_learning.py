import numpy as np

# Set the discount factor (Gamma) for future rewards.
Gamma = 0.8
print("Gamma:", str(Gamma))

# Define the rewards matrix for the environment (6 states/rooms).
# Negative values (-1) indicate illegal or unfavorable moves,
# 0 indicates a neutral move, and 100 represents a goal reward.
Rewards_Matrix = np.array([
    [-1, -1, -1, -1,  0, -1],
    [-1, -1, -1,  0, -1, 100],
    [-1, -1, -1,  0, -1, -1],
    [-1,  0,  0, -1,  0, -1],
    [ 0, -1, -1,  0, -1, 100],
    [-1,  0, -1, -1,  0, 100]
])
print("Rewards Matrix:")
print(Rewards_Matrix)

# Initialize the Q-Matrix with zeros.
# This matrix will store the learned Q-values for each state-action pair.
Q_Matrix = np.zeros((6, 6), dtype=int)
print("Q-Matrix Initialization:")
print(Q_Matrix)

# Function to get the possible actions for a given state.
# It returns the indices of the actions that have a non-negative reward.
def Possible_actions(state):
    current_state_row = Rewards_Matrix[state]
    Pos_Acts_index = np.where(current_state_row >= 0)[0]
    return Pos_Acts_index

# -------- FIRST EXAMPLE: One Step of Q-Learning --------
print("FIRST EXAMPLE")

# Randomly select an initial state (from 0 to 5).
Init_State = np.random.randint(6)
print("The initial state:", str(Init_State))

# Get the list of possible actions for the initial state.
Possible_Actions = Possible_actions(Init_State)
print("The possible actions:", str(Possible_Actions))

# Choose a random action from the possible actions.
Next_Random_Action = np.random.choice(Possible_Actions)
print("The random selected action:", str(Next_Random_Action))

# Get possible actions for the state reached after taking the chosen action.
Next_Possible_Actions = Possible_actions(Next_Random_Action)

# Update the Q-value for the (initial state, selected action) pair.
# Q(s, a) = R(s, a) + Gamma * max[ Q(next_state, possible_actions(next_state)) ]
Q_Matrix[Init_State, Next_Random_Action] = Rewards_Matrix[Init_State, Next_Random_Action] + \
    Gamma * max(Q_Matrix[Next_Random_Action, Next_Possible_Actions])
print("The updated Q-matrix:")
print(Q_Matrix)

# -------------------- TRAINING PHASE --------------------
print("TRAINING RESULTS")
Range = 10000  # Number of training iterations

for i in range(Range):
    # Randomly select a current state.
    current_state = np.random.randint(6)
    
    # Choose a random allowed action from the current state.
    Next_Random_Action = np.random.choice(Possible_actions(current_state))
    
    # Find the maximum Q-value for the next state (over its allowed actions).
    max_Q_value = max(Q_Matrix[Next_Random_Action, Possible_actions(Next_Random_Action)])
    
    # Update the Q-value for the current state and selected action.
    Q_Matrix[current_state, Next_Random_Action] = \
        Rewards_Matrix[current_state, Next_Random_Action] + Gamma * max_Q_value

print("The trained Q-Matrix after", str(Range), "re-iterations:")
print(Q_Matrix)

# -------------------- TESTING PHASE --------------------
print("TESTING RESULTS")
current_state = 0  # Start testing from state 0
steps = [current_state]  # To record the sequence of states
print("Randomly selected initial state:", str(current_state))

# Continue selecting the best action (with highest Q-value) until the goal state (state 5) is reached.
while current_state != 5:
    # Find the indices of actions with the highest Q-value for the current state.
    next_step_index = np.where(Q_Matrix[current_state,] == np.max(Q_Matrix[current_state,]))[0]
    
    # If there are multiple optimal actions, randomly choose one.
    if next_step_index.shape[0] > 1:
        next_step_index = np.random.choice(next_step_index)
    else:
        next_step_index = int(next_step_index)
    
    # Append the next state to the path and update the current state.
    steps.append(next_step_index)
    current_state = next_step_index

print("Selected path:", str(steps))
