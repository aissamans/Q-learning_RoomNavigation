# Q-learning_RoomNavigation

**Q-learning_RoomNavigation** is a Python project that demonstrates a basic Q-learning algorithm. It guides an agent through interconnected rooms by learning the optimal path using a reward matrix and iterative training.

---

## Overview

This repository contains:
- A Python script implementing a simple Q-learning algorithm.
- A reward matrix representing the environment with 6 states (rooms).
- Training logic that updates the Q-matrix to learn the optimal navigation path.
- A testing phase that uses the trained Q-matrix to determine the best route from a starting state to the goal.

---

## How It Works

1. **Environment Setup:**  
   The environment is modeled as 6 rooms with defined rewards for transitions. Positive rewards encourage the agent to take certain actions (like reaching a goal), while negative rewards penalize undesirable moves.

2. **Q-learning Algorithm:**  
   The algorithm updates the Q-matrix using the formula:  
   `Q(s, a) = R(s, a) + Gamma * max(Q(next_state, possible_actions))`  
   This helps the agent learn which actions yield the best cumulative reward.

3. **Training and Testing:**  
   - **Training:** The Q-matrix is iteratively updated over many episodes.
   - **Testing:** The agent uses the trained Q-matrix to determine the optimal path from a starting room to the goal.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/aissamans/Q-learning_RoomNavigation.git
   cd Q-learning_RoomNavigation

2. **Install Dependencies:** 
   - Ensure you have Python 3 installed. 
   - Then install the required packages: pip install numpy

## Usage

To run the Q-learning demonstration: python q_learning.py

The script will:
   - Initialize the Q-matrix.
   - Train the agent through 10,000 iterations.
   - Output the updated Q-matrix and the optimal path from the starting state to the goal state.

---

## Visuals

The repository includes diagrams and images that illustrate the reinforcement learning environment and the agent's navigation path. You can view them in the images/ directory.

## License

The code is available for personal/educational use.

## Contributing

Feel free to fork the repository and submit pull requests if you have improvements or additional features to suggest.

## Contact

For any questions or suggestions, please contact aissam.mansouri93@gmail.com.

---

Thank you for visiting my repository!
