# this file contains the code for the RL agent that learns the optimized strategy for the game using Q-Learning.
# As opposed to dead(stupid) agents which are hard coded to follow a definite strategy.

import numpy as np

# Reward Scheme and action encoding scheme
REWARD_SCHEME = {
    "both_cooperate": 3,  # Both players cooperate
    "both_defect": 1,    # Both players defect
    "cooperate_defect": 0,  # Player cooperates but the other defects
    "defect_cooperate": 5   # Player defects but the other cooperates
}

ACTION_ENCODING = {
    "defect": 0,
    "cooperate": 1
}

class RLAgent:
    def __init__(self, alpha, gamma, epsilon, num_opponents):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_tables = np.random.rand(num_opponents, *(4, 2))  # individual q tables dedicated to each opponent.
        self.action = np.random.choice([0, 1])  # A random pick between 0 and 1 for the first action
        self.rewards = []  # No rewards at the start
        self.points = 0  # No points at the start
        self.previous_action = self.action  # The previous action is the first action it takes
        self.opp_action = None  # The opponent's action is unknown at the start
        self.action_space = [0, 1]  # The action space is the set of all possible actions
        self.min_epsilon = 0.01  # Minimum value for epsilon
        self.decay_rate = 0.01  # Rate of decay for epsilon

    def choose_action(self, state, opp_index):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.q_tables[opp_index][state])  # state refers to the combined state (opp_action, previous_action)
        return action

    def update_state(self, opp_action, opp_index):
        self.previous_action = self.action
        self.opp_action = opp_action

        # Determine the current state based on the previous actions
        state = self.get_state(self.opp_action, self.previous_action)

        # Update the rewards list
        if opp_action != self.previous_action:
            if opp_action == ACTION_ENCODING["defect"]:
                self.rewards.append(REWARD_SCHEME["cooperate_defect"])
            else:
                self.rewards.append(REWARD_SCHEME["defect_cooperate"])
        else:
            if opp_action == ACTION_ENCODING["defect"]:
                self.rewards.append(REWARD_SCHEME["both_defect"])
            else:
                self.rewards.append(REWARD_SCHEME["both_cooperate"])

        self.points = sum(self.rewards)  # Update the points collected

        # Choose the next action based on the current state
        self.action = self.choose_action(state, opp_index)
        return self.action

    def learn(self, state, action, reward, next_state, opp_index):
        predict = self.q_tables[opp_index][state][action]
        env__state = next_state['player_actions'][-1] * 2 + next_state['opponent_actions'][-1] # handle the state of the environment adjusting for the q-table
        target = reward + self.gamma * np.max(self.q_tables[opp_index][env__state])
        self.q_tables[opp_index][state][action] += self.alpha * (target - predict)

    def update_epsilon(self, min_epsilon, decay_rate, episode):
        self.epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-decay_rate * episode)

    def get_state(self, opp_action, previous_action):
        """
        Returns the encoded state based on the opponent's action and the agent's previous action.
        This adjusts the state space to be 0, 1, 2, 3.
        """
        state = previous_action * 2 + opp_action # more like a binary encoding 00, 01, 10, 11
        return state
        