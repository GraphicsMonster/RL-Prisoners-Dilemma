# This file contains the code for the tournament between the RL agent and the definite strategy agents.
import numpy as np
import sys
sys.path.append('../Agents')

from definite_agents import Always_Coop, Always_Betray, DeflectOnceFor3Betrayals, Random_Choice, TitforTat
from RLagent import RLAgent
from environment import PrisonersDilemma

# Initialize the environment
env = PrisonersDilemma()

# Initialize the agents
al_coop = Always_Coop()
al_bet = Always_Betray()
deflect = DeflectOnceFor3Betrayals()
random = Random_Choice()
t4t = TitforTat()

opponent_indexing = {
    'Always_Coop': 0,
    'Always_Betray': 1,
    'DeflectOnceFor3Betrayals': 2,
    'Random_Choice': 3,
    'TitforTat': 4
}

num_opponents = len(opponent_indexing)

# Initialize the RL agent
rl_agent = RLAgent(alpha=0.1, gamma=0.6, epsilon=0.4, num_opponents=num_opponents)

# Define the hyperparameters
num_episodes = 100
num_rounds_per_episode = 120

# TODO: organize the opponent choosing process in a more non_random way -- DONE

# Training loop
rl_agent_scores = []
opponent_scores = []
for episode in range(num_episodes):
    env.reset()
    episode_score_agent = 0
    episode_score_opponent = 0

    for opponent_agent in [al_coop, al_bet, deflect, random, t4t]: # Agent pairs up against each agent the same number of times -- Fairer distribution of chances of learning.
        opp_index = opponent_indexing[opponent_agent.__class__.__name__]
        print("opponent: ", opponent_agent.__class__.__name__)

        for round in range(num_rounds_per_episode):
        
            # Play the round
            rl_action = rl_agent.update_state(rl_agent.previous_action, opponent_agent.action, opp_index=opp_index)
            opponent_action = opponent_agent.update(opponent_agent.previous_action, rl_action)

            # Update the RL agent's Q-table
            next_state, reward_agent, reward_opponent, done = env.step(rl_action, opponent_action)
            state = rl_agent.get_state(rl_agent.opp_action, rl_agent.previous_action) # Get the state based on the previous actions
            rl_agent.learn(state, rl_action, reward_agent, next_state, opp_index=opp_index)

            # Print round information
            print(f"Round {round+1}: RL Agent chose {rl_action}, Opponent chose {opponent_action}, Reward: {reward_agent}, Opponent Reward: {reward_opponent}")

            episode_score_agent += reward_agent
            episode_score_opponent += reward_opponent
            print(f"Episode {episode+1} Score: {episode_score_agent}, Opponent Score: {episode_score_opponent}")

        # Print Q-table
        print("Q-table:")
        for state, row in enumerate(rl_agent.q_tables[opp_index]):
            print(f"  State {state}: {row}")

    rl_agent_scores.append(episode_score_agent)
    opponent_scores.append(episode_score_opponent)
    rl_agent.update_epsilon(min_epsilon=0.01, decay_rate=0.001, episode=episode)

# Print final scores
print("Final scores:")
print("Agent score", sum(rl_agent_scores))
print("Cumulative Opponent score", sum(opponent_scores))

difference_percentage = (sum(rl_agent_scores) - sum(opponent_scores)) / sum(opponent_scores) * 100
print(f"Agent performed {difference_percentage}% better than the opponents")
