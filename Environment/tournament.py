# This is where we hold the tournament.
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

# Initialize the RL agent
rl_agent = RLAgent(alpha=0.1, gamma=0.6, epsilon=0.1)

# Define the hyperparameters
num_episodes = 40
num_rounds_per_episode = 40

# Training loop
rl_agent_scores = []
opponent_scores = []
for episode in range(num_episodes):
    env.reset()
    episode_score_agent = 0
    episode_score_opponent = 0

    # Pair the RL agent with a definite strategy agent
    opponent_agent = np.random.choice([al_coop, al_bet, deflect, random, t4t])

    for round in range(num_rounds_per_episode):
    
        # Play the round
        rl_action = rl_agent.update_state(rl_agent.previous_action, opponent_agent.action)
        opponent_action = opponent_agent.update(opponent_agent.previous_action, rl_action)

        # Update the RL agent's Q-table
        next_state, reward_agent, reward_opponent, done = env.step(rl_action, opponent_action)
        state = rl_agent.get_state(rl_agent.opp_action, rl_agent.previous_action) # Get the state based on the previous actions
        rl_agent.learn(state, rl_action, reward_agent, next_state)

        # Print round information
        print(f"Round {round+1}: RL Agent chose {rl_action}, Opponent chose {opponent_action}, Reward: {reward_agent}, Opponent Reward: {reward_opponent}")

        episode_score_agent += reward_agent
        episode_score_opponent += reward_opponent
        print(f"Episode {episode+1} Score: {episode_score_agent}, Opponent Score: {episode_score_opponent}")
    
    # Print Q-table
    print("Q-table:")
    for state, row in enumerate(rl_agent.q_table):
        print(f"  State {state}: {row}")

    rl_agent_scores.append(episode_score_agent)
    opponent_scores.append(episode_score_opponent)
    rl_agent.update_epsilon(min_epsilon=0.01, decay_rate=0.001, episode=episode)

# Print final scores
print("Final scores:")
print("Agent score", sum(rl_agent_scores))
print("Cumulative Opponent score", sum(opponent_scores))
