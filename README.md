# Prisoner's Dilemma -- Reinforcement Learning

## Overview
This project implements the classic **Prisoner's Dilemma** game using both definite strategy agents and a reinforcement learning (RL) agent. The goal is to hold a competition between different agents, where five agents follow definite strategies, and the sixth agent learns the optimal strategy through reinforcement learning.

The definite strategy agents include:
- Always cooperate
- Always betray
- Tit For Tat
- Deflect once for 3 betrayals
- Random action

The RL agent uses the Q-learning algorithm to learn and adapt its strategy based on the outcomes of previous actions.

## Organization
- **Agents**: Contains python files defining the dumb agents and the RL agent.
- **environment.py**: Contains code defining the game environment, 
- **game_env.py**: Implements the training loop and competition between agents.

## Requirements
- numpy
- pandas

## Usage
Once you've ensured the required packages are installed in your environment, go ahead and run the tournament.py file to find a line by line log of each agent's performance in each round of each epoch followed by a final score. The agent does win in most cases as expected.

