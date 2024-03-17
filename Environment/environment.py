# here we define the environment for the game
import numpy as np

class PrisonersDilemma:
    def __init__(self):
        self.state = None
    
    def reset(self):
        # Initialize the state of the environment
        self.state = {
            'player_actions': [],
            'opponent_actions': []
        }
    
    def step(self, player_action, opponent_action):
        # Update the state based on the actions taken by the player and opponent
        self.state['player_actions'].append(player_action)
        self.state['opponent_actions'].append(opponent_action)
        
        # Calculate rewards based on actions
        reward_player, reward_opponent = self.calculate_rewards(player_action, opponent_action)
        
        # Return the next state, rewards, and whether the game is done
        return self.state, reward_player, reward_opponent, False
    
    def calculate_rewards(self, player_action, opponent_action):
        # Define the reward scheme for the Prisoner's Dilemma game
        reward_scheme = {
            (1, 1): (3, 3),   # Both cooperate
            (1, 0): (0, 5),   # Player cooperates, opponent defects
            (0, 1): (5, 0),   # Player defects, opponent cooperates
            (0, 0): (1, 1)    # Both defect
        }
        
        # Lookup rewards based on actions
        reward_player, reward_opponent = reward_scheme[(player_action, opponent_action)]
        
        return reward_player, reward_opponent
