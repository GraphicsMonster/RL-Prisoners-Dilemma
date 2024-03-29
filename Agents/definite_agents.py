# We are going to implement a prisoner's dillema situation.
# We will hold a competition between different agents 5 of which will have definite strategies whilst the 6th one will be a RL agent that will learn the best possible strategy.
# Here we will implement the definite strategies.
# The strategies will be: 
# 1. Always cooperate
# 2. Tit for Tat
# 3. Deflect once for every 3 betrayals
# 4. Always betray
# 5. Random choice -- The agent randomly decides whether to betray or cooperate

import pandas as pd
import numpy as np

# Reward Scheme and action encoding scheme
REWARD_SCHEME = {
    "both_cooperate": 3,  # Both players cooperate
    "both_defect": 1,    # Both players defect
    "cooperate_defect": 0,  # Player cooperates but the other defects
    "defect_cooperate": 5   # Player defects but the other cooperates
}

ACTION_ENCODING = {
    "defect":  0,
    "cooperate":  1
}


class Always_Coop:
    def __init__(self, previous_action=None, opp_action=None, action=None, rewards=[]):
        self.previous_action = 1 # always cooperate
        self.opp_action = opp_action
        self.action = 1 # always cooperate
        self.rewards = rewards
        self.points = sum(self.rewards)
         

    def update(self, opp_action):
        self.opp_action = opp_action # update the opp_action
        self.previous_action = self.action
        self.action = 1  # always cooperate

        if opp_action != self.previous_action:
            if opp_action == 0:
                self.rewards.append(REWARD_SCHEME["cooperate_defect"])  # player cooperates but the other defects
            else:
                self.rewards.append(REWARD_SCHEME["defect_cooperate"])  # player defects but the other cooperates
        else: 
            if opp_action == 0:
                self.rewards.append(REWARD_SCHEME["both_defect"])  # both defect
            else: 
                self.rewards.append(REWARD_SCHEME["both_cooperate"])  # both cooperate

        self.points = sum(self.rewards)
        return self.action
    
class TitforTat:
    def __init__(self, previous_action=None, opp_action=None, action=None, rewards=[]):
        self.previous_action = np.random.choice([0, 1]) # random choice for the sake of the first move since there was no previous move but the update method requires one.
        self.opp_action = opp_action
        self.action = 1 # start by cooperating
        self.rewards = rewards
        self.points = sum(self.rewards)
    
    def update(self, opp_action): # update the state of the agent given a move made by the opponent
        self.opp_action = opp_action # update the opp_action
        self.previous_action = self.action # update the previous action
        self.action = self.opp_action # the agent will always copy the move made by the opponent in the previous round
        

        if opp_action!= self.previous_action:
                    if opp_action == 0:
                        self.rewards.append(REWARD_SCHEME["cooperate_defect"]) # player cooperates but the other defects
                    else:
                        self.rewards.append(REWARD_SCHEME["defect_cooperate"]) # player defects but the other cooperates
        else: 
                    if(opp_action == 0):
                        self.rewards.append(REWARD_SCHEME["both_defect"]) # both defect
                    else: 
                        self.rewards.append(REWARD_SCHEME["both_cooperate"]) # both cooperate
        
        self.points = sum(self.rewards)
        return self.action

class DeflectOnceFor3Betrayals:
    def __init__(self, previous_action=None, opp_action=None, action=None, rewards=[], betrayals=0, points=0):
        self.previous_action = np.random.choice([0, 1]) # only for the sake of the first round.
        self.opp_action = opp_action
        self.action = 1 # start by cooperating
        self.rewards = rewards
        self.betrayals = betrayals
        self.points = points
    
    def update(self, opp_action):
        self.previous_action = self.action
        self.opp_action = opp_action
        if(opp_action == 0):
            self.betrayals += 1
        
        self.action = self.act(self.betrayals)
        if opp_action!= self.previous_action:  # update the rewards based on the action taken by the opponent
                    if opp_action == 0:
                        self.rewards.append(REWARD_SCHEME["cooperate_defect"])
                    else:
                        self.rewards.append(REWARD_SCHEME["defect_cooperate"])
        else:
                    if(opp_action == 0):
                        self.rewards.append(REWARD_SCHEME["both_defect"])
                    else:
                        self.rewards.append(REWARD_SCHEME["both_cooperate"])

        self.points = sum(self.rewards)
        return self.action

    def act(self, betrayals): # act based on the number of betrayals. If 3 betrayals have been made, then the agent will defect
        if betrayals%3 == 0:
            self.betrayals = 0
            return 0
        else:
            return 1
    
class Always_Betray:
        def __init__(self, previous_action=None, opp_action=None, action=None, rewards=[], points=0):
            self.previous_action = 0 # always defect
            self.opp_action = opp_action
            self.action = 0  # always defect
            self.rewards = rewards
            self.points = points

        def update(self, opp_action):
            self.previous_action = self.action
            self.opp_action = opp_action
            self.action = 0  # always defect
            if opp_action!= self.previous_action:        # standard update of rewards
                    if opp_action == 0:
                        self.rewards.append(REWARD_SCHEME["cooperate_defect"])
                    else:
                        self.rewards.append(REWARD_SCHEME["defect_cooperate"])
            else: 
                    if(opp_action == 0):
                        self.rewards.append(REWARD_SCHEME["both_defect"])
                    else: 
                        self.rewards.append(REWARD_SCHEME["both_cooperate"])

            self.points = sum(self.rewards)
            return self.action
class Random_Choice:
        def __init__(self, previous_action=None, opp_action=None, action=None, rewards=[], points=0):
            self.previous_action = np.random.choice([0, 1])
            self.opp_action = opp_action
            self.action = np.random.choice([0,1])  # randomly choose to defect or cooperate
            self.rewards = rewards
            self.points = points

        def update(self, opp_action):  
            self.previous_action = self.action
            self.opp_action = opp_action
            self.action = np.random.choice([0,1])   # randomly choose to defect or cooperate
            if opp_action!= self.previous_action:       # standard update of rewards
                    if opp_action == 0:
                        self.rewards.append(REWARD_SCHEME["cooperate_defect"])
                    else:
                        self.rewards.append(REWARD_SCHEME["defect_cooperate"])
            else: 
                    if(opp_action == 0):
                        self.rewards.append(REWARD_SCHEME["both_defect"])
                    else: 
                        self.rewards.append(REWARD_SCHEME["both_cooperate"])

            self.points = sum(self.rewards)
            return self.action

