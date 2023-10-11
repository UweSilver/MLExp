from agent import Agent
import random
import numpy as np

class RandomAgent(Agent):
    def __init__(self, seed) -> None:
        random.seed(seed)
        super().__init__()
    def select_action(self, state):
        v = np.full(1, 0)
        return v
    
    def select_exploratory_action(self, state):
        v = np.full(1, random.random() - 0.5 * 4.)
        print(state)
        return v