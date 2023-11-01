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
        return v
    
    def save_models(self, path):
        with open(path + '_random', encoding='utf-8', mode='w') as file:
            file.write("empty!")
        pass

    def load_models(self, path):
        return super().load_models(path)
    
    def get_signature_values(self):
        signature = []
        return signature.extend(super().get_signature_values())