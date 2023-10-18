from agent import Agent
import random
import numpy as np
import math
cuberootK = 10 #
L = 1000
gamma = 0.9
alpha = 0.8
epsilon = 0.3

def state2idx(state):
    x = math.floor((state[0] / 2. + 0.5) * (cuberootK - 1.))
    y = math.floor((state[1] / 2. + 0.5) * (cuberootK - 1.))
    z = math.floor((state[2] / 16. + 0.5) * (cuberootK - 1.))
    idx = z * (cuberootK * cuberootK) + y * cuberootK + x
    # print("\nx, y, z, idx = " + str(x) + ", " + str(y) + ", " + str(z) + ", " + str(idx) + "\n")
    return idx

def action2idx(action):
    idx = math.floor(action[0] / 4 + 0.5 * (L - 1.))
    return int(idx)

def idx2action(idx):
    action = np.full(1, (1. / float(L) * float(idx) - 0.5) * 4.)
    return action

def maxQvalue(table, state):
    return table[state2idx(state)].max()

def maxAction(table, state):
    maxValue = np.finfo(np.float64).tiny
    maxIdx = 0
    for j in range(0, L):
        if maxValue < table[state2idx(state)][j]:
            maxValue = max(maxValue, table[state2idx(state)][j])
            maxIdx = j
    return idx2action(maxIdx)

class TableQAgent(Agent):
    def __init__(self, seed) -> None:
        random.seed(seed)
        np.random.seed(seed=seed)
        self.table = np.random.randn(cuberootK * cuberootK * cuberootK, L)
        print(str(self.table))
        super().__init__()
        pass

    def select_action(self, state):
        return maxAction(self.table, state)
    
    def select_exploratory_action(self, state):
        dice = random.random()
        if dice > epsilon:
            return self.select_action(state)
        else:
            idx = idx2action(math.floor(random.random() * float(L)))
            return idx
    
    def train(self, state, action, next_state, reward, done):
        delta = reward + gamma * maxQvalue(self.table, next_state) - self.table[state2idx(state)][action2idx(action)]
        self.table[state2idx(state)][action2idx(action)] = self.table[state2idx(state)][action2idx(action)] + alpha * delta

    def save_models(self, path):
        np.savetxt(path + '_tableq_' + str(cuberootK) + '_' + str(L), self.table)

    def load_models(self, path):
        self.table = np.loadtxt(path + '_tableq_' + str(cuberootK) + '_' + str(L))