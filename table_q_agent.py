from agent import Agent
import random
import numpy as np
import math

def state2idx(state, cuberootK):
    x = math.floor((state[0] / 2. + 0.5) * (cuberootK - 1.))
    y = math.floor((state[1] / 2. + 0.5) * (cuberootK - 1.))
    z = math.floor((state[2] / 16. + 0.5) * (cuberootK - 1.))
    idx = z * (cuberootK * cuberootK) + y * cuberootK + x
    # print("\nx, y, z, idx = " + str(x) + ", " + str(y) + ", " + str(z) + ", " + str(idx) + "\n")
    return idx

def action2idx(action, L):
    idx = math.floor(action[0] / 4 + 0.5 * (L - 1.))
    return int(idx)

def idx2action(idx, L):
    action = np.full(1, (1. / float(L) * float(idx) - 0.5) * 4.)
    # print("idx: " + str(idx) + " action: " + str(action))
    return action

def maxQvalue(table, state, cuberootK):
    return table[state2idx(state, cuberootK)].max()

def maxAction(table, state, L, cuberootK):
    maxValue = np.finfo(np.float64).tiny
    maxIdx = 0
    for j in range(0, L):
        if maxValue < table[state2idx(state, cuberootK)][j]:
            maxValue = max(maxValue, table[state2idx(state, cuberootK)][j])
            maxIdx = j

    #print("max Idx: " + str(maxIdx) + " maxValue: " + str(maxValue) + " state Idx: " + str(state2idx(state)) + " state: " + str(state))
    return idx2action(maxIdx, L)

class TableQAgent(Agent):
    def __init__(self, seed) -> None:
        random.seed(seed)
        np.random.seed(seed=seed)
        self.cuberootK = 3 #
        self.L = 10
        self.gamma = 0.99
        self.alpha = 0.0003
        self.epsilon = 0.05
        self.table = np.random.randn(self.cuberootK * self.cuberootK * self.cuberootK, self.L)
        super().__init__()
        pass

    def select_action(self, state):
        return maxAction(self.table, state, self.L, self.cuberootK)
    
    def select_exploratory_action(self, state):
        dice = random.random()
        if dice > self.epsilon:
            return self.select_action(state)
        else:
            idx = idx2action(math.floor(random.random() * float(self.L)), self.L)
            return idx
    
    def train(self, state, action, next_state, reward, done):
        delta = reward + self.gamma * maxQvalue(self.table, next_state, self.cuberootK) - self.table[state2idx(state, self.cuberootK)][action2idx(action, self.L)]
        self.table[state2idx(state, self.cuberootK)][action2idx(action, self.L)] = self.table[state2idx(state, self.cuberootK)][action2idx(action, self.L)] + self.alpha * delta

    def save_models(self, path):
        np.savetxt(path + '_tableq_' + str(self.cuberootK) + '_' + str(self.L), self.table)

    def load_models(self, path):
        self.table = np.loadtxt(path + '_tableq_' + str(self.cuberootK) + '_' + str(self.L))
