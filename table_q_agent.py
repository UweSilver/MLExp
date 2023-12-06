from agent import Agent
import random
import numpy as np
import math

def state2idx(state, cuberootK):
    x = int(math.floor((state[0] / 2. + 0.5) / (1.0/float(cuberootK))))
    y = int(math.floor((state[1] / 2. + 0.5) / (1.0/float(cuberootK))))
    z = int(math.floor((state[2] / 16. + 0.5) /  (1.0/float(cuberootK))))
    if x >= cuberootK:
        x = cuberootK-1
    if y >= cuberootK:
        y = cuberootK-1
    if z >= cuberootK:
        z = cuberootK-1
    idx = z * (cuberootK * cuberootK) + y * cuberootK + x
    return idx

def action2idx(action, L):
    idx = int(math.floor((action[0] / 4.0 + 0.5) / (1.0/float(L))))
    if idx >= L:
        idx = L - 1
    return int(idx)

def idx2action(idx, L):
    action = np.full(1, (1. / float(L) * float(idx) - 0.5) * 4. + 0.5 / float(L))
    return action

def maxQvalue(table, state, cuberootK):
    return table[state2idx(state=state,cuberootK=cuberootK)].max()

def maxQindex(table, state, cuberootK):
    return table[state2idx(state=state, cuberootK=cuberootK)].argmax()

def maxAction(table, state, L, cuberootK):
    maxIdx = maxQindex(table=table, state=state,cuberootK=cuberootK)
    return idx2action(maxIdx, L)

class TableQAgent(Agent):
    def __init__(self, seed) -> None:
        random.seed(seed)
        np.random.seed(seed=seed)
        self.cuberootK = 3 #
        self.L = 9
        self.gamma = 0.99
        self.alpha = 0.0003
        self.epsilon = 0.05
        self.table = np.random.randn(self.cuberootK * self.cuberootK * self.cuberootK, self.L)
        for k in range(0, self.cuberootK * self.cuberootK * self.cuberootK):
            for l in range(0, self.L):
                self.table[k][l] = self.table[k][l] * 0.00000001
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
