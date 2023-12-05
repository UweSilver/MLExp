from agent import Agent
import random
import numpy as np
import math

def state2idx(state, cuberootK):
    x = int((state[0] / 2. + 0.5) * (cuberootK))
    y = int((state[1] / 2. + 0.5) * (cuberootK))
    z = int((state[2] / 16. + 0.5) * (cuberootK))
    idx = z * (cuberootK * cuberootK) + y * cuberootK + x
    #print("\nx, y, z, idx = " + str(x) + ", " + str(y) + ", " + str(z) + ", " + str(idx))
    return idx

def action2idx(action, L):
    idx = math.floor((action[0] / 4 + 0.5) * (L - 1.))
    return int(idx)

def idx2action(idx, L):
    action = np.full(1, (1. / float(L) * float(idx) - 0.5) * 4.)
    #print("index = " + str(idx) + " action = " + str(action))
    return action

def maxQvalue(table, state, cuberootK):
    return table[state2idx(state, cuberootK)].max()

def maxAction(table, state, L, cuberootK):
    maxValue = np.finfo(np.float64).tiny
    maxIdx = 0
    for j in range(0, L):
        if maxValue < table[state2idx(state, cuberootK)][j]:
            maxValue = table[state2idx(state, cuberootK)][j]
            maxIdx = j
    #print("max action idx = " + str(maxIdx))
    return idx2action(maxIdx, L)

class TableQWithExpReplayAgent(Agent):
    def __init__(self, seed) -> None:
        random.seed(seed)
        np.random.seed(seed=seed)
        self.cuberootK = 3 #
        self.L = 9
        self.gamma = 0.99
        self.alpha = 0.0003
        self.epsilon = 0.05
        self.batch_size = 256
        self.exp_record = []
        self.table = np.random.randn(self.cuberootK * self.cuberootK * self.cuberootK, self.L)
        for k in range(0, self.cuberootK * self.cuberootK * self.cuberootK):
            for l in range(0, self.L):
                self.table[k][l] = self.table[k][l] * 0.000000008
        super().__init__()
        pass

    def select_action(self, state):
        action = maxAction(self.table, state, self.L, self.cuberootK)
        #print("select_action = " + str(action2idx(action, self.L)))
        return action
    
    def select_exploratory_action(self, state):
        dice = random.random()
        if dice > self.epsilon:
            action = self.select_action(state)
        else:
            index =  math.floor(random.random() * float(self.L))
            action = idx2action(index, self.L)

        #print("picked action index =" + str(action2idx(action, self.L)))
        return action
    
    def train(self, state, action, next_state, reward, done):
        self.exp_record.append((state, action, next_state, reward))
        #print("state idx = " + str(state2idx(state, self.cuberootK)) + " action idx = " + str(action2idx(action, self.L)))
        if len(self.exp_record) >= self.batch_size:
            for i in range(0, self.batch_size):
                h = random.randint(0, len(self.exp_record)-1)
                h_state, h_action, h_next_state, h_reward = self.exp_record[h]
                delta = h_reward + self.gamma * maxQvalue(self.table, h_next_state, self.cuberootK) - self.table[state2idx(h_state, self.cuberootK)][action2idx(h_action, self.L)]
                self.table[state2idx(h_state, self.cuberootK)][action2idx(h_action, self.L)] = self.table[state2idx(h_state, self.cuberootK)][action2idx(h_action, self.L)] + self.alpha * delta

    def save_models(self, path):
        np.savetxt(path + '_tableq_with_exp_replay_' + str(self.cuberootK) + '_' + str(self.L), self.table)

    def load_models(self, path):
        self.table = np.loadtxt(path + '_tableq_with_exp_replay_' + str(self.cuberootK) + '_' + str(self.L))
