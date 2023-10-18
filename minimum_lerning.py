from tqdm import tqdm
import gym
import numpy as np
import torch
import random

import random_agent
import table_q_agent
seed = 2
env = gym.make('Pendulum-v0')

agent = table_q_agent.TableQAgent(seed)
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
state = env.reset()
reward_sum = 0
for t in range(10000):
    env.render()
    action = agent.select_exploratory_action(state)
    next_state, reward, done, info = env.step(action)
    agent.train(state, action, next_state, reward, done)
    # print("reward: " + str(reward) + "\n")
    reward_sum = reward_sum + reward
    state = next_state
    if done:
        state = env.reset()
        print("sum: " + str(reward_sum))
        reward_sum = 0
env.close()

print(agent.table)