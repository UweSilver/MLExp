from tqdm import tqdm
import gym
import numpy as np
import torch
import random

import random_agent
seed = 2
env = gym.make('Pendulum-v0')

agent = random_agent.RandomAgent(seed)
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
state = env.reset()
for t in tqdm(range(10)):
    env.render()
    action = agent.select_exploratory_action(state)
    next_state, reward, done, info = env.step(action)
    agent.train(state, action, next_state, reward, done)
    state = next_state
    if done:
        state = env.reset()
env.close()