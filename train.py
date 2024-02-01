from tqdm import tqdm
import gym
import numpy as np
import random

import random_agent
import table_q_agent

def train_agent(agent, seed, episode_count):
    env = gym.make('Pendulum-v0')
    env.seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    state = env.reset()
    for e in range(0, episode_count):
        while True:
            # env.render()
            action = agent.select_exploratory_action(state)
            next_state, reward, done, info = env.step(action)
            agent.train(state, action, next_state, reward, done)
            state = next_state
            if done:
                env.seed(seed)
                state = env.reset()
                break
    env.close()