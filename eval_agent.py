from tqdm import tqdm
import gym
import numpy as np
import torch
import random

import random_agent
import table_q_agent

def eval_agent(agent, eval_episode_count, eval_env_seed):      
    env = gym.make('Pendulum-v0')
    env.seed(eval_env_seed)
    np.random.seed(eval_env_seed)
    random.seed(eval_env_seed)
    state = env.reset()
    results = []
    for e in range(0, eval_episode_count):
        reward_sum = 0
        while True:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                reward_sum += reward
                state = next_state
                if done:
                    state = env.reset()
                    results.append(reward_sum)
                    break
    env.close()
    return results