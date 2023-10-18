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
np.random.seed(seed)
random.seed(seed)
state = env.reset()

# セーブファイルの命名規則
# out\models\seed_(train_episode_count)_type_agent-specific-data

train_episode_target = 50
train_episode_count = 1
for t in tqdm(range(200 * train_episode_target)):
    env.render()
    action = agent.select_exploratory_action(state)
    next_state, reward, done, info = env.step(action)
    agent.train(state, action, next_state, reward, done)
    state = next_state
    if done:
        agent.save_models('out\\models\\' + str(seed) + '_' + str(train_episode_count))
        state = env.reset()
        train_episode_count+=1
env.close()