from tqdm import tqdm
import gym
import numpy as np
import torch
import random

import random_agent
seed = 5
model_seed = 2
train_episode = range(1,6)
test_episode_target = 10
env = gym.make('Pendulum-v0')

agent = random_agent.RandomAgent(seed)
env.seed(seed)
np.random.seed(seed)
random.seed(seed)
state = env.reset()

reward_sum = 0

for m in train_episode:
    agent.load_models('out\\models\\random_' + str(model_seed) + '_' + str(m))
    with open('out\\results\\random_' + str(model_seed) + '_' + str(m), mode="w", encoding="utf-8") as result_file:
                result_file.write("")
    for t in tqdm(range(200 * test_episode_target)):
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        reward_sum += reward
        state = next_state
        if done:
            state = env.reset()
            print(reward_sum)
            
            with open('out\\results\\random_' + str(model_seed) + '_' + str(m), mode="a", encoding="utf-8") as result_file:
                result_file.write(str(reward_sum) + "\n")
            reward_sum = 0
env.close()
