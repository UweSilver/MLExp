import random_agent
import train
import eval_agent
import table_q_agent

from matplotlib import pyplot
import numpy as np

agent_seed = 0
eval_seed = 100

train_target = 10
train_episode_target = 100
eval_episode_count = 10

for train_count in range(0,train_target):
    agent = table_q_agent.TableQAgent(seed=train_count)
    for train_episode_count in range(0, train_episode_target):
        train.train_agent(agent=agent, seed=train_count, episode_count=1)
        agent.save_models("out\\models\\" + str(train_count) + "_" + str(train_episode_count))

results = []
for train_episode_count in range(0, train_episode_target):
    result = np.zeros(train_target * eval_episode_count)
    for train_count in range(0, train_target):
        agent = table_q_agent.TableQAgent(agent_seed + train_count)
        agent.load_models("out\\models\\" + str(train_count) + "_" + str(train_episode_count))

        r = eval_agent.eval_agent(agent=agent, eval_episode_count=eval_episode_count, eval_env_seed=eval_seed + train_count)
        for i in range(0, len(r)):
            result[train_count * 10 + i] = (float)(r[i])
    results.append(result)
    np.savetxt("out/results/" + "tableq_" + str(train_count) + "_" + str(train_episode_count), result)



pyplot.boxplot(results)
pyplot.show()