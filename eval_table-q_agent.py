import random_agent
import train
import eval_agent
import table_q_agent
from tqdm import tqdm 
import multiprocessing

from matplotlib import pyplot
import numpy as np

agent_seed = 100
eval_seed = 12

train_target = 5
train_episode_target = int(500000 / 200)
eval_episode_count = 10
eval_episode_delta = int(train_episode_target / 10)


def train_once(train_number):
    agent = table_q_agent.TableQAgent(seed=train_number)
    for train_episode_count in tqdm(range(0, eval_episode_count)):
        agent.save_models("out\\models\\" + str(train_number) + "_" + str(train_episode_count * eval_episode_delta))
        train.train_agent(agent=agent, seed=train_number, episode_count=eval_episode_delta)

def main():
    print("start train")
    trainings = []
    for train_count in range(0,train_target):
        t = multiprocessing.Process(target=train_once, args=(train_count,))
        trainings.append(t)
    for t in trainings:
        t.start()
    for t in trainings:
        t.join()

    print("start evalate")
    results = []
    averages = []
    q25s = []
    q75s = []
    for train_episode_count in tqdm(range(0, train_episode_target)):
        if(train_episode_count % eval_episode_delta == 0):
            result = np.zeros(train_target * eval_episode_count )
            for train_count in range(0, train_target):
                agent = table_q_agent.TableQAgent(agent_seed + train_count)
                agent.load_models("out\\models\\" + str(train_count) + "_" + str(train_episode_count))
                r = eval_agent.eval_agent(agent=agent, eval_episode_count=eval_episode_count, eval_env_seed=eval_seed+train_count, draw=False)
                for i in range(0, len(r)):
                    result[train_count * eval_episode_count + i] = (float)(r[i])
            results.append(result)
            averages.append(np.mean(result))
            q75, q25 = np.percentile(result, [75, 25])
            q75s.append(q75)
            q25s.append(q25)
            np.savetxt("out/results/" + "tableq" + str(train_count) + "_" + str(train_episode_count), result)

    x = np.linspace(0, train_episode_target - eval_episode_delta, len(averages))
    pyplot.style.use('seaborn-whitegrid')
    pyplot.plot(x, averages)
    #pyplot.plot(x, results, 'o', color = 'black')
    pyplot.fill_between(x, q25s, q75s, alpha = 0.2)
    pyplot.show()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()