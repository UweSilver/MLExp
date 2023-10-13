from matplotlib import pyplot
import numpy as np

agent_name = "random"
agent_seed = 2
agent_train_range = range(1, 6)
eval_count = 10

result = []

for i in agent_train_range:
    r = np.zeros(eval_count)
    with open("out\\results\\" + agent_name + "_" + str(agent_seed) + "_" + str(i), mode="r", encoding="utf-8") as file:
        for j in range(0, eval_count):
            r[j] = (float)(file.readline().split("\n")[0])

    result.append(r)
pyplot.boxplot(result)
pyplot.show()