import gym
import pybullet_envs
env = gym.make('HumanoidBulletEnv-v0')
env.render(mode="human")
state=env.reset()
for t in range(1000):
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        break
env.close()