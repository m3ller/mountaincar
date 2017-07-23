import gym

env = gym.make("MountainCar-v0")
env.reset()

for _ in xrange (100):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        break
