import gym
import tensorflow as tf

# Environment documentation: https://github.com/openai/gym/wiki/MountainCar-v0
# Actions: [0]left push, [1]no push, [2]right pushA
# Observations: [0]position, [1]velocity

def main():
    env = gym.make("MountainCar-v0")
    env.reset()
    
    for _ in xrange (100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    
        if done:
            break

if __name__ == "__main__":
    main()
