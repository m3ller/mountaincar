import gym
import tensorflow as tf

class MountainCarLearner():
  # Environment doc: https://github.com/openai/gym/wiki/MountainCar-v0
  # Actions: [0]left push, [1]no push, [2]right pushA
  # Observations: [0]position, [1]velocity
  def __init__(self, environment):
    ACT_SPACE = 3
    OBS_SPACE = 2
    env = environment
 
  """ Policy network tries to learn the 'best' policy by trying to optimize
  over the predicted reward (as given by value_grad(..)).
  """
  def policy_grad(self):
      pass

  """ Value network learns to predict the reward for a given action.
  """
  def value_grad(self):
      pass

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
