import gym
import tensorflow as tf

class MountainCarLearner():
  # Environment doc: https://github.com/openai/gym/wiki/MountainCar-v0
  # Actions: [0]left push, [1]no push, [2]right push
  # Observations: [0]position, [1]velocity
  def __init__(self, environment):
    ACT_SPACE = 3
    OBS_SPACE = 2
    env = environment

  #TODO: add tensorboard
  """ Policy network tries to learn the 'best' policy by trying to optimize
  over the predicted reward (as given by value_grad(..)).
  """
  def policy_grad(self):
      # return action, act_loss, optimizer
      pass

  """ Value network learns to predict the reward (i.e. "future worth") of
  a given action.
  """
  def value_grad(self):
      # return value, val_loss, reward, optimizer
      pass

  """ Go through one episode (i.e. game) of Mountain Car
  """
  def run_episode(self):
      # Run policy_grad(observation) -> action.
      #   Maximize over predicted value of action
      # env.step(action) -> observation, reward.  Given from game environment
      # After game ends, update/improve value_grad(observation, reward).
      #   Keep track of geometrically summed value
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
