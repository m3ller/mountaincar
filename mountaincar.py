import gym
import numpy as np
import tensorflow as tf

class ReinforcementLearner():
    # Environment doc: https://github.com/openai/gym/wiki/MountainCar-v0
    # Actions: [0]left push, [1]no push, [2]right push
    # Observations: [0]position, [1]velocity
    def __init__(self, environment):
        self.env = environment
        self.n_act = environment.action_space.n
        self.n_obs = environment.observation_space.shape[0]
  
    #TODO: add tensorboard
    #TODO: batch the games
    """ Policy network tries to learn the 'best' policy by trying to optimize
    over the predicted reward (as given by value_grad(..)).
    """
    def policy_grad(self):
        # Build network
        observation = tf.placeholder(tf.float32, [1, self.n_obs])
        w = tf.get_variable("pg_weight", [self.n_obs, self.n_act])
        b = tf.get_variable("pg_bias", [1, self.n_act])
  
        logp = tf.matmul(observation, w) + b
        prob = tf.nn.softmax(logp)
        # Produces an action probability and selects with this distribution
        # return action, act_loss, optimizer
        return observation, prob
  
    """ Value network learns to predict the reward (i.e. "future worth") of
    a given action.
    """
    def value_grad(self):
        # return value, val_loss, reward, optimizer
        pass
  
    """ Go through one episode (i.e. game) of Mountain Car
    """
    def run_episode(self, sess, pg_obs, pg_prob):
        # Run policy_grad(observation) -> action.
        #   Maximize over predicted value of action
        # env.step(action) -> observation, reward.  Given from game environment
        # After game ends, update/improve value_grad(observation, reward).
        #   Keep track of geometrically summed value
        action = self.env.action_space.sample()  # initial action
        for _ in xrange(100):
            self.env.render()
            observation, reward, done, info = self.env.step(action)
            action_prob = sess.run(pg_prob, feed_dict={pg_obs: np.expand_dims(observation, 0)})
            action = np.argmax(action_prob)
    
            if done:
                break

  
def main():
    env = gym.make("MountainCar-v0")
    env.reset()

    # Build network
    learner = ReinforcementLearner(env)
    pg_obs, pg_prob = learner.policy_grad()
   
    # Run episode
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in xrange(5):
            learner.run_episode(sess, pg_obs, pg_prob)

if __name__ == "__main__":
    main()
