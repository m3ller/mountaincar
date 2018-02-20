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
  
    """ Value network learns to predict the EXPECTED reward of a given state.
    """
    def value_grad(self):
        # Build network
        observation = tf.placeholder(tf.float32, [None, self.n_obs])
        observed_value = tf.placeholder(tf.float32, [None])
        w = tf.get_variable("vg_weight", [self.n_obs, 1])
        b = tf.get_variable("vg_bias", [1])

        expected_value = tf.matmul(observation, w) + b
        diff = tf.abs(expected_value - observed_value)
        loss = tf.reduce_sum(diff)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        return observation, observed_value, optimizer
  
    """ Go through one episode (i.e. game) of Mountain Car
    """
    def run_episode(self, sess, pg_obs, pg_prob):
        observations = []
        actions = []
        rewards = []

        # Play through a game.  
        # Running policy_grad(observation) to produce actions in the game.
        observation = self.env.reset()   # initial observation
        for _ in xrange(100):
            self.env.render()
            action_prob = sess.run(pg_prob, feed_dict={pg_obs: np.expand_dims(observation, 0)})
            action = np.argmax(action_prob)
            new_observation, reward, done, info = self.env.step(action)

            # Store transistions and update
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            observation = new_observation
    
            if done:
                break

        return observations, actions, rewards

    """ Update the parameters in the policy and value networks.

    Update is dependent on the observed rewards from the previous game.
    """
    def update_param(self, sess, transition_tuple, pg_obs, pg_prob, vg_obs, vg_val, vg_optimizer):
        observations, actions, rewards = transition_tuple

        # Calculate observed value
        gamma = 0.8
        for i in xrange(2, len(rewards)+1):
            rewards[-i] += gamma * rewards[-i+1]    # Account for future reward

        # Update value_grad
        _ = sess.run(vg_optimizer, feed_dict={vg_obs: observations,
                                              vg_val: rewards})
        
  
def main():
    env = gym.make("MountainCar-v0")

    # Build network
    learner = ReinforcementLearner(env)
    pg_obs, pg_prob = learner.policy_grad()
    vg_obs, vg_val, vg_optimizer = learner.value_grad()
   
    # Run episode
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in xrange(5):
            transition_tuple = learner.run_episode(sess, pg_obs, pg_prob)
            learner.update_param(sess, transition_tuple, pg_obs, pg_prob, vg_obs, vg_val, vg_optimizer)

if __name__ == "__main__":
    main()
