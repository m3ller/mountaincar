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
    #TODO: figure out normalized way to calculate advantage
    #TODO: consider regularizing params
    """ Policy network tries to learn the 'best' policy by trying to optimize
    over the predicted reward (as given by value_grad(..)).
    """
    def policy_grad(self):
        # Build network
        observation = tf.placeholder(tf.float32, [1, self.n_obs], "pg_obs")
        w = tf.get_variable("pg_weight", [self.n_obs, self.n_act])
        b = tf.get_variable("pg_bias", [1, self.n_act])

        # Calculate probability
        logp = tf.matmul(observation, w) + b
        prob = tf.nn.softmax(logp)

        # Update network parameters
        action = tf.placeholder(tf.float32, [self.n_act], "pg_act")
        advantage = tf.placeholder(tf.float32, [1], "pg_advantage")

        old_prob = tf.subtract((action + 1), 2*action)
        adjustment = tf.multiply(prob, action) * advantage  # BE WARY OF BROADCASTING
        adj_prob = tf.add(old_prob, adjustment) 
        loss = -tf.reduce_sum(tf.abs(adj_prob))
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # Produces an action probability and selects with this distribution
        # return action, act_loss, optimizer
        return observation, prob, action, advantage, optimizer
  
    """ Value network learns to predict the EXPECTED reward of a given state.
    """
    def value_grad(self):
        # Build network
        observation = tf.placeholder(tf.float32, [None, self.n_obs], "vg_obs")
        observed_value = tf.placeholder(tf.float32, [None, 1], "vg_value")
        w = tf.get_variable("vg_weight", [self.n_obs, 1])
        b = tf.get_variable("vg_bias", [1])

        # Calculate and improve V(s) approximation
        expected_value = tf.matmul(observation, w) + b
        diff = expected_value - observed_value
        loss = tf.reduce_sum(tf.abs(diff))
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # Calculate the advantage
        # TODO: May need to fiddle with this advantage
        advantage = tf.exp(diff)

        return observation, observed_value, optimizer, advantage
  
    """ Go through one episode (i.e. game) of Mountain Car
    """
    def run_episode(self, sess, pg_obs, pg_prob, render_flag=False):
        observations = []
        actions = []
        rewards = []

        # Play through a game.  
        # Running policy_grad(observation) to produce actions in the game.
        observation = self.env.reset()   # initial observation
        for _ in xrange(200):
            if render_flag:
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

        # Convert transition lists to arrays
        observations = np.array(observations)
        rewards = np.expand_dims(np.array(rewards), 1)

        # Convert action choices to one-hots
        len_act = len(actions)
        action_onehot = np.zeros((len_act, self.n_act))
        action_onehot[range(len_act), actions] = 1

        return observations, action_onehot, rewards

    """ Update the parameters in the policy and value networks.

    Update is dependent on the observed rewards from the previous game.
    """
    def update_param(self, sess, transition_tuple, pg_obs, pg_prob, pg_action, pg_advantage, pg_optimizer, vg_obs, vg_val, vg_optimizer, vg_advantage):
        observations, actions, rewards = transition_tuple

        # Calculate observed value
        gamma = 0.9
        for i in xrange(2, len(rewards)+1):
            rewards[-i] += gamma * rewards[-i+1]    # Account for future reward

        # Update value_grad
        _, advantages = sess.run([vg_optimizer, vg_advantage], feed_dict={vg_obs: observations,
                                              vg_val: rewards})
       
        # Update policy_grad
        for (observation, action, advantage) in zip(observations, actions, advantages):
            _ = sess.run(pg_optimizer, feed_dict={pg_obs: np.expand_dims(observation, 0), pg_action: action, pg_advantage: advantage})
  
def main():
    env = gym.make("MountainCar-v0")

    # Build network
    learner = ReinforcementLearner(env)
    pg_obs, pg_prob, pg_action, pg_advantage, pg_optimizer = learner.policy_grad()
    vg_obs, vg_val, vg_optimizer, vg_advantage = learner.value_grad()
   
    # Run episode
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in xrange(500):
            if i % 10 == 0:
                print i

            transition_tuple = learner.run_episode(sess, pg_obs, pg_prob)
            learner.update_param(sess, transition_tuple, pg_obs, pg_prob, pg_action, pg_advantage, pg_optimizer, vg_obs, vg_val, vg_optimizer, vg_advantage)

        # Testing
        learner.run_episode(sess, pg_obs, pg_prob, True)

if __name__ == "__main__":
    main()
