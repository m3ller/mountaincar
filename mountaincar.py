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
  
    #TODO: batch the games
    """ Policy network tries to learn the 'best' policy by trying to optimize
    over the predicted reward (as given by value_grad(..)).
    """
    def policy_grad(self):
        # Build network
        observation = tf.placeholder(tf.float32, [None, self.n_obs], "pg_obs")

        n_hidden = 8
        w1 = tf.get_variable("pg_w1", [self.n_obs, n_hidden])
        b1 = tf.get_variable("pg_b1", [1, n_hidden])
        #w2 = tf.get_variable("pg_w2", [n_hidden, n_hidden])
        #b2 = tf.get_variable("pg_b2", [1, n_hidden])
        w3 = tf.get_variable("pg_w3", [n_hidden, self.n_act])
        b3 = tf.get_variable("pg_b3", [1, self.n_act])
        #w = tf.get_variable("pg_w", [self.n_obs, self.n_act])

        # Calculate probability
        temp_logp = tf.matmul(observation, w1) + b1
        temp_logp = tf.nn.relu(temp_logp)
        #temp_logp = tf.matmul(temp_logp, w2) + b2
        #temp_logp = tf.nn.relu(temp_logp)
        logp = tf.matmul(temp_logp, w3) + b3 + tf.constant([1., 1., 1.])
        #logp = tf.matmul(observation, w)
        prob = tf.nn.softmax(logp)

        # Update network parameters with advantage
        action = tf.placeholder(tf.float32, [None, self.n_act], "pg_act")
        advantage = tf.placeholder(tf.float32, [None], "pg_advantage")

        adjustment = tf.log(tf.reduce_sum(tf.multiply(prob, action), axis=1))
        #adjustment = tf.reduce_sum(tf.multiply(prob, action), axis=1)
        adjustment = tf.multiply(adjustment, advantage)  # BE WARY OF BROADCASTING
        loss = -tf.reduce_sum(adjustment, axis=0) #+ reg_constant * tf.reduce_sum(reg_losses)

        # optimizer
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

        # Store on TensorBoard
        smy_pg_loss = tf.summary.scalar("pg_loss", loss)
        smy_pg_prob = tf.summary.histogram("pg_prob", prob)
        smy_op = tf.summary.merge([smy_pg_loss, smy_pg_prob])

        return observation, prob, action, advantage, optimizer, smy_op
  
    """ Value network learns to predict the EXPECTED reward of a given state.
    """
    def value_grad(self):
        # Build network
        observation = tf.placeholder(tf.float32, [None, self.n_obs], "vg_obs")
        observed_value = tf.placeholder(tf.float32, [None, 1], "vg_value")

        n_hidden = 16
        w1 = tf.get_variable("vg_w1", [self.n_obs, n_hidden])
        b1 = tf.get_variable("vg_b1", [n_hidden])
        w2 = tf.get_variable("vg_w2", [n_hidden, n_hidden])
        b2 = tf.get_variable("vg_b2", [n_hidden])
        w3 = tf.get_variable("vg_w3", [n_hidden, 1])
        b3 = tf.get_variable("vg_b3", [1])

        temp_value = tf.matmul(observation, w1) + b1
        temp_value = tf.nn.relu(temp_value)
        temp_value = tf.matmul(temp_value, w2) + b2
        temp_value = tf.nn.relu(temp_value)
        expected_value = tf.matmul(temp_value, w3) + b3

        # Calculate and improve V(s) approximation
        diffs = observed_value - expected_value
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #reg_constant = 0.1
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # Calculate the advantage
        # TODO: May need to fiddle with this advantage
        #advantages = tf.exp(diffs)
        advantages = diffs

        # Store on TensorBoard
        smy_vg_loss = tf.summary.scalar("vg_loss", loss)
        smy_vg_advantage = tf.summary.histogram("vg_advantage", advantages)
        summary_op = tf.summary.merge([smy_vg_loss, smy_vg_advantage])
        return observation, observed_value, optimizer, advantages, summary_op
  
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
            action = np.random.choice(range(self.n_act), 1, p=action_prob[0])[0]
            new_observation, reward, done, info = self.env.step(action)

            # Store transistions and update
            observations.append(observation)
            actions.append(action)
            add_reward = observation[0]**2 * 10
            rewards.append(reward + add_reward)
            observation = new_observation[:]

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
    def update_param(self, sess, transition_tuple, pg_obs, pg_prob, pg_action, pg_advantage, pg_optimizer, pg_summary_loss, vg_obs, vg_val, vg_optimizer, vg_advantage, vg_summary_loss):
        observations, actions, rewards = transition_tuple

        # Calculate observed reward
        gamma = 0.9
        for i in xrange(2, len(rewards)+1):
            rewards[-i] += gamma * rewards[-i+1]    # Account for future reward

        # Calculate advantage
        advantages = sess.run(vg_advantage,\
                      feed_dict={vg_obs: observations, vg_val: rewards})

        # Update value_grad
        _, vg_loss = sess.run([vg_optimizer, vg_summary_loss],\
                      feed_dict={vg_obs: observations, vg_val: rewards})

        # Update policy_grad
        _, pg_loss = sess.run([pg_optimizer, pg_summary_loss],\
                      feed_dict={pg_obs: observations,\
                                 pg_action:actions,\
                                 pg_advantage: advantages[0]})
        return vg_loss, pg_loss
  
def main():
    env = gym.make("MountainCar-v0")
    #env = gym.make("CartPole-v0")

    # Build network
    learner = ReinforcementLearner(env)
    pg_obs, pg_prob, pg_act, pg_adv, pg_opt, pg_sumop = learner.policy_grad()
    vg_obs, vg_val, vg_opt, vg_adv, vg_sumop = learner.value_grad()
   
    # Run episode
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter("./tf_logs/", sess.graph)
        
        for i in xrange(3000):
            transition_tuple = learner.run_episode(sess, pg_obs, pg_prob)
            vg_summary, pg_summary = learner.update_param(sess, transition_tuple, pg_obs, pg_prob, pg_act, pg_adv, pg_opt, pg_sumop, vg_obs, vg_val, vg_opt, vg_adv, vg_sumop)

            #if i % 100 == 0:
            if i % 2 == 0:
                train_writer.add_summary(vg_summary, i)
                train_writer.add_summary(pg_summary, i)
        
        # Testing
        learner.run_episode(sess, pg_obs, pg_prob, True)

if __name__ == "__main__":
    main()
